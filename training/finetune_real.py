"""
Fine-tunes the detector on REAL ITC footage (dataset_real_itc), starting
from RF-DETR's own pretrained weights - no simulation/Blender lineage.

This is the "copy finetune.py and repoint it" step from
`real_deploy/DATA_COLLECTION.md` - kept as a separate file so the working
simulation training script (`finetune.py`) stays untouched.

Two deliberate differences from `finetune.py`:

1. **Two classes, not three.** `finetune.py` trains bale/crane_hook/
   crane_spike because the Blender/Three.js renders modelled the spike as
   its own object. In the real 2026-08-18 footage the spike is not a
   visually separable thing - the end effector reads as one integrated blue
   assembly - so labeling it as a third class would mean annotators guessing
   at a boundary that isn't there, which teaches the model noise.
   DATA_COLLECTION.md explicitly allows dropping it; `num_classes` and
   `class_names` below both reflect that and must stay in sync with however
   the dataset was actually labeled.

2. **Fewer default epochs.** `training/README.md` advises a short first pass
   on a small real dataset before committing to a long run; 40 epochs on
   ~400 images mostly overfits. Check mAP from a 15-epoch pass first.

No `--checkpoint` is passed to `RFDETRNano` by default, so it downloads/loads
its own stock pretrained weights for this model size and fine-tunes from
there - deliberately no continuity with the old simulation-trained
checkpoint. Pass `--checkpoint` explicitly if you ever want to continue from
a specific `.pth` instead.

Usage:
    py finetune_real.py --epochs 15 --batch-size 8

Caveat carried over from the recording analysis: dataset_real_itc comes from
a single evening session under artificial light, and the two camera views
overlap only marginally around the hook. A model trained here is a DETECTION
result - it does not validate triangulation or generalize to daylight.
"""
import argparse
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset_real_itc")
OUTPUT_DIR = os.path.join(REPO_ROOT, "training", "output_real")

CLASS_NAMES = ["bale", "crane_hook"]


def _batch_size(value):
    if value == "auto":
        return value
    return int(value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=_batch_size, default=8,
                   help="Integer micro-batch, or 'auto' to probe free GPU memory and "
                        "pick a safe micro-batch + grad_accum_steps to hit the effective batch target.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resolution", type=int, default=640)
    p.add_argument("--dataset-dir", default=DATASET_DIR)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--checkpoint", default=None,
                   help="Optional .pth to continue from. Omitted by default so RFDETRNano "
                        "starts from its own stock pretrained weights, not any simulation checkpoint.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-index", type=int, default=0,
                   help="Physical GPU index via CUDA_VISIBLE_DEVICES (0 = RTX 5060 Ti on this machine)")
    args = p.parse_args()

    for split in ("train", "valid"):
        ann = os.path.join(args.dataset_dir, split, "_annotations.coco.json")
        if not os.path.exists(ann):
            raise SystemExit(
                f"missing {ann}\n"
                "The extracted frames still have to be LABELED - see "
                "real_deploy/DATA_COLLECTION.md. Export COCO from Roboflow/CVAT "
                "into dataset_real_itc/{train,valid}/ and re-run."
            )

    if args.device.startswith("cuda"):
        # Same rfdetr device-normalization bug worked around in finetune.py:
        # "cuda:N" makes it build a list where a string is later expected.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    if args.device.startswith("cuda"):
        print(f"training device: {torch.cuda.get_device_name(0)} (physical index {args.gpu_index})")
    else:
        print(f"training device: {args.device}")

    print(f"base checkpoint: {args.checkpoint or 'RFDETRNano stock pretrained weights (no --checkpoint given)'}")
    from rfdetr import RFDETRNano
    kwargs = {"resolution": args.resolution, "num_classes": len(CLASS_NAMES)}
    if args.checkpoint:
        kwargs["pretrain_weights"] = args.checkpoint
    model = RFDETRNano(**kwargs)

    print(f"fine-tuning on {args.dataset_dir} -> {args.output_dir} "
          f"({args.epochs} epochs, batch size {args.batch_size}, classes {CLASS_NAMES})")
    model.train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        class_names=CLASS_NAMES,
    )
    print(f"\ndone - checkpoints under {args.output_dir}")


if __name__ == "__main__":
    main()
