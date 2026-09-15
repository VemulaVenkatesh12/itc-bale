"""
Extracts a labeling-ready frame set from the two real ITC recordings.

This is step 4 of `README.md`'s "Getting to a real Milestone 1" (collect and
label real footage), operating on the 2026-08-18 recordings rather than on
newly captured footage - see the caveats below before trusting a model
trained from this.

Two things this does that a plain `ffmpeg -vf fps=1` does NOT, both of which
matter for a dataset cut from continuous CCTV video:

1. **Near-duplicate rejection.** Long stretches of this footage are static
   (a parked truck, an idle belt). Uniform sampling would spend most of the
   labeling budget on many copies of the same frame, which inflates the
   dataset without teaching the model anything. Each candidate is compared
   against the last KEPT frame (downsampled grayscale, mean absolute
   difference) and dropped if it is too similar.

2. **Time-blocked train/valid split, not random.** Adjacent frames of a
   video are nearly identical, so a random split puts near-copies of the
   same moment in both train and valid - the validation mAP then measures
   memorization and reads far too high. The timeline is instead cut into
   blocks and the tail of each block is reserved for valid, so the two sets
   are temporally disjoint while both still span the whole session.

The output is IMAGES ONLY - `_annotations.coco.json` has to come from
labeling them (see DATA_COLLECTION.md). Frames are written at native
resolution: RF-DETR trains at 640, but labeling (and the small, distant
crane hook in particular) benefits from the full detail.

Usage:
    py extract_training_frames.py --video ..\\192.168.1.101_...mp4 \\
        --prefix cam101 --every 20 --out ..\\dataset_real_itc

Known limits of this specific footage (from the recording analysis):
  - cam .102 actually runs at 12.5 fps, not the 25 fps its container
    claims, so `--every` must be set from the real rate, not from
    `CAP_PROP_FPS` (which reports the container's wrong value).
  - Both cameras recorded a single evening session under artificial light.
    A model trained only on this will not have seen daylight conditions.
  - The two views overlap only marginally around the hook, so this dataset
    supports DETECTION work only - it is not sufficient to validate
    triangulation.
"""
import argparse
import csv
import os

import cv2
import numpy as np


def frame_signature(bgr, size=(64, 36)):
    """Cheap perceptual signature: downsampled grayscale, as float for diffing."""
    small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def collect_candidates(video_path, every, min_diff, max_frames):
    """Walks the video, returning [(frame_index, pos_msec, image)] for kept frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")

    kept = []
    last_sig = None
    frame_index = -1
    skipped_similar = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % every != 0:
            continue

        sig = frame_signature(frame)
        if last_sig is not None:
            if float(np.mean(np.abs(sig - last_sig))) < min_diff:
                skipped_similar += 1
                continue

        # POS_MSEC is read AFTER the grab, so it refers to the frame just read.
        kept.append((frame_index, cap.get(cv2.CAP_PROP_POS_MSEC), frame))
        last_sig = sig
        if max_frames and len(kept) >= max_frames:
            break

    total_read = frame_index + 1
    cap.release()
    return kept, total_read, skipped_similar


def assign_splits(count, blocks, valid_fraction):
    """Time-blocked split: the tail `valid_fraction` of each block goes to valid."""
    splits = []
    for i in range(count):
        block = min(int(i * blocks / count), blocks - 1)
        block_start = int(np.ceil(block * count / blocks))
        block_end = int(np.ceil((block + 1) * count / blocks))
        block_len = max(block_end - block_start, 1)
        pos_in_block = (i - block_start) / block_len
        splits.append("valid" if pos_in_block >= (1.0 - valid_fraction) else "train")
    return splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--prefix", required=True, help="filename prefix, e.g. cam101")
    p.add_argument("--out", required=True, help="dataset root; train/ and valid/ are created under it")
    p.add_argument("--every", type=int, required=True,
                   help="sample every Nth frame. Set from the REAL fps, not the container's claim")
    p.add_argument("--min-diff", type=float, default=2.5,
                   help="mean abs grayscale difference vs last kept frame (0-255) below which a frame is dropped")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--blocks", type=int, default=10, help="time blocks for the train/valid split")
    p.add_argument("--valid-fraction", type=float, default=0.2)
    p.add_argument("--jpeg-quality", type=int, default=92)
    args = p.parse_args()

    kept, total_read, skipped_similar = collect_candidates(
        args.video, args.every, args.min_diff, args.max_frames
    )
    if not kept:
        raise SystemExit("no frames kept - check --every / --min-diff")

    splits = assign_splits(len(kept), args.blocks, args.valid_fraction)
    for split in ("train", "valid"):
        os.makedirs(os.path.join(args.out, split), exist_ok=True)

    manifest_path = os.path.join(args.out, f"manifest_{args.prefix}.csv")
    counts = {"train": 0, "valid": 0}
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "split", "source_video", "frame_index", "pos_msec"])
        for (frame_index, pos_msec, image), split in zip(kept, splits):
            name = f"{args.prefix}_{frame_index:06d}.jpg"
            cv2.imwrite(
                os.path.join(args.out, split, name),
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
            )
            writer.writerow([name, split, os.path.basename(args.video), frame_index, f"{pos_msec:.1f}"])
            counts[split] += 1

    print(f"{args.prefix}: read {total_read} frames, "
          f"sampled every {args.every}, dropped {skipped_similar} near-duplicates")
    print(f"{args.prefix}: wrote {counts['train']} train + {counts['valid']} valid -> {args.out}")
    print(f"{args.prefix}: manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
