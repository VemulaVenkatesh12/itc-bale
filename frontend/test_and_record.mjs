import { chromium } from 'playwright';

const results = { errors: [], apiCalls: [], notes: [] };

// ---- Test 1: 2D Simulator full flow ----
{
  const browser = await chromium.launch({ args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  page.on('console', msg => { if (msg.type() === 'error') results.errors.push('[2D] ' + msg.text()); });
  page.on('pageerror', err => results.errors.push('[2D] pageerror: ' + err.message));
  page.on('response', async res => {
    if (res.url().includes('/api/')) results.apiCalls.push(`[2D] ${res.status()} ${res.url().split('/api/')[1]}`);
  });

  await page.goto('http://localhost:5173', { waitUntil: 'load', timeout: 30000 });
  await page.waitForTimeout(1000);
  await page.setInputFiles('input[type=file]', '/home/nsp/Desktop/ITC_bale_unloading_simulator/dataset_real_itc/train/cam101_a_004760.jpg');
  await page.waitForTimeout(2000);

  await page.click('button:has-text("Mark detection area")');
  const canvas = page.locator('.canvas-wrap, canvas, img').first();
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + 20, box.y + 20);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width - 20, box.y + box.height - 20, { steps: 10 });
  await page.mouse.up();
  await page.waitForTimeout(500);

  await page.click('button:has-text("Mark drop area")');
  await page.mouse.move(box.x + 30, box.y + 30);
  await page.mouse.down();
  await page.mouse.move(box.x + 150, box.y + 150, { steps: 10 });
  await page.mouse.up();
  await page.waitForTimeout(500);

  // The freshly-trained model (15 epochs from scratch) is less confident than
  // the old checkpoint - lower the threshold from the 0.6 default so results
  // are actually visible, matching the Live 3D tab's own 0.3 default.
  const thresholdInput = page.locator('label:has-text("Confidence threshold") input');
  await thresholdInput.fill('0.3');
  results.notes.push('lowered 2D confidence threshold to 0.3 (fresh model has lower confidence than old checkpoint)');

  await page.click('button:has-text("Simulate")');
  try {
    await page.waitForSelector('.simulate-btn:not(:has-text("Simulating"))', { timeout: 25000 });
    results.notes.push('2D simulate completed');
  } catch (e) {
    results.notes.push('2D simulate TIMED OUT');
  }
  await page.screenshot({ path: '/home/nsp/Desktop/ITC_bale_unloading_simulator/latest_model_2d_result.png' });
  results.notes.push('2D body snippet: ' + (await page.innerText('body')).slice(0, 300).replace(/\n/g, ' '));

  await browser.close();
}

// ---- Test 2: Live 3D Control, 5-minute recording ----
{
  const videoDir = '/home/nsp/Desktop/ITC_bale_unloading_simulator/latest_model_3d_video';
  const browser = await chromium.launch({ args: ['--no-sandbox'] });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    recordVideo: { dir: videoDir, size: { width: 1280, height: 800 } },
  });
  const page = await context.newPage();
  page.on('console', msg => { if (msg.type() === 'error') results.errors.push('[3D] ' + msg.text()); });
  page.on('pageerror', err => results.errors.push('[3D] pageerror: ' + err.message));
  page.on('response', async res => {
    if (res.url().includes('/api/detect') || res.url().includes('/api/control_tick')) {
      results.apiCalls.push(`[3D] ${res.status()} ${res.url().split('/').pop()}`);
    }
  });

  await page.goto('http://localhost:5173', { waitUntil: 'load', timeout: 30000 });
  await page.waitForTimeout(1000);
  await page.click('text=Live 3D Control');
  await page.waitForTimeout(1500);
  await page.click('button:has-text("Start")');

  await page.waitForTimeout(5 * 60 * 1000); // 5 minutes

  results.notes.push('3D final body snippet: ' + (await page.innerText('body')).slice(0, 300).replace(/\n/g, ' '));
  await page.screenshot({ path: '/home/nsp/Desktop/ITC_bale_unloading_simulator/latest_model_3d_final.png' });

  await page.close();
  await context.close();
  await browser.close();
}

// ---- Summarize ----
const okCount = results.apiCalls.filter(c => c.includes(' 200 ')).length;
const errCount = results.apiCalls.length - okCount;
console.log('=== TEST SUMMARY ===');
console.log('Total console/page errors:', results.errors.length);
if (results.errors.length) console.log(results.errors.slice(0, 20).join('\n'));
console.log('API calls:', results.apiCalls.length, '| 200 OK:', okCount, '| non-200:', errCount);
console.log('Notes:');
console.log(results.notes.join('\n'));
console.log('TEST_AND_RECORD_DONE');
