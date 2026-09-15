import { chromium } from 'playwright';

const browser = await chromium.launch({ args: ['--no-sandbox'] });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
const errors = [];
page.on('console', msg => { if (msg.type() === 'error') errors.push(msg.text()); });
page.on('pageerror', err => errors.push('pageerror: ' + err.message));

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

const thresholdInput = page.locator('label:has-text("Confidence threshold") input');
await thresholdInput.fill('0.3');
const val = await thresholdInput.inputValue();
console.log('confidence threshold field now reads:', val);

await page.click('button:has-text("Simulate")');
await page.waitForSelector('.simulate-btn:not(:has-text("Simulating"))', { timeout: 25000 }).catch(() => console.log('timeout waiting for simulate'));
await page.waitForTimeout(1000);

await page.screenshot({ path: '/home/nsp/Desktop/ITC_bale_unloading_simulator/retest_2d_result.png' });
console.log('body snippet:', (await page.innerText('body')).slice(0, 300));
console.log('errors:', errors.length, errors.join('\n'));

await browser.close();
