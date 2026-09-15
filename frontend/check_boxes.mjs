import { chromium } from 'playwright';

const browser = await chromium.launch({ args: ['--no-sandbox'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });

await page.goto('http://localhost:5173', { waitUntil: 'load', timeout: 30000 });
await page.waitForTimeout(1000);
await page.click('text=Live 3D Control');
await page.waitForTimeout(1500);
await page.click('button:has-text("Start")');
await page.waitForTimeout(15000);

await page.screenshot({ path: '/home/nsp/Desktop/ITC_bale_unloading_simulator/box_check_full.png' });

// zoom into just the camera panels
const camPanel = page.locator('text=Camera_Top').first();
const box = await camPanel.boundingBox();
if (box) {
  await page.screenshot({
    path: '/home/nsp/Desktop/ITC_bale_unloading_simulator/box_check_camera_top.png',
    clip: { x: box.x - 10, y: box.y - 250, width: 350, height: 280 },
  });
}

await browser.close();
console.log('done');
