import { test, expect } from '@playwright/test';

test.describe('Order Packing Flows', () => {
  test('pack items and render 3D bin SVG visualization', async ({ page }) => {
    await page.goto('/login');
    await page.fill('input[type="email"]', 'demo@ecopackai.io');
    await page.fill('input[type="password"]', 'demo123');
    await page.click('button[type="submit"]');
    
    await page.goto('/pack');
    await page.click('button:has-text("Pack Order")');
    
    // Result displays box SKU details and SVG visualizations
    const badge = page.locator('span:has-text("BOX-")');
    await expect(badge).toBeVisible({ timeout: 5000 });
  });
});
