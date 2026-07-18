import { test, expect } from '@playwright/test';

test.describe('Classify Flows', () => {
  test('submit fragility classification product dimensions', async ({ page }) => {
    await page.goto('/login');
    await page.fill('input[type="email"]', 'demo@ecopackai.io');
    await page.fill('input[type="password"]', 'demo123');
    await page.click('button[type="submit"]');
    
    await page.goto('/classify');
    await page.fill('input[name="length_cm"]', '20');
    await page.fill('input[name="width_cm"]', '15');
    await page.fill('input[name="height_cm"]', '10');
    await page.fill('input[name="weight_g"]', '800');
    await page.selectOption('select[name="material_type"]', 'glass');
    
    await page.click('button[type="submit"]');
    // Result displays badge
    const badge = page.locator('span:has-text("Low"), span:has-text("None"), span:has-text("Medium"), span:has-text("Critical")');
    await expect(badge).toBeVisible({ timeout: 5000 });
  });
});
