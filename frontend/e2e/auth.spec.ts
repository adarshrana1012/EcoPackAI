import { test, expect } from '@playwright/test';

test.describe('Authentication Journeys', () => {
  test('unauthenticated page request redirects to login', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL(/\/login/);
  });

  test('login form validation and demo account routing', async ({ page }) => {
    await page.goto('/login');
    await page.fill('input[type="email"]', 'demo@ecopackai.io');
    await page.fill('input[type="password"]', 'demo123');
    await page.click('button[type="submit"]');
    await expect(page).toHaveURL(/\/dashboard/);
  });
});
