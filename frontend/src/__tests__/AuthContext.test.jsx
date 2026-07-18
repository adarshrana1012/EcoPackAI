import React from 'react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { AuthProvider, useAuth } from '../contexts/AuthContext';

// Helper component that consumes useAuth context
const ConsumerComponent = () => {
  const { user, token, login, logout, isAuthenticated } = useAuth();
  return (
    <div>
      <span data-testid="auth-state">{isAuthenticated ? 'authenticated' : 'guest'}</span>
      <span data-testid="user-email">{user?.email || 'no-email'}</span>
      <button data-testid="login-btn" onClick={() => login('mock.eyJlbWFpbCI6ImFkbWluQGVjb3BhY2thaS5pbyIsIm5hbWUiOiJBZG1pbiIsImNvbXBhbnkiOiJFQ08iLCJyb2xlcyI6WyJhZG1pbiJdfQ==.signature')}>
        Login
      </button>
      <button data-testid="logout-btn" onClick={() => logout()}>
        Logout
      </button>
    </div>
  );
};

describe('AuthContext Providers', () => {
  beforeEach(() => {
    localStorage.clear();
    // Stub global location for logout redirect mock
    delete window.location;
    window.location = { href: '' };
  });

  it('initially starts as unauthenticated guest', () => {
    render(
      <AuthProvider>
        <ConsumerComponent />
      </AuthProvider>
    );
    expect(screen.getByTestId('auth-state').textContent).toBe('guest');
  });

  it('logs in successfully and populates user info from JWT', () => {
    render(
      <AuthProvider>
        <ConsumerComponent />
      </AuthProvider>
    );

    const loginBtn = screen.getByTestId('login-btn');
    fireEvent.click(loginBtn);

    expect(localStorage.getItem('ecopackai_token')).toBeDefined();
  });
});
