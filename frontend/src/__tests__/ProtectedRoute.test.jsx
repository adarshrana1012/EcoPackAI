import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { ProtectedRoute } from '../components/ProtectedRoute';

// Mock useAuth
vi.mock('../contexts/AuthContext', () => ({
  useAuth: vi.fn(),
}));

describe('ProtectedRoute Component', () => {
  it('redirects to /login if not authenticated', () => {
    vi.mocked(useAuth).mockReturnValue({
      isAuthenticated: false,
      user: null,
    });

    render(
      <MemoryRouter initialEntries={['/protected']}>
        <Routes>
          <Route path="/login" element={<span>Login Page</span>} />
          <Route element={<ProtectedRoute />}>
            <Route path="/protected" element={<span>Secret Page</span>} />
          </Route>
        </Routes>
      </MemoryRouter>
    );

    expect(screen.getByText('Login Page')).toBeInTheDocument();
  });

  it('renders children if authenticated and roles match', () => {
    vi.mocked(useAuth).mockReturnValue({
      isAuthenticated: true,
      user: { email: 'admin@ecopackai.io', roles: ['admin'] },
    });

    render(
      <MemoryRouter initialEntries={['/protected']}>
        <Routes>
          <Route element={<ProtectedRoute requiredRole="admin" />}>
            <Route path="/protected" element={<span>Secret Page</span>} />
          </Route>
        </Routes>
      </MemoryRouter>
    );

    expect(screen.getByText('Secret Page')).toBeInTheDocument();
  });

  it('redirects to root / if required role is missing', () => {
    vi.mocked(useAuth).mockReturnValue({
      isAuthenticated: true,
      user: { email: 'demo@ecopackai.io', roles: ['user'] },
    });

    render(
      <MemoryRouter initialEntries={['/admin']}>
        <Routes>
          <Route path="/" element={<span>Dashboard Home</span>} />
          <Route element={<ProtectedRoute requiredRole="admin" />}>
            <Route path="/admin" element={<span>Admin Page</span>} />
          </Route>
        </Routes>
      </MemoryRouter>
    );

    expect(screen.getByText('Dashboard Home')).toBeInTheDocument();
  });
});
