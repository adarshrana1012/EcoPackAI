import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PackOrder } from '../pages/PackOrder';
import { usePackOrder } from '../hooks/usePackOrder';

// Mock hook
vi.mock('../hooks/usePackOrder', () => ({
  usePackOrder: vi.fn(),
}));

describe('PackOrder Page', () => {
  it('renders initial list items properly', () => {
    vi.mocked(usePackOrder).mockReturnValue({
      pack: vi.fn(),
      loading: false,
      error: '',
      result: null,
    });

    render(<PackOrder />);
    expect(screen.getByText('prod-001')).toBeInTheDocument();
    expect(screen.getByText('prod-002')).toBeInTheDocument();
    expect(screen.getByText('prod-003')).toBeInTheDocument();
  });

  it('allows adding and removing items from list', () => {
    vi.mocked(usePackOrder).mockReturnValue({
      pack: vi.fn(),
      loading: false,
      error: '',
      result: null,
    });

    render(<PackOrder />);
    
    // Check initial rows
    expect(screen.getByText('prod-001')).toBeInTheDocument();
    
    // Remove first item
    const removeButtons = screen.getAllByRole('button', { name: /Remove item/i });
    fireEvent.click(removeButtons[0]);
    
    expect(screen.queryByText('prod-001')).not.toBeInTheDocument();
  });
});
