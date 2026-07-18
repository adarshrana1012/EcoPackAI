import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Classify } from '../pages/Classify';
import { useClassify } from '../hooks/useClassify';

// Mock hook
vi.mock('../hooks/useClassify', () => ({
  useClassify: vi.fn(),
}));

describe('Classify Page', () => {
  it('renders input elements correctly', () => {
    vi.mocked(useClassify).mockReturnValue({
      classify: vi.fn(),
      loading: false,
      error: '',
      result: null,
    });

    render(<Classify />);
    expect(screen.getByLabelText(/Material Type/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Length/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Width/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Height/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Weight/i)).toBeInTheDocument();
  });

  it('validates fields before submission', () => {
    const mockClassifyFn = vi.fn();
    vi.mocked(useClassify).mockReturnValue({
      classify: mockClassifyFn,
      loading: false,
      error: '',
      result: null,
    });

    render(<Classify />);
    const submitBtn = screen.getByRole('button', { name: /Classify Product/i });
    fireEvent.click(submitBtn);

    expect(mockClassifyFn).not.toHaveBeenCalled();
  });
});
