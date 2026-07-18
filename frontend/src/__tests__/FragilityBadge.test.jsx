import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FragilityBadge } from '../FragilityBadge';

describe('FragilityBadge Component', () => {
  it('renders Tier 0 with correct label and style classes', () => {
    render(<FragilityBadge tier={0} />);
    expect(screen.getByText('None')).toBeInTheDocument();
  });

  it('renders Tier 1 with correct label and style classes', () => {
    render(<FragilityBadge tier={1} />);
    expect(screen.getByText('Low')).toBeInTheDocument();
  });

  it('renders Tier 2 with correct label and style classes', () => {
    render(<FragilityBadge tier={2} />);
    expect(screen.getByText('Medium')).toBeInTheDocument();
  });

  it('renders Tier 3 with correct label and style classes', () => {
    render(<FragilityBadge tier={3} />);
    expect(screen.getByText('Critical')).toBeInTheDocument();
  });
});
