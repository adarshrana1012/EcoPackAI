import React, { useState } from 'react';

export const BoxVisualizer = ({ boxDimensions, placements }) => {
  const [hoveredItem, setHoveredItem] = useState(null);

  // Parse box dimensions, e.g., "30x20x15"
  const [boxL, boxW, boxH] = boxDimensions
    ? boxDimensions.split('x').map(Number)
    : [30, 20, 15];

  // Isometric projection math
  // Angle of projection (30 degrees)
  const alpha = 30 * (Math.PI / 180);
  const cosAlpha = Math.cos(alpha);
  const sinAlpha = Math.sin(alpha);

  // Scaling factor to fit SVG viewport
  const scale = Math.min(250 / boxL, 250 / boxW, 200 / boxH, 8);

  const project = (x, y, z) => {
    // Center of SVG is at (300, 250)
    const px = 300 + (x - y) * cosAlpha * scale;
    const py = 250 + (x + y) * sinAlpha * scale - z * scale;
    return { x: px, y: py };
  };

  // Sort placements from back (furthest) to front (closest) to camera
  // Back corner is (0,0,0), Front corner is (boxL, boxW, boxH)
  // Sort by x + y + z ascending
  const sortedPlacements = [...placements].sort((a, b) => {
    const depthA = a.x + a.y + a.z;
    const depthB = b.x + b.y + b.z;
    return depthA - depthB;
  });

  // Fragility tier colors
  const tierColors = {
    0: { border: '#10b981', fill: 'rgba(16, 185, 129, 0.4)' }, // None - Green
    1: { border: '#0ea5e9', fill: 'rgba(14, 165, 233, 0.4)' }, // Low - Blue
    2: { border: '#f59e0b', fill: 'rgba(245, 158, 11, 0.4)' }, // Medium - Amber
    3: { border: '#ef4444', fill: 'rgba(239, 68, 68, 0.4)' }, // Critical - Red
  };

  // Generate faces for a 3D block
  const renderItem = (item, index) => {
    const { x, y, z, placed_length: l, placed_width: w, placed_height: h, fragility_tier } = item;
    const color = tierColors[fragility_tier] || tierColors[0];
    const isHovered = hoveredItem?.item_id === item.item_id;

    // 8 vertices of the item block
    const p000 = project(x, y, z);
    const p100 = project(x + l, y, z);
    const p010 = project(x, y + w, z);
    const p001 = project(x, y, z + h);
    const p101 = project(x + l, y, z + h);
    const p011 = project(x, y + w, z + h);
    const p111 = project(x + l, y + w, z + h);

    const makePath = (points) => points.map((p) => `${p.x},${p.y}`).join(' ');

    return (
      <g
        key={`${item.item_id}-${index}`}
        className="cursor-pointer transition-all duration-200"
        onMouseEnter={() => setHoveredItem(item)}
        onMouseLeave={() => setHoveredItem(null)}
      >
        {/* Left/Front-Left Face */}
        <polygon
          points={makePath([p000, p010, p011, p001])}
          fill={isHovered ? 'rgba(16, 185, 129, 0.7)' : color.fill}
          stroke={color.border}
          strokeWidth="1.5"
          className="transition-colors duration-150"
        />
        {/* Right/Front-Right Face */}
        <polygon
          points={makePath([p000, p100, p101, p001])}
          fill={isHovered ? 'rgba(16, 185, 129, 0.65)' : color.fill}
          stroke={color.border}
          strokeWidth="1.5"
          className="transition-colors duration-150"
        />
        {/* Top Face */}
        <polygon
          points={makePath([p001, p101, p111, p011])}
          fill={isHovered ? 'rgba(16, 185, 129, 0.8)' : color.fill}
          stroke={color.border}
          strokeWidth="1.5"
          className="transition-colors duration-150"
        />
        {/* Inner grid lines for textured look */}
        <line x1={p001.x} y1={p001.y} x2={p111.x} y2={p111.y} stroke={color.border} strokeWidth="0.5" strokeDasharray="2 2" opacity="0.3" />
      </g>
    );
  };

  // Outer Box Vertices
  const b000 = project(0, 0, 0);
  const b100 = project(boxL, 0, 0);
  const b010 = project(0, boxW, 0);
  const b110 = project(boxL, boxW, 0);
  const b001 = project(0, 0, boxH);
  const b101 = project(boxL, 0, boxH);
  const b011 = project(0, boxW, boxH);
  const b111 = project(boxL, boxW, boxH);

  const makePath = (points) => points.map((p) => `${p.x},${p.y}`).join(' ');

  return (
    <div className="relative flex flex-col items-center bg-slate-900 border border-slate-800 rounded-2xl p-4 overflow-hidden h-[450px]">
      <div className="absolute top-4 left-4 z-10">
        <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Interactive 3D Packing Visualizer</span>
        <h3 className="text-sm font-semibold text-white mt-0.5">Box SKU: {boxDimensions ? 'Dynamic SKU' : 'N/A'}</h3>
      </div>

      {/* SVG Canvas */}
      <svg className="w-full h-full select-none" viewBox="0 0 600 450">
        {/* Box Back Frame (dashed/light lines) */}
        <polyline points={makePath([b100, b000, b010])} fill="none" stroke="#475569" strokeWidth="1" strokeDasharray="3 3" />
        <line x1={b000.x} y1={b000.y} x2={b001.x} y2={b001.y} stroke="#475569" strokeWidth="1" strokeDasharray="3 3" />

        {/* Render Sorted Placed Items */}
        {sortedPlacements.map((item, index) => renderItem(item, index))}

        {/* Box Front Frame (solid/prominent lines) */}
        <polygon points={makePath([b100, b110, b010, b011, b111, b101])} fill="none" stroke="#64748b" strokeWidth="2" />
        {/* Box Open Flaps Top */}
        <polyline points={makePath([b001, b101, b111, b011, b001])} fill="none" stroke="#64748b" strokeWidth="2" />
        {/* Connect outer edges */}
        <line x1={b100.x} y1={b100.y} x2={b101.x} y2={b101.y} stroke="#64748b" strokeWidth="2" />
        <line x1={b010.x} y1={b010.y} x2={b011.x} y2={b011.y} stroke="#64748b" strokeWidth="2" />
        <line x1={b110.x} y1={b110.y} x2={b111.x} y2={b111.y} stroke="#64748b" strokeWidth="2" />
      </svg>

      {/* Hover Info Tooltip overlay */}
      {hoveredItem ? (
        <div className="absolute bottom-4 left-4 right-4 bg-slate-950/90 border border-slate-800 rounded-xl p-3 backdrop-blur shadow-2xl transition-all duration-200">
          <div className="flex justify-between items-start">
            <div>
              <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wide">Item ID</span>
              <p className="text-xs font-semibold text-white">{hoveredItem.item_id}</p>
            </div>
            <div className="text-right">
              <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wide">Dimensions (cm)</span>
              <p className="text-xs font-semibold text-white">
                {hoveredItem.placed_length} x {hoveredItem.placed_width} x {hoveredItem.placed_height}
              </p>
            </div>
          </div>
          <div className="mt-2 flex items-center justify-between border-t border-slate-800/80 pt-2 text-[10px]">
            <span className="text-slate-400">Position: ({hoveredItem.x}, {hoveredItem.y}, {hoveredItem.z})</span>
            <span className="text-brand-green font-semibold">Fragility: {hoveredItem.fragility_label}</span>
          </div>
        </div>
      ) : (
        <div className="absolute bottom-4 text-center text-xs text-slate-500 font-medium">
          Hover over items to view packaging and location metadata.
        </div>
      )}
    </div>
  );
};
