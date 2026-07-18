import React, { useState } from 'react';
import { usePackOrder } from '../hooks/usePackOrder';
import { BoxVisualizer } from '../components/BoxVisualizer';
import { Plus, Trash2, Box, AlertTriangle, Scale, Percent, ClipboardList, Info, HelpCircle } from 'lucide-react';

const DEFAULT_ITEMS = [
  { item_id: 'prod-001', length_cm: 15, width_cm: 10, height_cm: 8, weight_g: 500, material_type: 'glass' },
  { item_id: 'prod-002', length_cm: 10, width_cm: 6, height_cm: 4, weight_g: 200, material_type: 'electronics' },
  { item_id: 'prod-003', length_cm: 20, width_cm: 15, height_cm: 2, weight_g: 100, material_type: 'apparel' },
];

export const PackOrder = () => {
  const { pack, loading, error, result } = usePackOrder();
  const [allowRotation, setAllowRotation] = useState(false);
  const [items, setItems] = useState(DEFAULT_ITEMS);

  const [newItem, setNewItem] = useState({
    item_id: '',
    length_cm: '',
    width_cm: '',
    height_cm: '',
    weight_g: '',
    material_type: 'standard',
  });

  const handleAddItem = (e) => {
    e.preventDefault();
    if (!newItem.length_cm || !newItem.width_cm || !newItem.height_cm || !newItem.weight_g) {
      alert('Please fill out all item specifications.');
      return;
    }

    const id = newItem.item_id.trim() || `item-${Date.now()}`;
    setItems((prev) => [
      ...prev,
      {
        item_id: id,
        length_cm: parseFloat(newItem.length_cm),
        width_cm: parseFloat(newItem.width_cm),
        height_cm: parseFloat(newItem.height_cm),
        weight_g: parseFloat(newItem.weight_g),
        material_type: newItem.material_type,
      },
    ]);

    setNewItem({
      item_id: '',
      length_cm: '',
      width_cm: '',
      height_cm: '',
      weight_g: '',
      material_type: 'standard',
    });
  };

  const handleRemoveItem = (index) => {
    setItems((prev) => prev.filter((_, i) => i !== index));
  };

  const handlePack = () => {
    if (items.length === 0) {
      alert('Add at least one item to pack.');
      return;
    }
    pack({
      allow_rotation: allowRotation,
      items: items,
    });
  };

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-8 bg-slate-50 dark:bg-slate-900/50">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
          Bin Packing Optimization
        </h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1">
          Perform 3D container selection and item layering alignment with safety constraints.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Side: Order Configurator */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 rounded-2xl shadow-xl p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-bold text-slate-900 dark:text-white">Order Configurator</h2>
              <button
                type="button"
                onClick={() => setItems(DEFAULT_ITEMS)}
                className="text-xs font-semibold text-brand-green hover:underline"
              >
                Reset to Preset
              </button>
            </div>

            {/* Config inputs */}
            <div className="flex items-center gap-2 mb-6">
              <input
                type="checkbox"
                id="allow_rotation"
                checked={allowRotation}
                onChange={(e) => setAllowRotation(e.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-brand-green focus:ring-brand-green/30"
              />
              <label htmlFor="allow_rotation" className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Allow 3D rotation of items
              </label>
            </div>

            {/* List of current items */}
            <div className="space-y-3 mb-6 max-h-56 overflow-y-auto pr-1">
              {items.map((item, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-800/80 rounded-xl"
                >
                  <div className="text-left">
                    <p className="text-xs font-bold text-slate-800 dark:text-slate-200">{item.item_id}</p>
                    <p className="text-[10px] text-slate-500">
                      {item.length_cm}x{item.width_cm}x{item.height_cm} cm | {item.weight_g}g | {item.material_type}
                    </p>
                  </div>
                  <button
                    onClick={() => handleRemoveItem(index)}
                    aria-label="Remove item"
                    className="p-1.5 text-slate-400 hover:text-brand-red hover:bg-brand-red/5 rounded-lg transition-colors"
                  >
                    <Trash2 className="h-4.5 w-4.5" />
                  </button>
                </div>
              ))}
              {items.length === 0 && (
                <div className="text-center py-6 text-sm text-slate-400">No items added yet.</div>
              )}
            </div>

            {/* Add item form inline */}
            <div className="border-t border-slate-100 dark:border-slate-700/50 pt-5">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Add Custom Item</h3>
              <form onSubmit={handleAddItem} className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <input
                      placeholder="SKU / ID"
                      value={newItem.item_id}
                      onChange={(e) => setNewItem({ ...newItem, item_id: e.target.value })}
                      className="w-full px-3 py-2 text-xs bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white outline-none focus:ring-1 focus:ring-brand-green"
                    />
                  </div>
                  <div>
                    <select
                      value={newItem.material_type}
                      onChange={(e) => setNewItem({ ...newItem, material_type: e.target.value })}
                      className="w-full px-3 py-2 text-xs bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white outline-none focus:ring-1 focus:ring-brand-green"
                    >
                      <option value="standard">Standard</option>
                      <option value="glass">Glass</option>
                      <option value="electronics">Electronics</option>
                      <option value="apparel">Apparel</option>
                      <option value="fragile_liquid">Fragile Liquid</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-2">
                  <input
                    type="number"
                    placeholder="L (cm)"
                    value={newItem.length_cm}
                    onChange={(e) => setNewItem({ ...newItem, length_cm: e.target.value })}
                    className="w-full px-2.5 py-2 text-xs bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white outline-none focus:ring-1 focus:ring-brand-green"
                  />
                  <input
                    type="number"
                    placeholder="W (cm)"
                    value={newItem.width_cm}
                    onChange={(e) => setNewItem({ ...newItem, width_cm: e.target.value })}
                    className="w-full px-2.5 py-2 text-xs bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white outline-none focus:ring-1 focus:ring-brand-green"
                  />
                  <input
                    type="number"
                    placeholder="H (cm)"
                    value={newItem.height_cm}
                    onChange={(e) => setNewItem({ ...newItem, height_cm: e.target.value })}
                    className="w-full px-2.5 py-2 text-xs bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white outline-none focus:ring-1 focus:ring-brand-green"
                  />
                  <input
                    type="number"
                    placeholder="Wt (g)"
                    value={newItem.weight_g}
                    onChange={(e) => setNewItem({ ...newItem, weight_g: e.target.value })}
                    className="w-full px-2.5 py-2 text-xs bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white outline-none focus:ring-1 focus:ring-brand-green"
                  />
                </div>

                <button
                  type="submit"
                  className="w-full flex items-center justify-center gap-1.5 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-brand-green hover:text-white dark:hover:bg-brand-green dark:text-slate-200 text-slate-700 rounded-xl font-bold text-xs transition"
                >
                  <Plus className="h-4 w-4" />
                  <span>Add Item to Order</span>
                </button>
              </form>
            </div>

            <button
              onClick={handlePack}
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 py-3 bg-brand-green hover:bg-emerald-600 text-white rounded-xl font-semibold shadow-lg shadow-brand-green/20 hover:shadow-xl transition disabled:opacity-50 mt-6"
            >
              {loading ? (
                <div className="h-5 w-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  <Box className="h-4.5 w-4.5" />
                  <span>Run Packing Engine</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Side: Results & Visualizer */}
        <div className="lg:col-span-7 space-y-6">
          {error && (
            <div className="flex items-center gap-2 p-3 text-sm text-brand-red bg-brand-red/10 border border-brand-red/25 rounded-xl">
              <AlertTriangle className="h-4.5 w-4.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {!result ? (
            <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 rounded-2xl p-8 text-center flex flex-col items-center justify-center min-h-[450px]">
              <HelpCircle className="h-16 w-16 text-slate-300 dark:text-slate-600 mb-4 animate-pulse" />
              <h3 className="text-lg font-bold text-slate-800 dark:text-slate-200">No Packing Results Available</h3>
              <p className="text-slate-500 dark:text-slate-400 max-w-sm mt-1">
                Configure your shipment list and run the packing solver to obtain 3D visual layer layouts and void calculations.
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Visualizer Panel */}
              <BoxVisualizer
                boxDimensions={result.box_dimensions}
                placements={result.placements}
              />

              {/* Numerical Metrics Panel */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 p-4 rounded-xl shadow">
                  <div className="flex items-center gap-1.5 text-slate-400">
                    <ClipboardList className="h-4 w-4" />
                    <span className="text-[10px] font-bold uppercase tracking-wider">SKU Code</span>
                  </div>
                  <p className="text-lg font-bold text-slate-900 dark:text-white mt-1">{result.box_sku}</p>
                </div>

                <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 p-4 rounded-xl shadow">
                  <div className="flex items-center gap-1.5 text-slate-400">
                    <Percent className="h-4 w-4" />
                    <span className="text-[10px] font-bold uppercase tracking-wider">Void Vol.</span>
                  </div>
                  <p className="text-lg font-bold text-slate-900 dark:text-white mt-1">{result.void_volume_pct.toFixed(1)}%</p>
                </div>

                <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 p-4 rounded-xl shadow">
                  <div className="flex items-center gap-1.5 text-slate-400">
                    <Scale className="h-4 w-4" />
                    <span className="text-[10px] font-bold uppercase tracking-wider">Mat. Weight</span>
                  </div>
                  <p className="text-lg font-bold text-slate-900 dark:text-white mt-1">{result.estimated_material_weight_g.toFixed(1)}g</p>
                </div>

                <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 p-4 rounded-xl shadow">
                  <div className="flex items-center gap-1.5 text-slate-400">
                    <AlertTriangle className="h-4 w-4" />
                    <span className="text-[10px] font-bold uppercase tracking-wider">Violations</span>
                  </div>
                  <p
                    className={`text-lg font-bold mt-1 ${
                      result.constraint_violations > 0 ? 'text-brand-red' : 'text-brand-green'
                    }`}
                  >
                    {result.constraint_violations}
                  </p>
                </div>
              </div>

              {/* Order Split Notification if any */}
              {result.requires_split && (
                <div className="flex items-start gap-3 p-4 bg-brand-amber/10 border border-brand-amber/25 rounded-xl">
                  <Info className="h-5 w-5 text-brand-amber shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-sm font-bold text-brand-amber">Order Split Required</h4>
                    <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
                      Due to heavy weights or fragility constraints, the following items had to be packaged in a separate container:
                      <span className="block mt-1 font-mono font-bold text-slate-800 dark:text-slate-200">
                        {result.separate_box_items.join(', ')}
                      </span>
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
