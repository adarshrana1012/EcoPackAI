import React, { useState } from 'react';
import { useClassify } from '../hooks/useClassify';
import { FragilityBadge } from '../components/FragilityBadge';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { AlertCircle, HelpCircle, CornerDownRight, Play } from 'lucide-react';

const PRESETS = [
  { name: 'Glass Wine Bottle', length: 30, width: 8, height: 8, weight: 1200, material: 'glass' },
  { name: 'Standard Smartphone', length: 16, width: 7.5, height: 0.8, weight: 200, material: 'electronics' },
  { name: 'Cotton T-Shirt', length: 25, width: 18, height: 2, weight: 150, material: 'apparel' },
  { name: 'Fragile Chemical Liquid', length: 20, width: 10, height: 10, weight: 950, material: 'fragile_liquid' },
];

export const Classify = () => {
  const { classify, loading, error, result } = useClassify();
  const [formData, setFormData] = useState({
    length_cm: '',
    width_cm: '',
    height_cm: '',
    weight_g: '',
    material_type: 'standard',
  });

  const [validationErrors, setValidationErrors] = useState({});

  const validate = () => {
    const errs = {};
    if (!formData.length_cm || Number(formData.length_cm) <= 0) errs.length_cm = 'Must be greater than 0';
    if (!formData.width_cm || Number(formData.width_cm) <= 0) errs.width_cm = 'Must be greater than 0';
    if (!formData.height_cm || Number(formData.height_cm) <= 0) errs.height_cm = 'Must be greater than 0';
    if (!formData.weight_g || Number(formData.weight_g) <= 0) errs.weight_g = 'Must be greater than 0';
    setValidationErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handlePreset = (preset) => {
    setFormData({
      length_cm: preset.length,
      width_cm: preset.width,
      height_cm: preset.height,
      weight_g: preset.weight,
      material_type: preset.material,
    });
    setValidationErrors({});
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!validate()) return;
    classify({
      length_cm: parseFloat(formData.length_cm),
      width_cm: parseFloat(formData.width_cm),
      height_cm: parseFloat(formData.height_cm),
      weight_g: parseFloat(formData.weight_g),
      material_type: formData.material_type,
    });
  };

  // Convert result.probabilities object to array for Recharts
  const chartData = result?.probabilities
    ? Object.keys(result.probabilities).map((key) => ({
        name: key,
        probability: result.probabilities[key] * 100,
      }))
    : [];

  const getBarColor = (name) => {
    if (name === 'Critical') return '#ef4444';
    if (name === 'Medium') return '#f59e0b';
    if (name === 'Low') return '#0ea5e9';
    return '#10b981';
  };

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-8 bg-slate-50 dark:bg-slate-900/50">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
          Fragility Classification
        </h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1">
          Predict the fragility tier of a product to determine safety constraints and optimal box packing strategies.
        </p>
      </div>

      {/* Preset Selectors */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {PRESETS.map((p) => (
          <button
            key={p.name}
            type="button"
            onClick={() => handlePreset(p)}
            className="flex items-center justify-between text-left px-4 py-3 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 rounded-xl hover:border-brand-green/50 dark:hover:border-brand-green/50 hover:bg-slate-50/50 dark:hover:bg-slate-800/80 transition-all group"
          >
            <span className="text-xs font-semibold text-slate-800 dark:text-slate-200">{p.name}</span>
            <CornerDownRight className="h-3.5 w-3.5 text-slate-400 group-hover:text-brand-green transition-colors" />
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Form panel */}
        <div className="lg:col-span-5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 rounded-2xl shadow-xl p-6">
          <h2 className="text-lg font-bold text-slate-900 dark:text-white mb-4">Product Attributes</h2>
          <form onSubmit={handleSubmit} className="space-y-5">
            {error && (
              <div className="flex items-center gap-2 p-3 text-sm text-brand-red bg-brand-red/10 border border-brand-red/25 rounded-lg">
                <AlertCircle className="h-4.5 w-4.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <div>
              <label htmlFor="material_type" className="block text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">
                Material Type
              </label>
              <select
                id="material_type"
                name="material_type"
                value={formData.material_type}
                onChange={handleChange}
                className="w-full px-3.5 py-2.5 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl text-slate-900 dark:text-white focus:ring-2 focus:ring-brand-green/30 focus:border-brand-green outline-none transition"
              >
                <option value="standard">Standard Cardboard/Wood</option>
                <option value="glass">Glass/Ceramic</option>
                <option value="electronics">Electronics/Components</option>
                <option value="apparel">Apparel/Textile</option>
                <option value="fragile_liquid">Fragile Liquids/Chemicals</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="length_cm" className="block text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">
                  Length (cm)
                </label>
                <input
                  id="length_cm"
                  name="length_cm"
                  type="number"
                  step="any"
                  value={formData.length_cm}
                  onChange={handleChange}
                  placeholder="e.g. 15"
                  className={`w-full px-3.5 py-2.5 bg-slate-50 dark:bg-slate-900 border ${
                    validationErrors.length_cm ? 'border-brand-red focus:ring-brand-red/25' : 'border-slate-200 dark:border-slate-700 focus:ring-brand-green/25'
                  } rounded-xl text-slate-900 dark:text-white outline-none focus:ring-2 transition`}
                />
                {validationErrors.length_cm && (
                  <span className="text-[10px] text-brand-red font-medium mt-1 block">{validationErrors.length_cm}</span>
                )}
              </div>

              <div>
                <label htmlFor="width_cm" className="block text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">
                  Width (cm)
                </label>
                <input
                  id="width_cm"
                  name="width_cm"
                  type="number"
                  step="any"
                  value={formData.width_cm}
                  onChange={handleChange}
                  placeholder="e.g. 10"
                  className={`w-full px-3.5 py-2.5 bg-slate-50 dark:bg-slate-900 border ${
                    validationErrors.width_cm ? 'border-brand-red focus:ring-brand-red/25' : 'border-slate-200 dark:border-slate-700 focus:ring-brand-green/25'
                  } rounded-xl text-slate-900 dark:text-white outline-none focus:ring-2 transition`}
                />
                {validationErrors.width_cm && (
                  <span className="text-[10px] text-brand-red font-medium mt-1 block">{validationErrors.width_cm}</span>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="height_cm" className="block text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">
                  Height (cm)
                </label>
                <input
                  id="height_cm"
                  name="height_cm"
                  type="number"
                  step="any"
                  value={formData.height_cm}
                  onChange={handleChange}
                  placeholder="e.g. 5"
                  className={`w-full px-3.5 py-2.5 bg-slate-50 dark:bg-slate-900 border ${
                    validationErrors.height_cm ? 'border-brand-red focus:ring-brand-red/25' : 'border-slate-200 dark:border-slate-700 focus:ring-brand-green/25'
                  } rounded-xl text-slate-900 dark:text-white outline-none focus:ring-2 transition`}
                />
                {validationErrors.height_cm && (
                  <span className="text-[10px] text-brand-red font-medium mt-1 block">{validationErrors.height_cm}</span>
                )}
              </div>

              <div>
                <label htmlFor="weight_g" className="block text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">
                  Weight (g)
                </label>
                <input
                  id="weight_g"
                  name="weight_g"
                  type="number"
                  step="any"
                  value={formData.weight_g}
                  onChange={handleChange}
                  placeholder="e.g. 500"
                  className={`w-full px-3.5 py-2.5 bg-slate-50 dark:bg-slate-900 border ${
                    validationErrors.weight_g ? 'border-brand-red focus:ring-brand-red/25' : 'border-slate-200 dark:border-slate-700 focus:ring-brand-green/25'
                  } rounded-xl text-slate-900 dark:text-white outline-none focus:ring-2 transition`}
                />
                {validationErrors.weight_g && (
                  <span className="text-[10px] text-brand-red font-medium mt-1 block">{validationErrors.weight_g}</span>
                )}
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 py-3 bg-brand-green hover:bg-emerald-600 text-white rounded-xl font-semibold shadow-lg shadow-brand-green/20 hover:shadow-xl transition-all disabled:opacity-50 mt-2"
            >
              {loading ? (
                <div className="h-5 w-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  <Play className="h-4.5 w-4.5 fill-current" />
                  <span>Run Inference</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Results panel */}
        <div className="lg:col-span-7 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 rounded-2xl shadow-xl p-6 min-h-[420px] flex flex-col justify-between">
          {!result ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
              <HelpCircle className="h-16 w-16 text-slate-300 dark:text-slate-600 mb-4 animate-bounce" />
              <h3 className="text-lg font-bold text-slate-800 dark:text-slate-200">No Inference Run Yet</h3>
              <p className="text-slate-500 dark:text-slate-400 max-w-sm mt-1">
                Enter product specifications or select a preset and click "Run Inference" to analyze the safety score.
              </p>
            </div>
          ) : (
            <div className="space-y-6 flex-1 flex flex-col">
              <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-700/50 pb-4">
                <div>
                  <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Classified Tier</span>
                  <div className="mt-1 flex items-center gap-3">
                    <span className="text-2xl font-bold text-slate-900 dark:text-white">{result.tier_label}</span>
                    <FragilityBadge tier={result.tier} />
                  </div>
                </div>

                <div className="text-right">
                  <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Model Confidence</span>
                  <div className="text-2xl font-black text-brand-green mt-1">
                    {(result.confidence * 100).toFixed(2)}%
                  </div>
                </div>
              </div>

              {/* Confidence Graph */}
              <div className="flex-1">
                <h4 className="text-xs uppercase font-bold tracking-wider text-slate-400 mb-3">Probability Distribution</h4>
                <div className="h-60 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} layout="vertical" margin={{ left: -10, right: 30, top: 10, bottom: 10 }}>
                      <XAxis type="number" domain={[0, 100]} stroke="#64748b" fontSize={10} tickFormatter={(v) => `${v}%`} />
                      <YAxis dataKey="name" type="category" stroke="#64748b" fontSize={10} width={65} />
                      <Tooltip
                        formatter={(value) => [`${value.toFixed(2)}%`, 'Probability']}
                        contentStyle={{
                          backgroundColor: '#1e293b',
                          borderColor: '#334155',
                          borderRadius: '8px',
                          color: '#fff',
                        }}
                      />
                      <Bar dataKey="probability" radius={[0, 4, 4, 0]} barSize={24}>
                        {chartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={getBarColor(entry.name)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
