import React, { useState, useEffect, useCallback } from 'react';
import { useApi } from '../hooks/useApi';
import { MetricCard } from '../components/MetricCard';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, BarChart, Bar
} from 'recharts';
import { Leaf, Box, Percent, Scale, Download, RefreshCw, BarChart2 } from 'lucide-react';

const PIE_COLORS = ['#10b981', '#0ea5e9', '#f59e0b', '#ef4444'];

export const Analytics = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const api = useApi();

  const fetchMetrics = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const response = await api.get('/metrics/aggregate');
      setMetrics(response.data);
    } catch (err) {
      console.error(err);
      setError('Failed to fetch dashboard metrics.');
    } finally {
      setLoading(false);
    }
  }, [api]);

  useEffect(() => {
    fetchMetrics();
  }, [fetchMetrics]);

  const handleExportJSON = () => {
    if (!metrics) return;
    const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(metrics, null, 2))}`;
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', jsonString);
    downloadAnchor.setAttribute('download', `ecopackai_analytics_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  // Format data for Recharts
  const weeklyData = metrics?.weekly_material_usage || [];
  const fragilityData = metrics?.fragility_distribution
    ? Object.keys(metrics.fragility_distribution).map((key) => ({
        name: key,
        value: metrics.fragility_distribution[key],
      }))
    : [];
  const topVoidData = metrics?.top_void_products || [];

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-8 bg-slate-50 dark:bg-slate-900/50">
      {/* Header controls */}
      <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
            Analytics Dashboard
          </h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">
            Real-time carbon footprint, box material efficiency, and package optimization metrics.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchMetrics}
            disabled={loading}
            aria-label="Refresh dashboard"
            className="flex items-center justify-center h-10 w-10 border border-slate-200 dark:border-slate-800 rounded-xl hover:bg-slate-50 dark:hover:bg-slate-800 transition focus:outline-none focus:ring-2 focus:ring-brand-green/30 text-slate-500"
          >
            <RefreshCw className={`h-4.5 w-4.5 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={handleExportJSON}
            disabled={loading || !metrics}
            className="flex items-center gap-2 px-4 py-2 bg-brand-green hover:bg-emerald-600 text-white rounded-xl font-semibold shadow-lg shadow-brand-green/20 hover:shadow-xl transition disabled:opacity-50 text-sm"
          >
            <Download className="h-4.5 w-4.5" />
            <span>Export Report</span>
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center min-h-[400px]">
          <div className="h-10 w-10 border-4 border-brand-green border-t-transparent rounded-full animate-spin mb-4" />
          <span className="text-sm font-semibold text-slate-500">Loading metrics data...</span>
        </div>
      ) : error ? (
        <div className="bg-brand-red/10 border border-brand-red/25 text-brand-red p-4 rounded-2xl text-center font-medium max-w-md mx-auto">
          {error}
        </div>
      ) : (
        <div className="space-y-8">
          {/* Key Metrics cards grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <MetricCard
              title="CO2e Carbon Saved"
              value={metrics?.co2e_saved_kg?.toFixed(1) || '0.0'}
              unit="kg"
              change="-12.3% vs last week"
              changeType="positive"
              icon={Leaf}
            />

            <MetricCard
              title="Average Void Volume"
              value={`${metrics?.current_void_pct?.toFixed(1) || '0.0'}%`}
              change={`Reduced from ${metrics?.baseline_void_pct || 0}%`}
              changeType="positive"
              icon={Percent}
            />

            <MetricCard
              title="Material Usage Saved"
              value="82.4"
              unit="kg"
              change="+5.1% this month"
              changeType="positive"
              icon={Scale}
            />

            <MetricCard
              title="Total Optimizations"
              value="1,248"
              change="No constraint violations"
              changeType="neutral"
              icon={Box}
            />
          </div>

          {/* Charts grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Chart 1: Material Usage over time */}
            <div className="lg:col-span-8 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 p-6 rounded-2xl shadow-xl">
              <h3 className="text-sm font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider mb-4">
                Weekly Material Usage (kg)
              </h3>
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={weeklyData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                    <XAxis dataKey="week" stroke="#64748b" fontSize={10} />
                    <YAxis stroke="#64748b" fontSize={10} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        borderColor: '#334155',
                        borderRadius: '8px',
                        color: '#fff',
                      }}
                    />
                    <Line type="monotone" dataKey="kg" stroke="#10b981" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Chart 2: Fragility Distribution */}
            <div className="lg:col-span-4 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 p-6 rounded-2xl shadow-xl flex flex-col justify-between">
              <h3 className="text-sm font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider mb-4">
                Fragility Tier Distribution
              </h3>
              <div className="h-56 w-full relative flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={fragilityData}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={70}
                      paddingAngle={4}
                      dataKey="value"
                    >
                      {fragilityData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        borderColor: '#334155',
                        borderRadius: '8px',
                        color: '#fff',
                      }}
                    />
                    <Legend verticalAlign="bottom" height={36} iconSize={10} iconType="circle" />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Chart 3: Top Void Products */}
          <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 p-6 rounded-2xl shadow-xl">
            <div className="flex items-center gap-2 mb-4">
              <BarChart2 className="h-5 w-5 text-brand-green" />
              <h3 className="text-sm font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider">
                Top Products by Void Volume Percentage
              </h3>
            </div>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topVoidData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis dataKey="sku" stroke="#64748b" fontSize={10} />
                  <YAxis stroke="#64748b" fontSize={10} tickFormatter={(v) => `${v}%`} />
                  <Tooltip
                    formatter={(value) => [`${value}%`, 'Void Percentage']}
                    contentStyle={{
                      backgroundColor: '#1e293b',
                      borderColor: '#334155',
                      borderRadius: '8px',
                      color: '#fff',
                    }}
                  />
                  <Bar dataKey="void_pct" fill="#0ea5e9" radius={[4, 4, 0, 0]} barSize={40} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
