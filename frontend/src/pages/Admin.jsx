import React, { useState, useEffect, useCallback } from 'react';
import { useApi } from '../hooks/useApi';
import {
  Settings, ShieldAlert, Activity, Play, RefreshCw, AlertCircle, CheckCircle, ArrowRight
} from 'lucide-react';

export const Admin = () => {
  const api = useApi();
  const [abResults, setAbResults] = useState(null);
  const [modelVersions, setModelVersions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [message, setMessage] = useState({ text: '', type: '' });

  // Traffic split state locally for the slider
  const [rlPct, setRlPct] = useState(50);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [abRes, modelRes] = await Promise.all([
        api.get('/ab-test/results'),
        api.get('/models/versions'),
      ]);
      setAbResults(abRes.data);
      setModelVersions(modelRes.data);
      if (abRes.data?.config) {
        setRlPct(abRes.data.config.rl_traffic_pct);
      }
    } catch (err) {
      console.error(err);
      setMessage({ text: 'Error fetching admin data.', type: 'error' });
    } finally {
      setLoading(false);
    }
  }, [api]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleUpdateTraffic = async () => {
    setActionLoading(true);
    setMessage({ text: '', type: '' });
    try {
      await api.post('/ab-test/config', {
        rl_traffic_pct: parseFloat(rlPct),
        ffd_traffic_pct: 100.0 - parseFloat(rlPct),
      });
      setMessage({ text: 'A/B test traffic configuration updated successfully.', type: 'success' });
      fetchData();
    } catch (err) {
      console.error(err);
      setMessage({ text: 'Failed to update traffic configuration.', type: 'error' });
    } finally {
      setActionLoading(false);
    }
  };

  const handleResetMetrics = async () => {
    if (!window.confirm('Are you sure you want to reset A/B test tracking metrics?')) return;
    setActionLoading(true);
    setMessage({ text: '', type: '' });
    try {
      await api.post('/ab-test/reset');
      setMessage({ text: 'A/B test metrics reset successfully.', type: 'success' });
      fetchData();
    } catch (err) {
      console.error(err);
      setMessage({ text: 'Failed to reset metrics.', type: 'error' });
    } finally {
      setActionLoading(false);
    }
  };

  const handlePromote = async (version) => {
    setActionLoading(true);
    setMessage({ text: '', type: '' });
    try {
      await api.post(`/models/promote/${version}`);
      setMessage({ text: `Model version ${version} promoted to production.`, type: 'success' });
      fetchData();
    } catch (err) {
      console.error(err);
      setMessage({ text: 'Failed to promote model.', type: 'error' });
    } finally {
      setActionLoading(false);
    }
  };

  const handleTriggerTrain = async () => {
    setActionLoading(true);
    setMessage({ text: '', type: '' });
    try {
      const response = await api.post('/train/trigger');
      setMessage({ text: response.data.message || 'Retraining pipeline triggered successfully.', type: 'success' });
    } catch (err) {
      console.error(err);
      setMessage({ text: 'Failed to trigger model retraining.', type: 'error' });
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-8 bg-slate-50 dark:bg-slate-900/50">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
            Admin Control Panel
          </h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">
            Manage Reinforcement Learning traffic splits, model promotions, and retraining jobs.
          </p>
        </div>

        <button
          onClick={fetchData}
          disabled={loading}
          aria-label="Refresh admin data"
          className="flex items-center justify-center h-10 w-10 border border-slate-200 dark:border-slate-800 rounded-xl hover:bg-slate-50 dark:hover:bg-slate-800 transition focus:outline-none focus:ring-2 focus:ring-brand-green/30 text-slate-500"
        >
          <RefreshCw className={`h-4.5 w-4.5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {message.text && (
        <div
          className={`flex items-center gap-3 p-4 rounded-xl border ${
            message.type === 'success'
              ? 'bg-emerald-50 dark:bg-emerald-950/20 text-emerald-800 dark:text-emerald-400 border-emerald-200 dark:border-emerald-900/30'
              : 'bg-rose-50 dark:bg-rose-950/20 text-rose-800 dark:text-rose-400 border-rose-200 dark:border-rose-900/30'
          }`}
        >
          {message.type === 'success' ? (
            <CheckCircle className="h-5 w-5 shrink-0" />
          ) : (
            <AlertCircle className="h-5 w-5 shrink-0" />
          )}
          <span className="text-sm font-semibold">{message.text}</span>
        </div>
      )}

      {loading ? (
        <div className="flex flex-col items-center justify-center min-h-[300px]">
          <div className="h-10 w-10 border-4 border-brand-green border-t-transparent rounded-full animate-spin mb-4" />
          <span className="text-sm font-semibold text-slate-500">Loading configurations...</span>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Side: A/B Test Routing */}
          <div className="lg:col-span-7 space-y-6">
            <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 rounded-2xl shadow-xl p-6">
              <div className="flex items-center gap-2 mb-6">
                <Settings className="h-5 w-5 text-brand-green" />
                <h2 className="text-lg font-bold text-slate-900 dark:text-white">A/B Testing Traffic Split</h2>
              </div>

              <div className="space-y-6">
                <div>
                  <div className="flex justify-between text-sm font-semibold text-slate-600 dark:text-slate-400 mb-2">
                    <span>FFD Baseline: {100 - rlPct}%</span>
                    <span>RL Policy: {rlPct}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={rlPct}
                    onChange={(e) => setRlPct(e.target.value)}
                    className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-brand-green focus:outline-none"
                  />
                </div>

                <div className="flex gap-3 justify-end">
                  <button
                    onClick={handleResetMetrics}
                    disabled={actionLoading}
                    className="px-4 py-2 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 rounded-xl hover:bg-slate-50 dark:hover:bg-slate-800 transition text-sm font-semibold"
                  >
                    Reset Metrics
                  </button>
                  <button
                    onClick={handleUpdateTraffic}
                    disabled={actionLoading}
                    className="px-4 py-2 bg-brand-green hover:bg-emerald-600 text-white rounded-xl font-semibold shadow-lg shadow-brand-green/20 hover:shadow-xl transition text-sm disabled:opacity-50"
                  >
                    Apply Config
                  </button>
                </div>
              </div>
            </div>

            {/* A/B Test Results Variant Table */}
            <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 rounded-2xl shadow-xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <Activity className="h-5 w-5 text-brand-green" />
                <h2 className="text-lg font-bold text-slate-900 dark:text-white">Live Variant Metrics</h2>
              </div>

              {abResults?.variants ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-100 dark:border-slate-700/50 text-slate-400 uppercase tracking-wider font-bold">
                        <th className="pb-3 font-semibold">Variant</th>
                        <th className="pb-3 font-semibold text-right">Requests</th>
                        <th className="pb-3 font-semibold text-right">Void Vol. (Mean)</th>
                        <th className="pb-3 font-semibold text-right">Violations (Total)</th>
                        <th className="pb-3 font-semibold text-right">Latency (Mean)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-700/50">
                      {abResults.variants.map((v) => (
                        <tr key={v.variant} className="text-slate-700 dark:text-slate-300">
                          <td className="py-3.5 font-semibold text-slate-900 dark:text-white">{v.variant}</td>
                          <td className="py-3.5 text-right font-medium">{v.request_count}</td>
                          <td className="py-3.5 text-right font-medium">{v.mean_void_pct}%</td>
                          <td className="py-3.5 text-right font-medium">{v.total_violations}</td>
                          <td className="py-3.5 text-right font-medium">{v.mean_compute_time_ms.toFixed(1)} ms</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>

                  {abResults.recommendation && (
                    <div className="mt-4 p-4 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-100 dark:border-slate-800 text-slate-600 dark:text-slate-400 leading-relaxed font-medium">
                      {abResults.recommendation}
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-sm text-slate-400">No active variants metrics found.</p>
              )}
            </div>
          </div>

          {/* Right Side: Model Lifecycle & Retraining */}
          <div className="lg:col-span-5 space-y-6">
            {/* Model Registry Panel */}
            <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 rounded-2xl shadow-xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <ShieldAlert className="h-5 w-5 text-brand-green" />
                <h2 className="text-lg font-bold text-slate-900 dark:text-white">Model Registry</h2>
              </div>

              {modelVersions ? (
                <div className="space-y-4">
                  {/* Production */}
                  <div className="p-4 bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-800/80 rounded-xl flex items-center justify-between">
                    <div>
                      <span className="text-[9px] uppercase font-bold tracking-wider text-slate-400">Production</span>
                      <p className="text-sm font-bold text-slate-800 dark:text-slate-200 mt-0.5">{modelVersions.production?.version}</p>
                      <p className="text-[10px] text-slate-500 mt-0.5">Accuracy: {(modelVersions.production?.accuracy * 100).toFixed(1)}%</p>
                    </div>
                    <span className="px-2.5 py-1 bg-brand-green/10 text-brand-green border border-brand-green/20 rounded-full text-[10px] font-bold">Active</span>
                  </div>

                  {/* Staging */}
                  <div className="p-4 bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-800/80 rounded-xl flex items-center justify-between">
                    <div>
                      <span className="text-[9px] uppercase font-bold tracking-wider text-slate-400">Staging</span>
                      <p className="text-sm font-bold text-slate-800 dark:text-slate-200 mt-0.5">{modelVersions.staging?.version}</p>
                      <p className="text-[10px] text-slate-500 mt-0.5">Accuracy: {(modelVersions.staging?.accuracy * 100).toFixed(1)}%</p>
                    </div>
                    <button
                      onClick={() => handlePromote(modelVersions.staging?.version)}
                      disabled={actionLoading}
                      className="flex items-center gap-1 px-3 py-1.5 bg-brand-green hover:bg-emerald-600 text-white rounded-lg text-xs font-semibold shadow hover:shadow-md transition disabled:opacity-50"
                    >
                      <span>Promote</span>
                      <ArrowRight className="h-3 w-3" />
                    </button>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-slate-400">No registry versions found.</p>
              )}
            </div>

            {/* Retraining panel */}
            <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 rounded-2xl shadow-xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <Play className="h-5 w-5 text-brand-green" />
                <h2 className="text-lg font-bold text-slate-900 dark:text-white">Model Retraining</h2>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mb-6 leading-relaxed">
                Trigger a batch retraining job over all historical packaging shipments and database records. The process executes asynchronously.
              </p>

              <button
                onClick={handleTriggerTrain}
                disabled={actionLoading}
                className="w-full flex items-center justify-center gap-2 py-3 bg-slate-900 hover:bg-slate-950 dark:bg-slate-700 dark:hover:bg-slate-650 text-white rounded-xl font-semibold shadow-lg transition disabled:opacity-50 text-sm"
              >
                <RefreshCw className="h-4.5 w-4.5" />
                <span>Trigger Retraining Job</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
