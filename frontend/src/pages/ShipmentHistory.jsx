import React, { useState } from 'react';
import { useShipments } from '../hooks/useQueries';
import { useApi } from '../hooks/useApi';
import { 
  Calendar, 
  Search, 
  AlertTriangle, 
  CheckCircle, 
  Download, 
  ChevronLeft, 
  ChevronRight,
  Filter,
  RefreshCw,
  Eye,
  Info
} from 'lucide-react';

export const ShipmentHistory = () => {
  const api = useApi();

  // Filter states
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [boxSku, setBoxSku] = useState('');
  const [damageReported, setDamageReported] = useState('all'); // 'all', 'true', 'false'
  const [page, setPage] = useState(1);

  // Active query parameters (controlled on submit)
  const [appliedFilters, setAppliedFilters] = useState({
    page: 1,
    page_size: 10,
    start_date: '',
    end_date: '',
    box_sku: '',
    damage_reported: undefined
  });

  // Query React Hook
  const { data, isLoading, isError, refetch, isPlaceholderData } = useShipments(appliedFilters);

  // Expose expanded rows
  const [expandedRows, setExpandedRows] = useState({});

  const toggleRow = (id) => {
    setExpandedRows(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const handleApplyFilters = (e) => {
    if (e) e.preventDefault();
    setPage(1);
    setAppliedFilters({
      page: 1,
      page_size: 10,
      start_date: startDate || undefined,
      end_date: endDate || undefined,
      box_sku: boxSku.trim() || undefined,
      damage_reported: damageReported === 'true' ? true : damageReported === 'false' ? false : undefined
    });
  };

  const handleResetFilters = () => {
    setStartDate('');
    setEndDate('');
    setBoxSku('');
    setDamageReported('all');
    setPage(1);
    setAppliedFilters({
      page: 1,
      page_size: 10,
      start_date: undefined,
      end_date: undefined,
      box_sku: undefined,
      damage_reported: undefined
    });
  };

  const handlePageChange = (newPage) => {
    setPage(newPage);
    setAppliedFilters(prev => ({ ...prev, page: newPage }));
  };

  const handleExportCSV = async () => {
    try {
      const params = {};
      if (appliedFilters.start_date) params.start_date = appliedFilters.start_date;
      if (appliedFilters.end_date) params.end_date = appliedFilters.end_date;

      const response = await api.get('/shipments/export', {
        params,
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `shipments_export_${new Date().toISOString().slice(0,10)}.csv`);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
    } catch (error) {
      console.error('CSV Export failed:', error);
    }
  };

  // Client-side statistics calculated from the current page's data
  const pageShipments = data?.shipments || [];
  const totalOnPage = pageShipments.length;

  const meanVoidPct = totalOnPage > 0
    ? (pageShipments.reduce((sum, s) => sum + (s.void_volume_pct || 0), 0) / totalOnPage).toFixed(1)
    : '0.0';

  const totalMaterialWeightKg = totalOnPage > 0
    ? (pageShipments.reduce((sum, s) => sum + (s.material_weight_g || 0), 0) / 1000).toFixed(2)
    : '0.00';

  const damageRatePct = totalOnPage > 0
    ? ((pageShipments.filter(s => s.damage_reported).length / totalOnPage) * 100).toFixed(1)
    : '0.0';

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-8 bg-slate-50 dark:bg-slate-900/50">
      
      {/* Title Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-black md:text-4xl tracking-tight text-slate-900 dark:text-white">
            Shipment History
          </h1>
          <p className="text-slate-500 dark:text-slate-400 text-sm leading-relaxed mt-1">
            Browse, filter, and export historical operational bin-packing metrics.
          </p>
        </div>
        <button
          onClick={handleExportCSV}
          className="inline-flex items-center gap-2 px-4 py-2.5 bg-brand-green hover:bg-emerald-600 active:bg-emerald-700 text-white text-sm font-semibold rounded-xl shadow-md shadow-brand-green/20 transition-all duration-150"
        >
          <Download className="h-4 w-4" />
          Export CSV
        </button>
      </div>

      {/* Filter Bar */}
      <form onSubmit={handleApplyFilters} className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 rounded-2xl p-5 shadow-sm space-y-4">
        <div className="flex items-center gap-2 pb-2 border-b border-slate-100 dark:border-slate-700/50 text-slate-800 dark:text-slate-200">
          <Filter className="h-4 w-4 text-brand-green" />
          <h3 className="text-xs font-bold uppercase tracking-wider">Search & Filters</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Start Date */}
          <div className="space-y-1 text-left">
            <label className="text-xs font-bold text-slate-500 dark:text-slate-400">Start Date</label>
            <div className="relative">
              <Calendar className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 pointer-events-none" />
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 rounded-xl pl-10 pr-4 py-2 text-sm text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-brand-green/30 focus:border-brand-green"
              />
            </div>
          </div>

          {/* End Date */}
          <div className="space-y-1 text-left">
            <label className="text-xs font-bold text-slate-500 dark:text-slate-400">End Date</label>
            <div className="relative">
              <Calendar className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 pointer-events-none" />
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 rounded-xl pl-10 pr-4 py-2 text-sm text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-brand-green/30 focus:border-brand-green"
              />
            </div>
          </div>

          {/* Box SKU */}
          <div className="space-y-1 text-left">
            <label className="text-xs font-bold text-slate-500 dark:text-slate-400">Box SKU</label>
            <div className="relative">
              <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 pointer-events-none" />
              <input
                type="text"
                placeholder="e.g. BOX-M4"
                value={boxSku}
                onChange={(e) => setBoxSku(e.target.value)}
                className="w-full bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 rounded-xl pl-10 pr-4 py-2 text-sm text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-brand-green/30 focus:border-brand-green placeholder-slate-400"
              />
            </div>
          </div>

          {/* Damage Reported */}
          <div className="space-y-1 text-left">
            <label className="text-xs font-bold text-slate-500 dark:text-slate-400">Damage Status</label>
            <select
              value={damageReported}
              onChange={(e) => setDamageReported(e.target.value)}
              className="w-full bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 rounded-xl px-4 py-2 text-sm text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-brand-green/30 focus:border-brand-green"
            >
              <option value="all">All Shipments</option>
              <option value="true">Damage Reported</option>
              <option value="false">No Damage</option>
            </select>
          </div>
        </div>

        <div className="flex justify-end gap-3 pt-2">
          <button
            type="button"
            onClick={handleResetFilters}
            className="px-4 py-2 bg-slate-100 dark:bg-slate-700/60 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs font-bold rounded-lg transition-colors duration-150"
          >
            Reset
          </button>
          <button
            type="submit"
            className="px-4 py-2 bg-brand-green hover:bg-emerald-600 active:bg-emerald-700 text-white text-xs font-bold rounded-lg transition-colors duration-150 shadow-md shadow-brand-green/10"
          >
            Apply Filters
          </button>
        </div>
      </form>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          { title: 'Total Records (Page)', val: totalOnPage, unit: 'shipments' },
          { title: 'Mean Void Volume', val: meanVoidPct, unit: '%' },
          { title: 'Total Material weight', val: totalMaterialWeightKg, unit: 'kg' },
          { title: 'Damage Rate', val: damageRatePct, unit: '%' }
        ].map((c, i) => (
          <div key={i} className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 rounded-2xl p-5 shadow-sm text-left">
            <span className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider">{c.title}</span>
            <div className="flex items-baseline gap-1.5 mt-2">
              <span className="text-3xl font-black text-slate-900 dark:text-white tracking-tight">{c.val}</span>
              <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">{c.unit}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Shipments Table Container */}
      <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 rounded-2xl shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[800px] text-left border-collapse">
            <thead>
              <tr className="bg-slate-50 dark:bg-slate-900/40 border-b border-slate-200 dark:border-slate-700/60">
                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 w-12"></th>
                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">Shipment ID</th>
                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">Order ID</th>
                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">Box SKU</th>
                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">Void %</th>
                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">Material (g)</th>
                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">CO2e (kg)</th>
                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 text-center">Damage</th>
                <th className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">Packed At</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-700/50">
              {isLoading ? (
                <tr>
                  <td colSpan="9" className="px-6 py-16 text-center">
                    <div className="flex flex-col items-center gap-3">
                      <RefreshCw className="h-8 w-8 text-brand-green animate-spin" />
                      <span className="text-sm font-semibold text-slate-500 dark:text-slate-400">Loading shipments history...</span>
                    </div>
                  </td>
                </tr>
              ) : isError ? (
                <tr>
                  <td colSpan="9" className="px-6 py-16 text-center">
                    <div className="flex flex-col items-center gap-2 text-rose-500">
                      <AlertTriangle className="h-8 w-8" />
                      <span className="text-sm font-semibold">Failed to fetch shipment history.</span>
                    </div>
                  </td>
                </tr>
              ) : pageShipments.length === 0 ? (
                <tr>
                  <td colSpan="9" className="px-6 py-16 text-center text-slate-500 dark:text-slate-400">
                    No shipments found matching the filters.
                  </td>
                </tr>
              ) : (
                pageShipments.map((s) => {
                  const isExpanded = !!expandedRows[s.shipment_id];
                  
                  // Color code void %
                  let voidColor = 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20';
                  if (s.void_volume_pct >= 35) {
                    voidColor = 'bg-rose-500/10 text-rose-500 border-rose-500/20';
                  } else if (s.void_volume_pct >= 20) {
                    voidColor = 'bg-amber-500/10 text-amber-500 border-amber-500/20';
                  }

                  return (
                    <React.Fragment key={s.shipment_id}>
                      <tr className="hover:bg-slate-50/50 dark:hover:bg-slate-800/30 transition-colors duration-150">
                        <td className="px-6 py-4">
                          <button
                            onClick={() => toggleRow(s.shipment_id)}
                            className="p-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-700/60 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-400 transition-colors focus:outline-none"
                          >
                            <Eye className="h-4 w-4" />
                          </button>
                        </td>
                        <td className="px-6 py-4 font-mono text-xs font-bold text-slate-700 dark:text-slate-300" title={s.shipment_id}>
                          {s.shipment_id.slice(0, 8)}...
                        </td>
                        <td className="px-6 py-4 text-sm font-semibold text-slate-800 dark:text-slate-200">
                          {s.order_id || 'N/A'}
                        </td>
                        <td className="px-6 py-4">
                          <span className="inline-flex px-2.5 py-0.5 rounded-full text-xs font-bold bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-600/50">
                            {s.box_sku || 'N/A'}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <span className={`inline-flex px-2 py-0.5 rounded-full text-xs font-black border ${voidColor}`}>
                            {s.void_volume_pct !== null ? `${s.void_volume_pct.toFixed(1)}%` : '0.0%'}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-sm font-semibold text-slate-800 dark:text-slate-200">
                          {s.material_weight_g !== null ? `${s.material_weight_g.toFixed(0)}g` : '0g'}
                        </td>
                        <td className="px-6 py-4 text-sm font-semibold text-slate-800 dark:text-slate-200">
                          {s.co2e_kg !== null ? `${s.co2e_kg.toFixed(3)} kg` : '0.000 kg'}
                        </td>
                        <td className="px-6 py-4 text-center">
                          <div className="flex justify-center">
                            {s.damage_reported ? (
                              <AlertTriangle className="h-5 w-5 text-rose-500" title="Damage reported" />
                            ) : (
                              <CheckCircle className="h-5 w-5 text-emerald-500" title="No damage" />
                            )}
                          </div>
                        </td>
                        <td className="px-6 py-4 text-xs font-bold text-slate-500 dark:text-slate-400">
                          {s.packed_at ? new Date(s.packed_at).toLocaleString() : 'N/A'}
                        </td>
                      </tr>
                      {isExpanded && (
                        <tr className="bg-slate-50/50 dark:bg-slate-900/20">
                          <td colSpan="9" className="px-6 py-4">
                            <div className="flex items-start gap-3 text-left p-3 border border-slate-200 dark:border-slate-800 rounded-xl bg-white dark:bg-slate-800/40">
                              <Info className="h-5 w-5 text-brand-green shrink-0 mt-0.5" />
                              <div className="space-y-1.5">
                                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800 dark:text-slate-200">Shipment Details & Metadata</h4>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-1 text-xs">
                                  <div><span className="text-slate-400">Full Shipment ID:</span> <span className="font-mono text-slate-700 dark:text-slate-300 select-all">{s.shipment_id}</span></div>
                                  <div><span className="text-slate-400">Originated Order:</span> <span className="font-semibold text-slate-700 dark:text-slate-300">{s.order_id || 'N/A'}</span></div>
                                  <div><span className="text-slate-400">Damage Claims:</span> <span className="font-semibold text-slate-700 dark:text-slate-300">{s.damage_reported ? 'Reported (Pending review)' : 'None'}</span></div>
                                  <div><span className="text-slate-400">Box SKU:</span> <span className="font-mono text-slate-700 dark:text-slate-300">{s.box_sku || 'N/A'}</span></div>
                                </div>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination controls */}
        {data && data.pages > 1 && (
          <div className="flex items-center justify-between px-6 py-4 bg-slate-50 dark:bg-slate-900/20 border-t border-slate-100 dark:border-slate-700/60">
            <span className="text-xs text-slate-500 dark:text-slate-400">
              Showing page <b>{data.page}</b> of <b>{data.pages}</b> (Total: <b>{data.total}</b> records)
            </span>
            <div className="flex gap-2">
              <button
                disabled={page <= 1 || isPlaceholderData}
                onClick={() => handlePageChange(page - 1)}
                className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/80 text-xs font-bold text-slate-500 dark:text-slate-400 disabled:opacity-40 transition-colors"
              >
                <ChevronLeft className="h-4 w-4" />
                Prev
              </button>
              <button
                disabled={page >= data.pages || isPlaceholderData}
                onClick={() => handlePageChange(page + 1)}
                className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/80 text-xs font-bold text-slate-500 dark:text-slate-400 disabled:opacity-40 transition-colors"
              >
                Next
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
