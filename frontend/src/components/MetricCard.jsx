import React from 'react';
import { ArrowUpRight, ArrowDownRight } from 'lucide-react';

export const MetricCard = ({ title, value, change, changeType, icon: Icon, unit }) => {
  const isPositive = changeType === 'positive';
  const isNeutral = changeType === 'neutral';

  return (
    <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 rounded-2xl p-5 shadow-lg flex items-center justify-between transition-all hover:shadow-xl duration-200">
      <div className="space-y-2 text-left">
        <span className="text-[10px] font-extrabold uppercase tracking-wider text-slate-400">
          {title}
        </span>
        <div className="flex items-baseline gap-1">
          <span className="text-2xl font-black text-slate-900 dark:text-white">
            {value}
          </span>
          {unit && (
            <span className="text-xs font-semibold text-slate-500">
              {unit}
            </span>
          )}
        </div>
        
        {change && (
          <div className="flex items-center gap-1">
            {isNeutral ? null : isPositive ? (
              <ArrowUpRight className="h-3.5 w-3.5 text-brand-green" />
            ) : (
              <ArrowDownRight className="h-3.5 w-3.5 text-brand-red" />
            )}
            <span
              className={`text-xs font-bold ${
                isNeutral
                  ? 'text-slate-400'
                  : isPositive
                  ? 'text-brand-green'
                  : 'text-brand-red'
              }`}
            >
              {change}
            </span>
          </div>
        )}
      </div>

      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-slate-50 dark:bg-slate-900/60 text-slate-500 border border-slate-100 dark:border-slate-700/50">
        <Icon className="h-5.5 w-5.5 text-brand-green" />
      </div>
    </div>
  );
};
