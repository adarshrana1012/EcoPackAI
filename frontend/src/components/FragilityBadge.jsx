import React from 'react';
import { CheckCircle2, ShieldAlert, AlertTriangle, AlertOctagon } from 'lucide-react';

export const FragilityBadge = ({ tier }) => {
  const configs = {
    0: {
      label: 'None',
      classes: 'bg-emerald-100/80 text-emerald-800 border-emerald-200/50 dark:bg-emerald-950/30 dark:text-emerald-400 dark:border-emerald-900/50',
      icon: CheckCircle2,
    },
    1: {
      label: 'Low',
      classes: 'bg-sky-100/80 text-sky-800 border-sky-200/50 dark:bg-sky-950/30 dark:text-sky-400 dark:border-sky-900/50',
      icon: ShieldAlert,
    },
    2: {
      label: 'Medium',
      classes: 'bg-amber-100/80 text-amber-800 border-amber-200/50 dark:bg-amber-950/30 dark:text-amber-400 dark:border-amber-900/50',
      icon: AlertTriangle,
    },
    3: {
      label: 'Critical',
      classes: 'bg-rose-100/80 text-rose-800 border-rose-200/50 dark:bg-rose-950/30 dark:text-rose-400 dark:border-rose-900/50',
      icon: AlertOctagon,
    },
  };

  const config = configs[tier] || configs[0];
  const Icon = config.icon;

  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${config.classes}`}>
      <Icon className="h-3.5 w-3.5" />
      {config.label}
    </span>
  );
};
