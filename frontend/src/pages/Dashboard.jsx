import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { FileSearch, Boxes, BarChart3, Settings, Shield, Cpu, Sparkles } from 'lucide-react';

export const Dashboard = () => {
  const { user } = useAuth();
  const isAdmin = user?.roles?.includes('admin');

  const modules = [
    {
      title: 'Fragility Classifier',
      desc: 'Predict product fragility tier using machine learning inference.',
      path: '/classify',
      icon: FileSearch,
      color: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20',
    },
    {
      title: '3D Packing Engine',
      desc: 'Optimize shipments to minimize void volume and material weight.',
      path: '/pack',
      icon: Boxes,
      color: 'bg-sky-500/10 text-sky-500 border-sky-500/20',
    },
    {
      title: 'Analytics Dashboard',
      desc: 'Monitor weekly usage, carbon savings, and void distributions.',
      path: '/analytics',
      icon: BarChart3,
      color: 'bg-amber-500/10 text-amber-500 border-amber-500/20',
    },
    ...(isAdmin
      ? [
          {
            title: 'Admin Controls',
            desc: 'Adjust policy routing, A/B config split, and trigger retrains.',
            path: '/admin',
            icon: Settings,
            color: 'bg-purple-500/10 text-purple-500 border-purple-500/20',
          },
        ]
      : []),
  ];

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-8 bg-slate-50 dark:bg-slate-900/50">
      {/* Welcome Banner */}
      <div className="relative bg-white dark:bg-slate-900 text-slate-900 dark:text-white rounded-3xl p-6 md:p-8 overflow-hidden shadow-xl border border-slate-200 dark:border-slate-800 transition-colors duration-300">
        <div className="absolute top-1/2 right-10 -translate-y-1/2 opacity-10 pointer-events-none">
          <Sparkles className="h-64 w-64 text-brand-green" />
        </div>
        
        <div className="relative z-10 max-w-xl text-left space-y-3">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider bg-brand-green/20 text-brand-green border border-brand-green/30">
            <Cpu className="h-3 w-3 animate-spin" />
            System Live
          </span>
          <h1 className="text-3xl font-black md:text-4xl tracking-tight leading-none">
            Hello, {user?.name || 'User'}!
          </h1>
          <p className="text-slate-500 dark:text-slate-400 text-sm leading-relaxed">
            Welcome to the EcoPackAI optimization suite. Select a capability below to begin optimizing containers, classifying materials, or evaluating operational savings.
          </p>
        </div>
      </div>

      {/* Main Grid */}
      <div className="space-y-4">
        <h2 className="text-sm font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider">
          System Core Modules
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {modules.map((m) => {
            const Icon = m.icon;
            return (
              <Link
                key={m.path}
                to={m.path}
                className="group bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 rounded-2xl p-6 shadow hover:shadow-xl transition-all duration-200 text-left flex flex-col justify-between h-48 hover:border-brand-green/30"
              >
                <div>
                  <div className={`inline-flex h-10 w-10 items-center justify-center rounded-xl border ${m.color} mb-4`}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <h3 className="text-base font-bold text-slate-900 dark:text-white group-hover:text-brand-green transition-colors">
                    {m.title}
                  </h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
                    {m.desc}
                  </p>
                </div>
                <span className="text-[10px] font-bold text-slate-400 group-hover:text-brand-green transition-colors mt-4 inline-flex items-center gap-1">
                  Launch Module &rarr;
                </span>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Platform Info Footer Card */}
      <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700/80 p-6 rounded-2xl shadow flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3 text-left">
          <div className="h-10 w-10 rounded-full bg-brand-green/10 flex items-center justify-center text-brand-green">
            <Shield className="h-5 w-5" />
          </div>
          <div>
            <h4 className="text-sm font-bold text-slate-800 dark:text-slate-200">Production-Grade Optimization</h4>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Leveraging Random Forest classifiers and Reinforcement Learning packing engines.
            </p>
          </div>
        </div>
        <div className="text-xs text-slate-400 font-mono">
          v1.0.0 Stable
        </div>
      </div>
    </div>
  );
};
