import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import {
  LayoutDashboard,
  FileSearch,
  Boxes,
  BarChart3,
  Settings,
  ChevronLeft,
  ChevronRight,
  History
} from 'lucide-react';

export const Sidebar = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const { user } = useAuth();
  const isAdmin = user?.roles?.includes('admin');

  const navItems = [
    { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
    { name: 'Classify Item', path: '/classify', icon: FileSearch },
    { name: 'Pack Order', path: '/pack', icon: Boxes },
    { name: 'Analytics', path: '/analytics', icon: BarChart3 },
    { name: 'History', path: '/shipments', icon: History },
    ...(isAdmin ? [{ name: 'Admin Controls', path: '/admin', icon: Settings }] : []),
  ];

  return (
    <aside
      className={`relative flex flex-col border-r border-slate-200/80 dark:border-slate-800/80 bg-white dark:bg-slate-900 transition-all duration-350 ease-in-out ${
        isCollapsed ? 'w-16' : 'w-64'
      }`}
      aria-label="Sidebar Navigation"
    >
      {/* Navigation Links */}
      <nav className="flex-1 space-y-1.5 px-3 py-6">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group relative focus:outline-none focus:ring-2 focus:ring-brand-green/30 ${
                  isActive
                    ? 'bg-brand-green text-white font-medium shadow-md shadow-brand-green/20'
                    : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/50 hover:text-slate-900 dark:hover:text-white'
                }`
              }
              aria-label={item.name}
            >
              <Icon className="h-5 w-5 shrink-0" />
              {!isCollapsed && (
                <span className="text-sm tracking-wide transition-opacity duration-300">
                  {item.name}
                </span>
              )}
              {/* Tooltip for Collapsed Sidebar */}
              {isCollapsed && (
                <div className="absolute left-16 z-50 invisible group-hover:visible bg-slate-800 dark:bg-slate-700 text-white text-xs px-2.5 py-1.5 rounded-md font-medium whitespace-nowrap shadow-lg transition-all duration-150">
                  {item.name}
                </div>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Collapse Toggle Button at bottom */}
      <div className="p-3 border-t border-slate-200/80 dark:border-slate-800/80 flex justify-end">
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400 transition-colors focus:outline-none focus:ring-2 focus:ring-brand-green/30"
        >
          {isCollapsed ? <ChevronRight className="h-4.5 w-4.5" /> : <ChevronLeft className="h-4.5 w-4.5" />}
        </button>
      </div>
    </aside>
  );
};
