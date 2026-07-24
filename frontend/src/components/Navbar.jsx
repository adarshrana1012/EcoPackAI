import React from 'react';
import { useAuth } from '../contexts/AuthContext';
import { Leaf, LogOut, User } from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';

export const Navbar = () => {
  const { user, logout } = useAuth();

  return (
    <nav className="sticky top-0 z-40 w-full backdrop-blur-md bg-white/75 dark:bg-slate-900/75 border-b border-slate-200/80 dark:border-slate-800/80 transition-colors duration-300">
      <div className="mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo Section */}
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-green/10 text-brand-green">
              <Leaf className="h-5 w-5 animate-pulse" />
            </div>
            <span className="text-xl font-bold tracking-tight text-brand-dark dark:text-white">
              EcoPack<span className="text-brand-green">AI</span>
            </span>
          </div>

          {/* User Profile / Theme / Logout */}
          <div className="flex items-center gap-3">
            <ThemeToggle />
            
            <div className="flex items-center gap-3 bg-slate-100/85 dark:bg-slate-800/80 py-1.5 pl-3 pr-4 rounded-full border border-slate-200/50 dark:border-slate-700/50">
              <div className="flex h-7 w-7 items-center justify-center rounded-full bg-brand-green text-white font-medium text-sm">
                {user?.name ? user.name[0].toUpperCase() : <User className="h-4 w-4" />}
              </div>
              <div className="hidden sm:flex flex-col text-left">
                <span className="text-xs font-semibold text-slate-800 dark:text-slate-200 leading-tight">
                  {user?.name || 'Demo User'}
                </span>
                <span className="text-[10px] text-slate-500 dark:text-slate-400 leading-tight">
                  {user?.roles?.join(', ') || 'Guest'}
                </span>
              </div>
            </div>

            <button
              onClick={logout}
              aria-label="Logout"
              className="flex items-center justify-center h-9 w-9 rounded-lg border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:text-brand-red hover:border-brand-red/35 dark:hover:border-brand-red/35 hover:bg-brand-red/5 dark:hover:bg-brand-red/5 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-brand-red/20"
            >
              <LogOut className="h-4.5 w-4.5" />
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};
