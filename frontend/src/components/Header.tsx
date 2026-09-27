import React, { useState } from 'react';
import { Activity, Menu, RefreshCw, X } from 'lucide-react';
import { HealthResponse } from '../types/ner';

interface HeaderProps {
  health: HealthResponse | null;
  healthLoading: boolean;
  onRefreshHealth: () => void;
}

export const Header: React.FC<HeaderProps> = ({ health, healthLoading, onRefreshHealth }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const isHealthy = health?.status === 'healthy' && health?.models_loaded;

  const navLinks = [
    { label: 'Overview', href: '#overview' },
    { label: 'Pipeline', href: '#pipeline' },
    { label: 'Analyze', href: '#analyze' },
    { label: 'Examples', href: '#examples' },
    { label: 'Ontology', href: '#ontology' },
    { label: 'Models', href: '#models' },
    { label: 'Benchmark', href: '#benchmark' },
  ];

  return (
    <header className="border-b border-slate-200/90 bg-white/95 backdrop-blur-md sticky top-0 z-40 shadow-2xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand & Title */}
        <a href="#overview" className="flex items-center space-x-3 group">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-700 via-indigo-600 to-sky-600 flex items-center justify-center text-white shadow-sm shadow-indigo-200 group-hover:scale-105 transition-transform">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-mono font-extrabold tracking-widest text-indigo-600 uppercase">TRIEVO</span>
              <span className="text-slate-300">|</span>
              <h1 className="text-base font-bold tracking-tight text-slate-900">Clinical NER</h1>
              <span className="text-[10px] px-1.5 py-0.2 rounded font-mono font-medium bg-slate-100 text-slate-600 border border-slate-200">
                FYP
              </span>
            </div>
            <p className="text-[11px] text-slate-500 font-medium">
              Biomedical Named Entity Recognition
            </p>
          </div>
        </a>

        {/* Desktop Navigation Links */}
        <nav className="hidden lg:flex items-center space-x-1">
          {navLinks.map((link) => (
            <a
              key={link.label}
              href={link.href}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-600 hover:text-indigo-600 hover:bg-slate-50 transition-colors"
            >
              {link.label}
            </a>
          ))}
        </nav>

        {/* API Health Status Indicator */}
        <div className="flex items-center space-x-2">
          <div
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-full border text-xs font-medium transition-all ${
              healthLoading
                ? 'bg-slate-50 border-slate-200 text-slate-500'
                : isHealthy
                ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
                : 'bg-rose-50 border-rose-200 text-rose-800'
            }`}
          >
            <span className="relative flex h-2 w-2">
              {isHealthy && (
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              )}
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${
                  healthLoading
                    ? 'bg-slate-400'
                    : isHealthy
                    ? 'bg-emerald-500'
                    : 'bg-rose-500'
                }`}
              ></span>
            </span>

            <span className="font-mono">
              {healthLoading
                ? 'Connecting...'
                : isHealthy
                ? `API Connected (${health?.models?.length || 4} Models)`
                : 'API Offline'}
            </span>

            <button
              onClick={onRefreshHealth}
              disabled={healthLoading}
              title="Refresh connection status"
              className="ml-1 text-slate-400 hover:text-slate-600 disabled:opacity-50 transition-colors p-0.5"
              aria-label="Refresh API Status"
            >
              <RefreshCw className={`w-3 h-3 ${healthLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {/* Mobile Menu Toggle Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="lg:hidden p-1.5 rounded-lg text-slate-600 hover:bg-slate-100"
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div className="lg:hidden border-t border-slate-200 bg-white px-4 py-3 space-y-1 shadow-md">
          {navLinks.map((link) => (
            <a
              key={link.label}
              href={link.href}
              onClick={() => setMobileMenuOpen(false)}
              className="block px-3 py-2 rounded-lg text-sm font-medium text-slate-700 hover:bg-indigo-50 hover:text-indigo-600"
            >
              {link.label}
            </a>
          ))}
        </div>
      )}
    </header>
  );
};
