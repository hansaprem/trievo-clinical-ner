import React from 'react';
import { Entity } from '../types/ner';
import { Activity, Beaker, Dna, HeartPulse, Sparkles } from 'lucide-react';

interface StatisticsCardsProps {
  entities: Entity[];
}

export const StatisticsCards: React.FC<StatisticsCardsProps> = ({ entities }) => {
  if (!entities || entities.length === 0) return null;

  const total = entities.length;
  const diseases = entities.filter((e) => e.label === 'DISEASE').length;
  const chemicals = entities.filter((e) => e.label === 'CHEMICAL').length;
  const anatomy = entities.filter((e) => e.label === 'ANATOMY').length;
  const molecular = entities.filter((e) =>
    ['PROTEIN', 'DNA', 'RNA', 'CELL_TYPE', 'CELL_LINE'].includes(e.label)
  ).length;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-indigo-600" />
          <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
            Inference Metrics
          </h3>
        </div>
        <span className="text-[11px] text-emerald-700 font-mono bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
          Generated from current inference
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        {/* Total Entities */}
        <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500">Total Entities</span>
            <Activity className="w-4 h-4 text-indigo-500" />
          </div>
          <div className="mt-2 text-2xl sm:text-3xl font-extrabold text-slate-900 font-mono">{total}</div>
          <div className="text-[10px] text-slate-400 mt-1 font-mono">Deduplicated spans</div>
        </div>

        {/* Diseases */}
        <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500">Diseases</span>
            <HeartPulse className="w-4 h-4 text-rose-500" />
          </div>
          <div className="mt-2 text-2xl sm:text-3xl font-extrabold text-rose-700 font-mono">{diseases}</div>
          <div className="text-[10px] text-rose-600/70 mt-1 font-mono">Pathology</div>
        </div>

        {/* Chemicals */}
        <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500">Chemicals</span>
            <Beaker className="w-4 h-4 text-teal-500" />
          </div>
          <div className="mt-2 text-2xl sm:text-3xl font-extrabold text-teal-700 font-mono">{chemicals}</div>
          <div className="text-[10px] text-teal-600/70 mt-1 font-mono">Pharmacology</div>
        </div>

        {/* Anatomy */}
        <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500">Anatomy</span>
            <Sparkles className="w-4 h-4 text-amber-500" />
          </div>
          <div className="mt-2 text-2xl sm:text-3xl font-extrabold text-amber-700 font-mono">{anatomy}</div>
          <div className="text-[10px] text-amber-600/70 mt-1 font-mono">Organs &amp; Tissues</div>
        </div>

        {/* Molecular / Genetics */}
        <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs col-span-2 sm:col-span-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500">Molecular</span>
            <Dna className="w-4 h-4 text-indigo-500" />
          </div>
          <div className="mt-2 text-2xl sm:text-3xl font-extrabold text-indigo-700 font-mono">{molecular}</div>
          <div className="text-[10px] text-indigo-600/70 mt-1 font-mono">Protein / DNA / Cell</div>
        </div>
      </div>
    </div>
  );
};
