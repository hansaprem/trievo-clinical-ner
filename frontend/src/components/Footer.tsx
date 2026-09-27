import React from 'react';
import { Activity, ShieldAlert } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="mt-16 border-t border-slate-200/80 bg-white py-10 text-slate-500 text-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="w-7 h-7 rounded-xl bg-gradient-to-tr from-indigo-700 to-sky-600 flex items-center justify-center text-white shadow-2xs">
              <Activity className="w-4 h-4" />
            </div>
            <div>
              <span className="font-extrabold text-slate-900 tracking-tight text-sm">
                TRIEVO CLINICAL NER
              </span>
              <p className="text-[11px] text-slate-500">
                Biomedical Named Entity Recognition using Multi-Model PubMedBERT
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4 text-[11px] text-slate-400 font-mono">
            <span className="px-2 py-0.5 rounded bg-slate-50 border border-slate-200 text-slate-600">
              Research Prototype
            </span>
            <span className="px-2 py-0.5 rounded bg-slate-50 border border-slate-200 text-slate-600">
              Clinical NLP
            </span>
            <span className="px-2 py-0.5 rounded bg-slate-50 border border-slate-200 text-slate-600">
              Biomedical AI
            </span>
          </div>
        </div>

        {/* Disclaimer Callout */}
        <div className="pt-4 border-t border-slate-100 flex items-start space-x-2.5 text-[11px] text-slate-400 leading-relaxed">
          <ShieldAlert className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
          <p>
            <strong>Research Prototype Disclaimer:</strong> This interface is a scientific demonstration prototype
            for multi-dataset biomedical token classification and named entity recognition. It does not provide medical
            diagnoses, treatment recommendations, clinical prognostications, or clinical decision-support advice.
          </p>
        </div>
      </div>
    </footer>
  );
};
