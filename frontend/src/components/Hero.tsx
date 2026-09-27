import React from 'react';
import { ArrowDown, Cpu, Database, Layers, Sparkles, Zap } from 'lucide-react';
import { BiomedicalNeuralCore } from './BiomedicalNeuralCore';

export const Hero: React.FC = () => {
  return (
    <section id="overview" className="pt-8 pb-12 border-b border-slate-200/80">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
        {/* Left Column: Headline, Description & CTAs */}
        <div className="lg:col-span-7 space-y-6">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
            <span>Biomedical NLP &amp; Clinical Token Classification</span>
          </div>

          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-slate-900 tracking-tight leading-tight">
            Clinical Intelligence, <br className="hidden sm:inline" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-700 via-sky-600 to-teal-600">
              Extracted from Text.
            </span>
          </h2>

          <p className="text-base sm:text-lg text-slate-600 leading-relaxed max-w-2xl">
            Identify clinically relevant biomedical entities from unstructured text using a unified
            multi-model PubMedBERT inference pipeline.
          </p>

          {/* Action CTAs */}
          <div className="flex flex-wrap items-center gap-3 pt-2">
            <a
              href="#analyze"
              className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-indigo-600 text-white font-semibold text-sm hover:bg-indigo-700 active:bg-indigo-800 shadow-sm shadow-indigo-200 transition-all group"
            >
              <span>Analyze Clinical Text</span>
              <ArrowDown className="w-4 h-4 group-hover:translate-y-0.5 transition-transform" />
            </a>

            <a
              href="#examples"
              className="inline-flex items-center space-x-2 px-5 py-3 rounded-xl bg-white text-slate-700 font-semibold text-sm hover:bg-slate-50 border border-slate-200 shadow-2xs transition-all"
            >
              <span>Explore Examples (8)</span>
            </a>
          </div>

          {/* Factual Badges Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 border-t border-slate-100">
            <div className="bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs flex items-center space-x-2.5">
              <div className="w-7 h-7 rounded-lg bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 shrink-0">
                <Layers className="w-3.5 h-3.5" />
              </div>
              <div>
                <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Architecture</div>
                <div className="text-xs font-bold text-slate-800">4 Trained Models</div>
              </div>
            </div>

            <div className="bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs flex items-center space-x-2.5">
              <div className="w-7 h-7 rounded-lg bg-teal-50 border border-teal-100 flex items-center justify-center text-teal-600 shrink-0">
                <Database className="w-3.5 h-3.5" />
              </div>
              <div>
                <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Taxonomy</div>
                <div className="text-xs font-bold text-slate-800">8 Entity Types</div>
              </div>
            </div>

            <div className="bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs flex items-center space-x-2.5">
              <div className="w-7 h-7 rounded-lg bg-sky-50 border border-sky-100 flex items-center justify-center text-sky-600 shrink-0">
                <Cpu className="w-3.5 h-3.5" />
              </div>
              <div>
                <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Language Model</div>
                <div className="text-xs font-bold text-slate-800">PubMedBERT</div>
              </div>
            </div>

            <div className="bg-white p-3 rounded-xl border border-slate-200/80 shadow-2xs flex items-center space-x-2.5">
              <div className="w-7 h-7 rounded-lg bg-emerald-50 border border-emerald-100 flex items-center justify-center text-emerald-600 shrink-0">
                <Zap className="w-3.5 h-3.5" />
              </div>
              <div>
                <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Pipeline</div>
                <div className="text-xs font-bold text-slate-800">Real-Time API</div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Subtle 3D Biomedical Neural Core */}
        <div className="lg:col-span-5 flex justify-center">
          <div className="w-full max-w-md bg-white rounded-3xl border border-slate-200/90 shadow-sm p-3 relative overflow-hidden">
            <div className="absolute top-3 left-4 z-10 flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-slate-400">
                Interactive Neural Core (Three.js)
              </span>
            </div>
            <BiomedicalNeuralCore />
          </div>
        </div>
      </div>
    </section>
  );
};
