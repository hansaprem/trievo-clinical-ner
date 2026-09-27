import React from 'react';
import { Binary, Cpu, FileText, Layers, ShieldCheck } from 'lucide-react';

export const PipelineFlow: React.FC = () => {
  return (
    <section id="pipeline" className="py-10 border-b border-slate-200/80">
      <div className="text-center max-w-3xl mx-auto mb-8">
        <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700 text-xs font-semibold uppercase tracking-wider font-mono">
          <Binary className="w-3.5 h-3.5 text-indigo-600" />
          <span>Multi-Model Inference Lifecycle</span>
        </div>
        <h3 className="mt-2 text-2xl font-extrabold text-slate-900 tracking-tight">
          From Clinical Text to Structured Knowledge
        </h3>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed">
          Four independently fine-tuned PubMedBERT checkpoints process input text in parallel.
          Their raw token-level span predictions are resolved via deterministic multi-model deduplication.
        </p>
      </div>

      {/* Horizontal Interactive Pipeline Visual */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-3 relative items-stretch">
        {/* Step 1: Clinical Text */}
        <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs flex flex-col justify-between hover:border-indigo-200 transition-all">
          <div>
            <div className="w-8 h-8 rounded-lg bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 mb-3">
              <FileText className="w-4 h-4" />
            </div>
            <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider">Step 01</span>
            <h4 className="text-sm font-bold text-slate-900 mt-1">Clinical Text</h4>
            <p className="text-xs text-slate-500 mt-1 leading-relaxed">
              Unstructured clinical notes, pathology reports, or literature abstracts.
            </p>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-100 text-[11px] font-mono text-slate-400">
            Raw string format
          </div>
        </div>

        {/* Step 2: Tokenization & Windowing */}
        <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs flex flex-col justify-between hover:border-indigo-200 transition-all">
          <div>
            <div className="w-8 h-8 rounded-lg bg-sky-50 border border-sky-100 flex items-center justify-center text-sky-600 mb-3">
              <Binary className="w-4 h-4" />
            </div>
            <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider">Step 02</span>
            <h4 className="text-sm font-bold text-slate-900 mt-1">Tokenization</h4>
            <p className="text-xs text-slate-500 mt-1 leading-relaxed">
              WordPiece subwords with 384-token sliding window and 64-token overlap stride.
            </p>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-100 text-[11px] font-mono text-slate-400">
            Offsets preserved
          </div>
        </div>

        {/* Step 3: 4 Model Inference */}
        <div className="bg-indigo-50/50 p-4 rounded-2xl border border-indigo-200/80 shadow-xs flex flex-col justify-between md:col-span-1">
          <div>
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white mb-3 shadow-xs">
              <Cpu className="w-4 h-4" />
            </div>
            <span className="text-[10px] font-mono font-bold text-indigo-600 uppercase tracking-wider">Step 03</span>
            <h4 className="text-sm font-extrabold text-slate-900 mt-1">4 Model Inference</h4>
            <div className="mt-2 space-y-1 text-[11px] font-mono">
              <div className="px-1.5 py-0.5 rounded bg-white border border-indigo-100 text-slate-700 flex justify-between">
                <span>BC5CDR</span>
                <span className="text-[10px] text-teal-600 font-bold">Chem • Dis</span>
              </div>
              <div className="px-1.5 py-0.5 rounded bg-white border border-indigo-100 text-slate-700 flex justify-between">
                <span>NCBI Disease</span>
                <span className="text-[10px] text-rose-600 font-bold">Disease</span>
              </div>
              <div className="px-1.5 py-0.5 rounded bg-white border border-indigo-100 text-slate-700 flex justify-between">
                <span>JNLPBA</span>
                <span className="text-[10px] text-indigo-600 font-bold">Molecular</span>
              </div>
              <div className="px-1.5 py-0.5 rounded bg-white border border-indigo-100 text-slate-700 flex justify-between">
                <span>AnatEM</span>
                <span className="text-[10px] text-amber-600 font-bold">Anatomy</span>
              </div>
            </div>
          </div>
          <p className="mt-2 text-[10px] text-indigo-900/70 italic">
            *Independently trained models
          </p>
        </div>

        {/* Step 4: Entity Resolution */}
        <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs flex flex-col justify-between hover:border-indigo-200 transition-all">
          <div>
            <div className="w-8 h-8 rounded-lg bg-teal-50 border border-teal-100 flex items-center justify-center text-teal-600 mb-3">
              <Layers className="w-4 h-4" />
            </div>
            <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider">Step 04</span>
            <h4 className="text-sm font-bold text-slate-900 mt-1">Entity Resolution</h4>
            <p className="text-xs text-slate-500 mt-1 leading-relaxed">
              Deterministic deduplication, maximal span overlap handling, joint provenance merging.
            </p>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-100 text-[11px] font-mono text-slate-400">
            No score hallucination
          </div>
        </div>

        {/* Step 5: Structured Entities */}
        <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs flex flex-col justify-between hover:border-indigo-200 transition-all">
          <div>
            <div className="w-8 h-8 rounded-lg bg-emerald-50 border border-emerald-100 flex items-center justify-center text-emerald-600 mb-3">
              <ShieldCheck className="w-4 h-4" />
            </div>
            <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider">Step 05</span>
            <h4 className="text-sm font-bold text-slate-900 mt-1">Structured Entities</h4>
            <p className="text-xs text-slate-500 mt-1 leading-relaxed">
              Standardized JSON response with character offsets, confidence scores, and origin models.
            </p>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-100 text-[11px] font-mono text-emerald-600 font-bold">
            8 Entity Ontology
          </div>
        </div>
      </div>
    </section>
  );
};
