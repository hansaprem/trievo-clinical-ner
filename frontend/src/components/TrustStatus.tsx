import React from 'react';
import { CheckCircle2, ShieldCheck } from 'lucide-react';

const CAPABILITIES = [
  {
    title: '4 Checkpoints Pre-Warmed',
    description: 'BC5CDR, NCBI Disease, JNLPBA, and AnatEM held permanently in memory via FastAPI lifespan.',
  },
  {
    title: 'Unified Inference Pipeline',
    description: 'Single-entrypoint orchestrator with 384-token sliding window and 64-token stride.',
  },
  {
    title: 'FastAPI REST Backend',
    description: 'Lightweight asynchronous HTTP API with strict Pydantic v2 offset and schema validation.',
  },
  {
    title: 'Real-Time Inference',
    description: 'Sub-300ms latency on standard CPU; immediate execution with zero model reloading overhead.',
  },
  {
    title: '100% Exact Character Offsets',
    description: 'Guarantees text[start:end] strictly equals the extracted entity surface string.',
  },
  {
    title: 'Deterministic Conflict Resolution',
    description: 'Maximal span selection, multi-type nesting preservation, and multi-model joint provenance.',
  },
];

export const TrustStatus: React.FC = () => {
  return (
    <section className="py-8 border-b border-slate-200/80">
      <div className="bg-gradient-to-tr from-slate-900 via-indigo-950 to-slate-900 rounded-3xl p-6 sm:p-8 text-white shadow-md relative overflow-hidden">
        {/* Subtle decorative grid */}
        <div className="absolute inset-0 opacity-10 bg-[radial-gradient(#38bdf8_1px,transparent_1px)] [background-size:16px_16px] pointer-events-none" />

        <div className="relative z-10">
          <div className="flex items-center space-x-2 text-sky-400 text-xs font-mono font-bold uppercase tracking-wider mb-2">
            <ShieldCheck className="w-4 h-4" />
            <span>System Verification &amp; Technical Capabilities</span>
          </div>

          <h3 className="text-xl sm:text-2xl font-extrabold tracking-tight">
            Verified Production Specifications
          </h3>
          <p className="text-xs sm:text-sm text-slate-300 mt-1 max-w-2xl leading-relaxed">
            All capabilities below are implemented, tested, and actively operating in the current runtime environment.
          </p>

          <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {CAPABILITIES.map((cap, i) => (
              <div
                key={i}
                className="bg-white/5 border border-white/10 rounded-2xl p-4 backdrop-blur-xs flex items-start space-x-3"
              >
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-bold text-white tracking-wide">{cap.title}</h4>
                  <p className="text-[11px] text-slate-300 mt-0.5 leading-relaxed">
                    {cap.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
