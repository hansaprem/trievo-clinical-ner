import React from 'react';
import { Layers } from 'lucide-react';

interface ModelInfo {
  name: string;
  dataset: string;
  entities: string[];
  description: string;
}

const MODELS: ModelInfo[] = [
  {
    name: 'BC5CDR',
    dataset: 'BioCreative V CDR Corpus',
    entities: ['Chemical', 'Disease'],
    description: 'Chemical compounds and disease entity extraction in biomedical texts.',
  },
  {
    name: 'NCBI Disease',
    dataset: 'NCBI Disease Corpus',
    entities: ['Disease'],
    description: 'Specialized disease mention recognition from PubMed abstracts.',
  },
  {
    name: 'JNLPBA',
    dataset: 'BioNLP 2004 / JNLPBA',
    entities: ['Protein', 'DNA', 'RNA', 'Cell Type', 'Cell Line'],
    description: 'Molecular biology, genomics, and cellular entity identification.',
  },
  {
    name: 'AnatEM',
    dataset: 'Anatomy Entity Mention Corpus',
    entities: ['Anatomy'],
    description: 'Anatomical structures, organs, tissues, and organism subdivisions.',
  },
];

export const ModelCoverage: React.FC = () => {
  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs p-5 transition-all">
      <div className="flex items-center space-x-2 mb-3 border-b border-slate-100 pb-2">
        <Layers className="w-4 h-4 text-indigo-600" />
        <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
          Multi-Model Ensemble Coverage
        </h3>
        <span className="text-[11px] text-slate-400 font-mono">Factual Architecture</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {MODELS.map((model) => (
          <div
            key={model.name}
            className="p-3.5 rounded-xl border border-slate-200/70 bg-slate-50/50 hover:bg-slate-50 transition-all flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-900 text-sm">{model.name}</span>
                <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-white text-slate-600 border border-slate-200">
                  PubMedBERT
                </span>
              </div>
              <p className="text-[11px] text-slate-500 mt-1 line-clamp-2">
                {model.description}
              </p>
            </div>

            <div className="mt-3 pt-2 border-t border-slate-200/60">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 block mb-1">
                Target Entities
              </span>
              <div className="flex flex-wrap gap-1">
                {model.entities.map((ent) => (
                  <span
                    key={ent}
                    className="text-[11px] font-medium px-2 py-0.5 rounded-md bg-white text-slate-700 border border-slate-200 shadow-2xs"
                  >
                    {ent}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
