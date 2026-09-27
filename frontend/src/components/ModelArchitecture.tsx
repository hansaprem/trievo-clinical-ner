import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Cpu } from 'lucide-react';
import { ModelDetail } from '../types/ner';

export const MODELS_DATA: ModelDetail[] = [
  {
    id: 'bc5cdr',
    name: 'BC5CDR Checkpoint',
    corpus: 'BioCreative V CDR Corpus (900-sentence test)',
    domain: 'Pharmacology & Disease Pathology',
    entityTypes: ['Chemical', 'Disease'],
    backbone: 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    inferenceRole: 'Primary extractor for pharmaceutical compounds, chemical molecules, and pathological disease mentions.',
    description: 'Fine-tuned with Cross-Entropy loss over 1,488 gold entity mentions. Resolves disease overlap with NCBI Disease.',
  },
  {
    id: 'ncbi_disease',
    name: 'NCBI Disease Checkpoint',
    corpus: 'NCBI Disease Corpus (900-sentence test)',
    domain: 'Clinical Disorder & Syndrome Pathology',
    entityTypes: ['Disease'],
    backbone: 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    inferenceRole: 'Specialized secondary classifier for complex and inherited clinical disorders in biomedical abstracts.',
    description: 'Provides corroboration and dual-provenance reinforcement for disease mentions detected concurrently by BC5CDR.',
  },
  {
    id: 'jnlpba',
    name: 'JNLPBA Checkpoint',
    corpus: 'BioNLP 2004 / JNLPBA (900-sentence test)',
    domain: 'Molecular Biology & Genetics',
    entityTypes: ['Protein', 'DNA', 'RNA', 'Cell Type', 'Cell Line'],
    backbone: 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    inferenceRole: 'High-density multi-class token classifier for biochemical, genetic, and physiological cell structures.',
    description: 'Evaluated across 1,946 gold entities with 31.48% boundary error margin. Handles complex genomic and protein terms.',
  },
  {
    id: 'anatem',
    name: 'AnatEM Checkpoint',
    corpus: 'AnatEM Anatomy Entity Mention Corpus (900-sentence test)',
    domain: 'Anatomical Structures & Tissues',
    entityTypes: ['Anatomy'],
    backbone: 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    inferenceRole: 'Dedicated anatomical structure extractor identifying organs, tissue layers, and gross anatomical systems.',
    description: 'Achieves 79.59% F1 and 81.44% precision. Captures fine-grained tissue margins and anatomical biopsy sites.',
  },
];

export const ModelArchitecture: React.FC = () => {
  const [expandedModel, setExpandedModel] = useState<string | null>('bc5cdr');

  return (
    <section id="models" className="py-10 border-b border-slate-200/80">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between mb-6 gap-2">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700 text-xs font-semibold uppercase tracking-wider font-mono">
            <Cpu className="w-3.5 h-3.5 text-indigo-600" />
            <span>Ensemble Specifications</span>
          </div>
          <h3 className="mt-2 text-2xl font-extrabold text-slate-900 tracking-tight">
            Multi-Model Clinical NER Architecture
          </h3>
          <p className="mt-1 text-sm text-slate-500">
            Four independent checkpoints executed simultaneously and unified at runtime.
          </p>
        </div>
        <div className="text-xs font-mono text-indigo-700 bg-indigo-50 px-2.5 py-1 rounded-lg border border-indigo-100">
          Independent Weights • Runtime Resolution
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {MODELS_DATA.map((model) => {
          const isExpanded = expandedModel === model.id;

          return (
            <div
              key={model.id}
              className={`p-5 rounded-2xl border transition-all ${
                isExpanded
                  ? 'bg-white border-indigo-300 shadow-sm ring-1 ring-indigo-500/20'
                  : 'bg-white border-slate-200/80 hover:border-slate-300 shadow-2xs'
              }`}
            >
              <div
                onClick={() => setExpandedModel(isExpanded ? null : model.id)}
                className="flex items-start justify-between cursor-pointer"
              >
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="font-extrabold text-base text-slate-900">{model.name}</span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200 font-semibold">
                      PubMedBERT
                    </span>
                  </div>
                  <div className="text-xs text-slate-500 mt-1 font-medium">{model.corpus}</div>
                </div>

                <button
                  type="button"
                  className="p-1 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
                  aria-label={isExpanded ? 'Collapse model details' : 'Expand model details'}
                >
                  {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
              </div>

              {/* Entity Badges */}
              <div className="mt-3 flex flex-wrap gap-1.5">
                {model.entityTypes.map((ent) => (
                  <span
                    key={ent}
                    className="text-xs font-medium px-2 py-0.5 rounded-md bg-indigo-50 text-indigo-800 border border-indigo-100"
                  >
                    {ent}
                  </span>
                ))}
              </div>

              {/* Expandable Technical Specification Drawer */}
              {isExpanded && (
                <div className="mt-4 pt-4 border-t border-slate-100 space-y-3 text-xs animate-in fade-in duration-200">
                  <div>
                    <span className="font-bold text-slate-700 block">Backbone Model:</span>
                    <code className="text-[11px] font-mono text-indigo-900 bg-slate-50 px-2 py-1 rounded border border-slate-200 block mt-1 break-all">
                      {model.backbone}
                    </code>
                  </div>

                  <div>
                    <span className="font-bold text-slate-700 block">Inference Role:</span>
                    <p className="text-slate-600 mt-0.5 leading-relaxed">{model.inferenceRole}</p>
                  </div>

                  <div>
                    <span className="font-bold text-slate-700 block">Technical Context:</span>
                    <p className="text-slate-500 mt-0.5 leading-relaxed">{model.description}</p>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
};
