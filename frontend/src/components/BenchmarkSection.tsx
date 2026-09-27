import React from 'react';
import { BarChart3 } from 'lucide-react';
import { BenchmarkMetrics, ErrorAnalysisCategory } from '../types/ner';

export const BENCHMARKS: BenchmarkMetrics[] = [
  {
    dataset: 'BC5CDR',
    domain: 'Biomedical Literature',
    testSentences: 900,
    precision: 74.98,
    recall: 78.76,
    f1: 76.83,
    entityTypes: 'Chemical, Disease',
    checkpoint: 'models/bc5cdr_pubmedbert',
  },
  {
    dataset: 'NCBI Disease',
    domain: 'Biomedical Literature',
    testSentences: 900,
    precision: 75.81,
    recall: 75.81,
    f1: 75.81,
    entityTypes: 'Disease',
    checkpoint: 'models/ncbi_disease_pubmedbert',
  },
  {
    dataset: 'JNLPBA',
    domain: 'Molecular Biology',
    testSentences: 900,
    precision: 65.85,
    recall: 75.69,
    f1: 70.43,
    entityTypes: 'Protein, DNA, RNA, Cell Type, Cell Line',
    checkpoint: 'models/jnlpba_pubmedbert',
  },
  {
    dataset: 'AnatEM',
    domain: 'Biomedical Literature',
    testSentences: 900,
    precision: 81.44,
    recall: 77.83,
    f1: 79.59,
    entityTypes: 'Anatomy',
    checkpoint: 'models/anatem_pubmedbert',
  },
];

export const ERROR_CATEGORIES: ErrorAnalysisCategory[] = [
  {
    title: 'Span Boundary Inconsistencies',
    percentageRange: '31.5% – 36.5% of total errors',
    description: 'Partial span mismatches where the model captures the core medical term but differs on modifier adjectives (e.g. "acute", "chronic", "bilateral").',
    clinicalSignificance: 'Entity semantics preserved; deterministic maximal span resolution resolves multi-model boundary overlaps.',
  },
  {
    title: 'Pure False Negatives (Omissions)',
    percentageRange: '14.1% – 42.8% of total errors',
    description: 'Uncommon synonyms, specialized abbreviations, or low-frequency anatomical terms absent in the training corpus.',
    clinicalSignificance: 'Addressed via multi-model ensemble overlap; missing disease terms in BC5CDR are often caught by NCBI Disease.',
  },
  {
    title: 'Pure False Positives (Over-Extraction)',
    percentageRange: '20.7% – 41.3% of total errors',
    description: 'General language terms with biomedical roots or borderline scientific jargon classified as entities.',
    clinicalSignificance: 'Restrained with confidence thresholds; false positive rates are highest in complex molecular text (JNLPBA).',
  },
  {
    title: 'Entity-Type Confusion',
    percentageRange: '< 2.0% in primary domains',
    description: 'Cross-type classification errors between distinct clinical categories.',
    clinicalSignificance: 'Negligible in disjoint domains; primarily occurs in molecular sub-taxonomies (e.g. DNA vs Protein vs RNA).',
  },
];

export const BenchmarkSection: React.FC = () => {
  return (
    <section id="benchmark" className="py-10 border-b border-slate-200/80">
      {/* Benchmark Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between mb-6 gap-2">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700 text-xs font-semibold uppercase tracking-wider font-mono">
            <BarChart3 className="w-3.5 h-3.5 text-indigo-600" />
            <span>Empirical Validation</span>
          </div>
          <h3 className="mt-2 text-2xl font-extrabold text-slate-900 tracking-tight">
            Research &amp; Validation Benchmark
          </h3>
          <p className="mt-1 text-sm text-slate-500">
            Factual performance evaluation across 900 standardized test sentences per dataset (3,600 total).
          </p>
        </div>
        <div className="text-xs font-mono text-slate-600 bg-slate-100 px-3 py-1 rounded-lg border border-slate-200">
          Macro-average F1: <strong className="text-indigo-600 font-bold">75.67%</strong>
        </div>
      </div>

      {/* Benchmark Metrics Table */}
      <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm overflow-hidden mb-8">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-50/80 border-b border-slate-200/80 text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                <th scope="col" className="py-3.5 px-4 sm:px-6">Dataset</th>
                <th scope="col" className="py-3.5 px-4 sm:px-6">Target Domain</th>
                <th scope="col" className="py-3.5 px-4 sm:px-6">Precision</th>
                <th scope="col" className="py-3.5 px-4 sm:px-6">Recall</th>
                <th scope="col" className="py-3.5 px-4 sm:px-6">Entity F1</th>
                <th scope="col" className="py-3.5 px-4 sm:px-6">Test Sentences</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-sm">
              {BENCHMARKS.map((item) => (
                <tr key={item.dataset} className="hover:bg-slate-50/70 transition-colors">
                  <td className="py-3.5 px-4 sm:px-6 font-bold text-slate-900">
                    <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-800">
                      {item.dataset}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 sm:px-6 text-xs text-slate-600 font-medium">
                    {item.domain}
                  </td>
                  <td className="py-3.5 px-4 sm:px-6 font-mono text-xs font-semibold text-slate-700">
                    {item.precision.toFixed(2)}%
                  </td>
                  <td className="py-3.5 px-4 sm:px-6 font-mono text-xs font-semibold text-slate-700">
                    {item.recall.toFixed(2)}%
                  </td>
                  <td className="py-3.5 px-4 sm:px-6">
                    <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-100">
                      {item.f1.toFixed(2)}%
                    </span>
                  </td>
                  <td className="py-3.5 px-4 sm:px-6 font-mono text-xs text-slate-500">
                    {item.testSentences} sentences
                  </td>
                </tr>
              ))}

              {/* Macro Average Row */}
              <tr className="bg-indigo-50/40 font-bold border-t-2 border-indigo-100">
                <td className="py-3.5 px-4 sm:px-6 font-mono text-xs text-indigo-900">
                  Macro-Average
                </td>
                <td className="py-3.5 px-4 sm:px-6 text-xs text-indigo-800">
                  Multi-Domain Standardized Average
                </td>
                <td className="py-3.5 px-4 sm:px-6 font-mono text-xs text-indigo-900">
                  74.52%
                </td>
                <td className="py-3.5 px-4 sm:px-6 font-mono text-xs text-indigo-900">
                  77.02%
                </td>
                <td className="py-3.5 px-4 sm:px-6">
                  <span className="font-mono text-xs font-extrabold px-2.5 py-0.5 rounded bg-indigo-600 text-white shadow-2xs">
                    75.67% F1
                  </span>
                </td>
                <td className="py-3.5 px-4 sm:px-6 font-mono text-xs text-indigo-800">
                  3,600 held-out
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="p-3 bg-slate-50/60 border-t border-slate-200/60 text-center text-[11px] text-slate-500 font-mono">
          *Evaluated on standardized 900-sentence test partitions using exact BIO token classification metrics.
        </div>
      </div>

      {/* Error Analysis Section */}
      <div className="space-y-4">
        <div>
          <h4 className="text-base font-extrabold text-slate-900 tracking-tight">
            Scientific Error Analysis (Empirical Breakdown)
          </h4>
          <p className="text-xs text-slate-500 mt-0.5">
            Systematic audit of 5,778 model predictions against gold-standard biomedical annotations.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {ERROR_CATEGORIES.map((cat, idx) => (
            <div
              key={idx}
              className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-2xs hover:border-slate-300 transition-all"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-sm text-slate-900">{cat.title}</span>
                <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-rose-50 text-rose-700 border border-rose-100">
                  {cat.percentageRange}
                </span>
              </div>
              <p className="text-xs text-slate-600 leading-relaxed mb-2">
                {cat.description}
              </p>
              <div className="text-[11px] text-indigo-900 bg-indigo-50/50 p-2.5 rounded-xl border border-indigo-100/60">
                <strong className="font-semibold">Pipeline Mitigation:</strong> {cat.clinicalSignificance}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
