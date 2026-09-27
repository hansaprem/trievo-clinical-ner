import React from 'react';
import { ArrowUpRight, Beaker, BookOpen, Dna, HeartPulse, Microscope, Pill, Sparkles } from 'lucide-react';
import { ExampleScenario } from '../types/ner';

interface ExampleLibraryProps {
  onSelectExample: (text: string, autoAnalyze?: boolean) => void;
  isLoading: boolean;
}

export const EXAMPLES: ExampleScenario[] = [
  {
    id: 'cat-1-general',
    category: 'General Clinical',
    title: 'Pulmonology Case',
    text: 'The patient was diagnosed with pneumonia and prescribed aspirin. CT imaging demonstrated involvement of the right lung, while elevated p53 protein levels were noted.',
    targetTags: ['Disease', 'Chemical', 'Anatomy', 'Protein'],
    purpose: 'Tests multi-domain extraction across respiratory pathology, drug therapy, and organ involvement.',
  },
  {
    id: 'cat-2-oncology',
    category: 'Oncology',
    title: 'Hematologic Malignancy',
    text: 'Acute lymphoblastic leukemia was identified in the bone marrow biopsy, and methotrexate chemotherapy was promptly initiated.',
    targetTags: ['Disease', 'Anatomy', 'Chemical'],
    purpose: 'Assesses oncological disease classification combined with anatomical biopsy site and chemotherapy agent.',
  },
  {
    id: 'cat-3-molecular',
    category: 'Molecular Biology',
    title: 'Cellular Signaling',
    text: 'Elevated p53 protein expression was observed in the human tumor cells, along with significantly increased cytoplasmic mRNA levels.',
    targetTags: ['Protein', 'Cell Type', 'RNA'],
    purpose: 'Evaluates molecular genetics entity recognition spanning functional proteins, cellular types, and transcripts.',
  },
  {
    id: 'cat-4-genetics',
    category: 'Genetics',
    title: 'Genomic Mutation',
    text: 'The study identified an oncogenic mutation adjacent to the c-myc gene and sequenced the corresponding target DNA locus.',
    targetTags: ['DNA', 'Gene', 'Sequence'],
    purpose: 'Focuses on nucleic acid and genomic sequence annotations extracted by the JNLPBA checkpoint.',
  },
  {
    id: 'cat-5-pharmacology',
    category: 'Pharmacology',
    title: 'Immunosuppressive Regimen',
    text: 'Following allograft surgery, the patient received tacrolimus, cyclosporine, and low-dose methotrexate during the maintenance period.',
    targetTags: ['Chemical', 'Pharmacology'],
    purpose: 'Tests compound extraction across multiple therapeutic agents and chemical inhibitors.',
  },
  {
    id: 'cat-6-anatomy',
    category: 'Anatomy',
    title: 'Visceral Tissue Pathology',
    text: 'Contrast imaging demonstrated localized hypodensity involving the left kidney cortex and posterior right lung parenchyma.',
    targetTags: ['Anatomy', 'Organs', 'Tissues'],
    purpose: 'Validates AnatEM anatomical extraction performance on fine-grained physiological and tissue mentions.',
  },
  {
    id: 'cat-7-mixed',
    category: 'Mixed Biomedical',
    title: 'Multidisciplinary Findings',
    text: 'A biopsy of the renal parenchyma in a patient with chronic glomerulonephritis confirmed antibody deposition and marked cellular infiltration.',
    targetTags: ['Anatomy', 'Disease', 'Cell Type'],
    purpose: 'Evaluates simultaneous multi-dataset complementary extraction across systemic disorders and tissues.',
  },
  {
    id: 'cat-8-custom',
    category: 'Custom Input',
    title: 'User-Defined Clinical Note',
    text: 'Enter your own clinical progress note, literature abstract, or patient case study in the editor above.',
    targetTags: ['User Text', 'Real-Time'],
    purpose: 'Allows supervisor and evaluators to input novel medical texts for unbiased evaluation.',
  },
];

export const ExampleLibrary: React.FC<ExampleLibraryProps> = ({ onSelectExample, isLoading }) => {
  const getIconForCategory = (cat: string) => {
    switch (cat) {
      case 'General Clinical':
        return <HeartPulse className="w-4 h-4 text-rose-500" />;
      case 'Oncology':
        return <Microscope className="w-4 h-4 text-indigo-500" />;
      case 'Molecular Biology':
        return <Dna className="w-4 h-4 text-purple-500" />;
      case 'Genetics':
        return <Dna className="w-4 h-4 text-fuchsia-500" />;
      case 'Pharmacology':
        return <Pill className="w-4 h-4 text-teal-500" />;
      case 'Anatomy':
        return <Beaker className="w-4 h-4 text-amber-500" />;
      case 'Mixed Biomedical':
        return <Sparkles className="w-4 h-4 text-sky-500" />;
      default:
        return <BookOpen className="w-4 h-4 text-slate-500" />;
    }
  };

  return (
    <section id="examples" className="py-10 border-b border-slate-200/80">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between mb-6 gap-2">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700 text-xs font-semibold uppercase tracking-wider font-mono">
            <BookOpen className="w-3.5 h-3.5 text-indigo-600" />
            <span>Interactive Evaluation Suite</span>
          </div>
          <h3 className="mt-2 text-2xl font-extrabold text-slate-900 tracking-tight">
            Curated Example Library (8 Categories)
          </h3>
          <p className="mt-1 text-sm text-slate-500">
            Select any scenario below to load the text into the editor and test the ensemble's live extraction capability.
          </p>
        </div>
        <span className="text-xs font-mono text-slate-400">Click card to test</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {EXAMPLES.map((ex) => (
          <div
            key={ex.id}
            onClick={() => {
              if (isLoading) return;
              if (ex.id === 'cat-8-custom') {
                const el = document.getElementById('clinical-text-input');
                el?.focus();
              } else {
                onSelectExample(ex.text, true);
              }
            }}
            className={`group bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs hover:border-indigo-300 hover:shadow-xs transition-all cursor-pointer flex flex-col justify-between ${
              isLoading ? 'opacity-60 cursor-not-allowed' : ''
            }`}
          >
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-1.5 text-xs font-semibold text-slate-600">
                  {getIconForCategory(ex.category)}
                  <span>{ex.category}</span>
                </div>
                <ArrowUpRight className="w-3.5 h-3.5 text-slate-400 group-hover:text-indigo-600 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
              </div>

              <h4 className="text-sm font-bold text-slate-900 group-hover:text-indigo-600 transition-colors">
                {ex.title}
              </h4>

              <p className="mt-2 text-xs text-slate-600 line-clamp-3 italic bg-slate-50/70 p-2 rounded-lg border border-slate-100 font-sans">
                "{ex.text}"
              </p>

              <p className="mt-2 text-[11px] text-slate-500 leading-normal">
                {ex.purpose}
              </p>
            </div>

            <div className="mt-3 pt-2.5 border-t border-slate-100 flex flex-wrap gap-1">
              {ex.targetTags.map((tag) => (
                <span
                  key={tag}
                  className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};
