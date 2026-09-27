/**
 * Clinical NER Data Models & Type Definitions
 */

export interface Entity {
  text: string;
  label: string;
  start: number;
  end: number;
  source_model: string;
  confidence: number;
}

export interface PredictRequest {
  text: string;
}

export interface PredictResponse {
  text: string;
  entities: Entity[];
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy' | string;
  models_loaded: boolean;
  models: string[];
}

export interface RootResponse {
  name: string;
  status: string;
  version: string;
}

export type EntityLabel =
  | 'CHEMICAL'
  | 'DISEASE'
  | 'ANATOMY'
  | 'PROTEIN'
  | 'DNA'
  | 'RNA'
  | 'CELL_TYPE'
  | 'CELL_LINE';

export interface EntityStyle {
  label: EntityLabel;
  displayName: string;
  badgeBg: string;
  badgeText: string;
  badgeBorder: string;
  highlightBg: string;
  highlightText: string;
  highlightBorder: string;
  dotColor: string;
  domain: 'Pharmacology' | 'Pathology' | 'Anatomy' | 'Molecular / Genetics';
  description: string;
  sourceModel: string;
}

export const ENTITY_STYLES: Record<string, EntityStyle> = {
  CHEMICAL: {
    label: 'CHEMICAL',
    displayName: 'Chemical',
    badgeBg: 'bg-teal-50',
    badgeText: 'text-teal-800',
    badgeBorder: 'border-teal-200',
    highlightBg: 'bg-teal-100/70 hover:bg-teal-100',
    highlightText: 'text-teal-900',
    highlightBorder: 'border-teal-400',
    dotColor: 'bg-teal-500',
    domain: 'Pharmacology',
    description: 'Drugs, small therapeutic molecules, metabolites, and synthetic chemical agents.',
    sourceModel: 'BC5CDR',
  },
  DISEASE: {
    label: 'DISEASE',
    displayName: 'Disease',
    badgeBg: 'bg-rose-50',
    badgeText: 'text-rose-800',
    badgeBorder: 'border-rose-200',
    highlightBg: 'bg-rose-100/70 hover:bg-rose-100',
    highlightText: 'text-rose-900',
    highlightBorder: 'border-rose-400',
    dotColor: 'bg-rose-500',
    domain: 'Pathology',
    description: 'Pathological conditions, clinical disorders, syndromes, signs, and symptoms.',
    sourceModel: 'BC5CDR, NCBI Disease',
  },
  ANATOMY: {
    label: 'ANATOMY',
    displayName: 'Anatomy',
    badgeBg: 'bg-amber-50',
    badgeText: 'text-amber-800',
    badgeBorder: 'border-amber-200',
    highlightBg: 'bg-amber-100/70 hover:bg-amber-100',
    highlightText: 'text-amber-900',
    highlightBorder: 'border-amber-400',
    dotColor: 'bg-amber-500',
    domain: 'Anatomy',
    description: 'Anatomical structures, organs, biological tissues, and anatomical subdivisions.',
    sourceModel: 'AnatEM',
  },
  PROTEIN: {
    label: 'PROTEIN',
    displayName: 'Protein',
    badgeBg: 'bg-indigo-50',
    badgeText: 'text-indigo-800',
    badgeBorder: 'border-indigo-200',
    highlightBg: 'bg-indigo-100/70 hover:bg-indigo-100',
    highlightText: 'text-indigo-900',
    highlightBorder: 'border-indigo-400',
    dotColor: 'bg-indigo-500',
    domain: 'Molecular / Genetics',
    description: 'Polypeptides, functional proteins, enzymes, receptors, and protein complexes.',
    sourceModel: 'JNLPBA',
  },
  DNA: {
    label: 'DNA',
    displayName: 'DNA',
    badgeBg: 'bg-purple-50',
    badgeText: 'text-purple-800',
    badgeBorder: 'border-purple-200',
    highlightBg: 'bg-purple-100/70 hover:bg-purple-100',
    highlightText: 'text-purple-900',
    highlightBorder: 'border-purple-400',
    dotColor: 'bg-purple-500',
    domain: 'Molecular / Genetics',
    description: 'Genes, deoxyribonucleic acid sequences, promoters, and genetic loci.',
    sourceModel: 'JNLPBA',
  },
  RNA: {
    label: 'RNA',
    displayName: 'RNA',
    badgeBg: 'bg-fuchsia-50',
    badgeText: 'text-fuchsia-800',
    badgeBorder: 'border-fuchsia-200',
    highlightBg: 'bg-fuchsia-100/70 hover:bg-fuchsia-100',
    highlightText: 'text-fuchsia-900',
    highlightBorder: 'border-fuchsia-400',
    dotColor: 'bg-fuchsia-500',
    domain: 'Molecular / Genetics',
    description: 'Messenger RNA (mRNA), non-coding RNAs, and transfer RNAs.',
    sourceModel: 'JNLPBA',
  },
  CELL_TYPE: {
    label: 'CELL_TYPE',
    displayName: 'Cell Type',
    badgeBg: 'bg-emerald-50',
    badgeText: 'text-emerald-800',
    badgeBorder: 'border-emerald-200',
    highlightBg: 'bg-emerald-100/70 hover:bg-emerald-100',
    highlightText: 'text-emerald-900',
    highlightBorder: 'border-emerald-400',
    dotColor: 'bg-emerald-500',
    domain: 'Molecular / Genetics',
    description: 'Natural somatic cells, immune cells, leukocytes, and physiological cell types.',
    sourceModel: 'JNLPBA',
  },
  CELL_LINE: {
    label: 'CELL_LINE',
    displayName: 'Cell Line',
    badgeBg: 'bg-sky-50',
    badgeText: 'text-sky-800',
    badgeBorder: 'border-sky-200',
    highlightBg: 'bg-sky-100/70 hover:bg-sky-100',
    highlightText: 'text-sky-900',
    highlightBorder: 'border-sky-400',
    dotColor: 'bg-sky-500',
    domain: 'Molecular / Genetics',
    description: 'Immortalized cell lines, cultured laboratory lineages, and experimental clones.',
    sourceModel: 'JNLPBA',
  },
};

export interface EntityStatistics {
  total: number;
  diseases: number;
  chemicals: number;
  anatomy: number;
  molecular: number;
}

export interface ExampleScenario {
  id: string;
  category: string;
  title: string;
  text: string;
  targetTags: string[];
  purpose: string;
}

export interface BenchmarkMetrics {
  dataset: string;
  domain: string;
  testSentences: number;
  precision: number;
  recall: number;
  f1: number;
  entityTypes: string;
  checkpoint: string;
}

export interface ErrorAnalysisCategory {
  title: string;
  percentageRange: string;
  description: string;
  clinicalSignificance: string;
}

export interface ModelDetail {
  id: string;
  name: string;
  corpus: string;
  domain: string;
  entityTypes: string[];
  backbone: string;
  inferenceRole: string;
  description: string;
}
