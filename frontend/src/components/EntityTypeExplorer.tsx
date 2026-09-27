import React from 'react';
import { ENTITY_STYLES, EntityLabel } from '../types/ner';
import { Beaker, Dna, HeartPulse, Layers, Microscope, Pill, Sparkles } from 'lucide-react';

interface EntityTypeExplorerProps {
  activeFilter?: string | null;
  onSelectFilter?: (filter: string | null) => void;
}

export const EntityTypeExplorer: React.FC<EntityTypeExplorerProps> = ({
  activeFilter,
  onSelectFilter,
}) => {
  const styles = Object.values(ENTITY_STYLES);

  const getIcon = (label: EntityLabel) => {
    switch (label) {
      case 'CHEMICAL':
        return <Pill className="w-4 h-4 text-teal-600" />;
      case 'DISEASE':
        return <HeartPulse className="w-4 h-4 text-rose-600" />;
      case 'ANATOMY':
        return <Beaker className="w-4 h-4 text-amber-600" />;
      case 'PROTEIN':
        return <Microscope className="w-4 h-4 text-indigo-600" />;
      case 'DNA':
      case 'RNA':
        return <Dna className="w-4 h-4 text-purple-600" />;
      case 'CELL_TYPE':
      case 'CELL_LINE':
        return <Sparkles className="w-4 h-4 text-emerald-600" />;
      default:
        return <Layers className="w-4 h-4 text-slate-600" />;
    }
  };

  return (
    <section id="ontology" className="py-10 border-b border-slate-200/80">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between mb-6 gap-2">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700 text-xs font-semibold uppercase tracking-wider font-mono">
            <Layers className="w-3.5 h-3.5 text-indigo-600" />
            <span>Biomedical Knowledge Ontology</span>
          </div>
          <h3 className="mt-2 text-2xl font-extrabold text-slate-900 tracking-tight">
            8 Biomedical Entity Types
          </h3>
          <p className="mt-1 text-sm text-slate-500">
            Standardized BIO-IOB2 taxonomy recognized across pharmacology, pathology, anatomy, and genetics.
          </p>
        </div>
        <span className="text-xs font-mono text-slate-400">Click card to filter results</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {styles.map((style) => {
          const isSelected = activeFilter === style.label;

          return (
            <div
              key={style.label}
              onClick={() => onSelectFilter?.(isSelected ? null : style.label)}
              className={`p-4 rounded-2xl border transition-all cursor-pointer flex flex-col justify-between ${
                isSelected
                  ? 'bg-indigo-50/90 border-indigo-500 shadow-sm ring-2 ring-indigo-500/20'
                  : 'bg-white border-slate-200/80 hover:border-indigo-300 hover:shadow-xs shadow-2xs'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <div className="p-1.5 rounded-lg bg-slate-50 border border-slate-100">
                      {getIcon(style.label)}
                    </div>
                    <div>
                      <h4 className="text-sm font-bold text-slate-900 tracking-tight">
                        {style.displayName}
                      </h4>
                      <span className="text-[10px] font-mono text-slate-400">
                        [{style.label}]
                      </span>
                    </div>
                  </div>
                  <span className={`w-2.5 h-2.5 rounded-full ${style.dotColor}`} />
                </div>

                <p className="text-xs text-slate-600 leading-relaxed">
                  {style.description}
                </p>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px]">
                <span className="text-slate-400 font-medium">Source:</span>
                <span className="font-mono font-semibold text-slate-700 bg-slate-50 px-2 py-0.5 rounded border border-slate-200">
                  {style.sourceModel}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
};
