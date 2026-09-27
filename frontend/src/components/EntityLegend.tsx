import React from 'react';
import { ENTITY_STYLES } from '../types/ner';

export const EntityLegend: React.FC = () => {
  const styles = Object.values(ENTITY_STYLES);

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs p-5 transition-all">
      <div className="flex items-center justify-between mb-3 border-b border-slate-100 pb-2">
        <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
          Entity Ontology & Color Taxonomy (8 Classes)
        </h3>
        <span className="text-[11px] text-slate-400 font-mono">Standardized BIO-IOB2</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        {styles.map((style) => (
          <div
            key={style.label}
            className={`flex items-center space-x-2 px-3 py-2 rounded-lg border text-xs font-medium ${style.badgeBg} ${style.badgeBorder} ${style.badgeText} transition-all`}
          >
            <span className={`w-2.5 h-2.5 rounded-full ${style.dotColor} shrink-0`} />
            <div className="truncate">
              <span className="font-bold tracking-tight">{style.displayName}</span>
              <span className="block text-[10px] opacity-75 font-mono">[{style.label}]</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
