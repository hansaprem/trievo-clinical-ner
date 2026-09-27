import React, { useState } from 'react';
import { Entity, ENTITY_STYLES } from '../types/ner';
import { Sparkles, X } from 'lucide-react';

interface EntityHighlightedTextProps {
  text: string;
  entities: Entity[];
  selectedEntityIndex?: number | null;
  onSelectEntity?: (index: number | null) => void;
}

interface TextSegment {
  start: number;
  end: number;
  text: string;
  coveringEntities: { entity: Entity; originalIndex: number }[];
  endingEntities: { entity: Entity; originalIndex: number }[];
}

export const EntityHighlightedText: React.FC<EntityHighlightedTextProps> = ({
  text,
  entities,
  selectedEntityIndex,
  onSelectEntity,
}) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  if (!text) return null;

  if (!entities || entities.length === 0) {
    return (
      <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8 transition-all">
        <h3 className="text-base font-extrabold text-slate-900 mb-3 flex items-center space-x-2">
          <span>Annotated Clinical Text</span>
        </h3>
        <div className="p-5 rounded-2xl bg-slate-50 border border-slate-200/60 font-sans text-sm sm:text-base text-slate-600 leading-relaxed whitespace-pre-wrap">
          {text}
        </div>
        <p className="mt-3 text-xs text-slate-400 italic">
          No clinical or biomedical entities were detected with high confidence in this sample.
        </p>
      </div>
    );
  }

  // 1. Gather all unique offset boundary points
  const boundariesSet = new Set<number>([0, text.length]);
  entities.forEach((e) => {
    if (e.start >= 0 && e.start <= text.length) boundariesSet.add(e.start);
    if (e.end >= 0 && e.end <= text.length) boundariesSet.add(e.end);
  });
  const boundaries = Array.from(boundariesSet).sort((a, b) => a - b);

  // 2. Build non-overlapping intervals strictly preserving original characters
  const segments: TextSegment[] = [];
  for (let i = 0; i < boundaries.length - 1; i++) {
    const start = boundaries[i];
    const end = boundaries[i + 1];
    if (start >= end) continue;

    const segmentText = text.slice(start, end);
    const covering: { entity: Entity; originalIndex: number }[] = [];
    const ending: { entity: Entity; originalIndex: number }[] = [];

    entities.forEach((entity, index) => {
      if (entity.start <= start && end <= entity.end) {
        covering.push({ entity, originalIndex: index });
      }
      if (entity.end === end) {
        ending.push({ entity, originalIndex: index });
      }
    });

    segments.push({
      start,
      end,
      text: segmentText,
      coveringEntities: covering,
      endingEntities: ending,
    });
  }

  const selectedEntity =
    selectedEntityIndex !== null && selectedEntityIndex !== undefined
      ? entities[selectedEntityIndex]
      : null;

  return (
    <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8 transition-all">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-4 h-4 text-indigo-600" />
          <h3 className="text-base font-extrabold text-slate-900 tracking-tight">
            Annotated Clinical Text
          </h3>
          <span className="text-xs font-mono font-medium px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-100">
            {entities.length} {entities.length === 1 ? 'entity' : 'entities'}
          </span>
        </div>
        <span className="text-xs text-slate-400 font-mono">
          Hover for tooltip • Click span to open detail panel
        </span>
      </div>

      {/* Main Annotated Text Box */}
      <div className="p-6 rounded-2xl bg-slate-50/70 border border-slate-200/80 font-sans text-sm sm:text-base text-slate-800 leading-loose sm:leading-loose whitespace-pre-wrap select-text shadow-inner">
        {segments.map((seg, segIdx) => {
          if (seg.coveringEntities.length === 0) {
            return <span key={segIdx}>{seg.text}</span>;
          }

          const isMultiLayered = seg.coveringEntities.length > 1;
          const primaryEntityData = seg.coveringEntities[0];
          const style =
            ENTITY_STYLES[primaryEntityData.entity.label] || ENTITY_STYLES.CHEMICAL;

          const isSelected = seg.coveringEntities.some(
            (c) => c.originalIndex === selectedEntityIndex
          );
          const isHovered = seg.coveringEntities.some(
            (c) => c.originalIndex === hoveredIndex
          );

          return (
            <React.Fragment key={segIdx}>
              <mark
                onMouseEnter={() => setHoveredIndex(primaryEntityData.originalIndex)}
                onMouseLeave={() => setHoveredIndex(null)}
                onClick={() =>
                  onSelectEntity?.(
                    isSelected ? null : primaryEntityData.originalIndex
                  )
                }
                title={`${primaryEntityData.entity.label} | ${(
                  primaryEntityData.entity.confidence * 100
                ).toFixed(1)}% | Source: ${primaryEntityData.entity.source_model}`}
                className={`cursor-pointer rounded-sm px-1 py-0.5 transition-all border-b-2 font-medium ${
                  style.highlightBg
                } ${style.highlightText} ${style.highlightBorder} ${
                  isSelected
                    ? 'ring-2 ring-indigo-500 ring-offset-2 font-bold shadow-xs'
                    : isHovered
                    ? 'ring-1 ring-slate-400'
                    : ''
                } ${isMultiLayered ? 'outline-dashed outline-1 outline-indigo-500' : ''}`}
              >
                {seg.text}
              </mark>

              {/* Concluding Badges */}
              {seg.endingEntities.map(({ entity, originalIndex }) => {
                const badgeStyle =
                  ENTITY_STYLES[entity.label] || ENTITY_STYLES.CHEMICAL;
                const badgeSelected = selectedEntityIndex === originalIndex;

                return (
                  <span
                    key={`badge-${originalIndex}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectEntity?.(badgeSelected ? null : originalIndex);
                    }}
                    onMouseEnter={() => setHoveredIndex(originalIndex)}
                    onMouseLeave={() => setHoveredIndex(null)}
                    title={`Source: ${entity.source_model} | Confidence: ${(
                      entity.confidence * 100
                    ).toFixed(2)}%`}
                    className={`inline-flex items-center space-x-1 mx-1 px-1.5 py-0.5 rounded text-[10px] sm:text-[11px] font-mono font-bold uppercase tracking-wider cursor-pointer border align-baseline select-none transition-all ${
                      badgeStyle.badgeBg
                    } ${badgeStyle.badgeBorder} ${badgeStyle.badgeText} ${
                      badgeSelected ? 'ring-2 ring-indigo-500 ring-offset-1' : 'hover:opacity-90'
                    }`}
                  >
                    <span>{badgeStyle.displayName}</span>
                    <span className="opacity-75 font-normal">
                      {(entity.confidence * 100).toFixed(0)}%
                    </span>
                  </span>
                );
              })}
            </React.Fragment>
          );
        })}
      </div>

      {/* Selected Entity Detail Drawer / Card */}
      {selectedEntity && (
        <div className="mt-5 p-4 rounded-2xl bg-indigo-50/70 border border-indigo-100 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs animate-in fade-in duration-200">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-extrabold text-indigo-900 uppercase font-mono tracking-wider">
              Entity Inspector:
            </span>
            <span className="font-bold text-slate-900 bg-white px-2.5 py-1 rounded-lg border border-indigo-200 shadow-2xs text-sm">
              "{selectedEntity.text}"
            </span>
            <span className="font-mono text-slate-500 bg-white/70 px-2 py-0.5 rounded border border-indigo-100">
              span [{selectedEntity.start}:{selectedEntity.end}]
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="text-slate-600">
              Type:{' '}
              <span className="font-bold text-indigo-900 font-mono">
                {selectedEntity.label}
              </span>
            </div>
            <div className="text-slate-600">
              Confidence:{' '}
              <span className="font-bold text-indigo-900 font-mono">
                {(selectedEntity.confidence * 100).toFixed(2)}%
              </span>
            </div>
            <div className="text-slate-600">
              Model:{' '}
              <span className="font-bold text-indigo-900 font-mono">
                {selectedEntity.source_model}
              </span>
            </div>
            <button
              onClick={() => onSelectEntity?.(null)}
              className="p-1 rounded-lg hover:bg-indigo-100 text-indigo-600 transition-colors"
              title="Close Inspector"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
