import React, { useMemo, useState } from 'react';
import { Entity, ENTITY_STYLES } from '../types/ner';
import { ArrowUpDown, Check, Copy, Search, Table as TableIcon } from 'lucide-react';

interface EntityTableProps {
  entities: Entity[];
  selectedEntityIndex?: number | null;
  onSelectEntity?: (index: number | null) => void;
  activeFilter?: string | null;
  onFilterChange?: (filter: string | null) => void;
}

export const EntityTable: React.FC<EntityTableProps> = ({
  entities,
  selectedEntityIndex,
  onSelectEntity,
  activeFilter,
  onFilterChange,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [sortField, setSortField] = useState<'offset' | 'confidence' | 'text' | 'label'>('offset');
  const [sortAsc, setSortAsc] = useState(true);
  const [copied, setCopied] = useState(false);

  if (!entities || entities.length === 0) return null;

  // Filter and sort items
  const processedEntities = useMemo(() => {
    let items = entities.map((e, idx) => ({ ...e, originalIndex: idx }));

    // Category filter
    if (activeFilter) {
      items = items.filter((e) => e.label === activeFilter);
    }

    // Search query filter
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      items = items.filter(
        (e) =>
          e.text.toLowerCase().includes(q) ||
          e.label.toLowerCase().includes(q) ||
          e.source_model.toLowerCase().includes(q)
      );
    }

    // Sort
    items.sort((a, b) => {
      let comparison = 0;
      if (sortField === 'offset') {
        comparison = a.start - b.start || a.end - b.end;
      } else if (sortField === 'confidence') {
        comparison = a.confidence - b.confidence;
      } else if (sortField === 'text') {
        comparison = a.text.localeCompare(b.text);
      } else if (sortField === 'label') {
        comparison = a.label.localeCompare(b.label);
      }
      return sortAsc ? comparison : -comparison;
    });

    return items;
  }, [entities, activeFilter, searchQuery, sortField, sortAsc]);

  const handleSort = (field: 'offset' | 'confidence' | 'text' | 'label') => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(true);
    }
  };

  const handleCopyJSON = () => {
    navigator.clipboard.writeText(JSON.stringify(entities, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-white rounded-3xl border border-slate-200/90 shadow-sm overflow-hidden transition-all">
      {/* Table Toolbar */}
      <div className="p-5 sm:p-6 border-b border-slate-200/80 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center space-x-2">
          <TableIcon className="w-4 h-4 text-indigo-600" />
          <h3 className="text-base font-extrabold text-slate-900 tracking-tight">
            Extracted Entities Table
          </h3>
          <span className="text-xs font-mono font-medium px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
            {processedEntities.length} of {entities.length}
          </span>
        </div>

        {/* Search, Filter reset, and Copy JSON */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search extracted entities..."
              className="pl-8 pr-3 py-1.5 text-xs rounded-xl border border-slate-200 focus:outline-hidden focus:ring-1 focus:ring-indigo-500 w-48 sm:w-56"
            />
          </div>

          {activeFilter && (
            <button
              onClick={() => onFilterChange?.(null)}
              className="inline-flex items-center space-x-1 text-xs px-2.5 py-1.5 rounded-xl bg-indigo-50 text-indigo-700 border border-indigo-100 font-medium hover:bg-indigo-100 transition-colors"
            >
              <span>Filter: {activeFilter}</span>
              <span>×</span>
            </button>
          )}

          <button
            type="button"
            onClick={handleCopyJSON}
            className="inline-flex items-center space-x-1.5 text-xs font-semibold px-3 py-1.5 rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-50 transition-colors shadow-2xs"
            title="Copy raw JSON predictions"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-600" />
                <span className="text-emerald-700">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-slate-400" />
                <span>Copy JSON</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-slate-50/80 border-b border-slate-200/80 text-[11px] font-bold text-slate-500 uppercase tracking-wider select-none">
              <th scope="col" className="py-3 px-4 sm:px-6">#</th>
              <th
                scope="col"
                onClick={() => handleSort('text')}
                className="py-3 px-4 sm:px-6 cursor-pointer hover:text-slate-800"
              >
                <div className="flex items-center space-x-1">
                  <span>Entity</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-400" />
                </div>
              </th>
              <th
                scope="col"
                onClick={() => handleSort('label')}
                className="py-3 px-4 sm:px-6 cursor-pointer hover:text-slate-800"
              >
                <div className="flex items-center space-x-1">
                  <span>Type</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-400" />
                </div>
              </th>
              <th
                scope="col"
                onClick={() => handleSort('confidence')}
                className="py-3 px-4 sm:px-6 cursor-pointer hover:text-slate-800"
              >
                <div className="flex items-center space-x-1">
                  <span>Confidence</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-400" />
                </div>
              </th>
              <th scope="col" className="py-3 px-4 sm:px-6">Source Model</th>
              <th
                scope="col"
                onClick={() => handleSort('offset')}
                className="py-3 px-4 sm:px-6 cursor-pointer hover:text-slate-800"
              >
                <div className="flex items-center space-x-1">
                  <span>Span</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-400" />
                </div>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-sm">
            {processedEntities.length > 0 ? (
              processedEntities.map((item, rowIdx) => {
                const style = ENTITY_STYLES[item.label] || ENTITY_STYLES.CHEMICAL;
                const isSelected = selectedEntityIndex === item.originalIndex;

                return (
                  <tr
                    key={rowIdx}
                    onClick={() =>
                      onSelectEntity?.(isSelected ? null : item.originalIndex)
                    }
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-indigo-50/80 hover:bg-indigo-100/70 font-medium'
                        : 'hover:bg-slate-50/70'
                    }`}
                  >
                    <td className="py-3 px-4 sm:px-6 font-mono text-xs text-slate-400">
                      {rowIdx + 1}
                    </td>

                    <td className="py-3 px-4 sm:px-6 font-semibold text-slate-900">
                      <span className="inline-block px-2 py-0.5 rounded-lg bg-slate-100/90 text-slate-900 border border-slate-200/80">
                        {item.text}
                      </span>
                    </td>

                    <td className="py-3 px-4 sm:px-6">
                      <span
                        className={`inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-md text-xs font-bold uppercase tracking-wider border ${style.badgeBg} ${style.badgeBorder} ${style.badgeText}`}
                      >
                        <span className={`w-1.5 h-1.5 rounded-full ${style.dotColor}`} />
                        <span>{style.displayName}</span>
                      </span>
                    </td>

                    <td className="py-3 px-4 sm:px-6">
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-xs font-bold text-slate-800">
                          {(item.confidence * 100).toFixed(2)}%
                        </span>
                        <div className="w-16 bg-slate-100 rounded-full h-1.5 overflow-hidden hidden sm:block">
                          <div
                            className="bg-indigo-600 h-1.5 rounded-full"
                            style={{ width: `${Math.min(100, Math.max(0, item.confidence * 100))}%` }}
                          />
                        </div>
                      </div>
                    </td>

                    <td className="py-3 px-4 sm:px-6">
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-slate-100 text-slate-700 border border-slate-200">
                        {item.source_model}
                      </span>
                    </td>

                    <td className="py-3 px-4 sm:px-6 font-mono text-xs text-slate-500">
                      {item.start}–{item.end}
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={6} className="py-8 text-center text-xs text-slate-400 font-mono">
                  No entities matching your query or filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
