import React, { useEffect, useState } from 'react';
import { Binary, Loader2, Play, RotateCcw, Sparkles } from 'lucide-react';

interface ClinicalInputProps {
  text: string;
  onTextChange: (newText: string) => void;
  onAnalyze: () => void;
  onClear: () => void;
  onLoadDefaultExample: () => void;
  isLoading: boolean;
  disabled?: boolean;
}

export const ClinicalInput: React.FC<ClinicalInputProps> = ({
  text,
  onTextChange,
  onAnalyze,
  onClear,
  onLoadDefaultExample,
  isLoading,
  disabled = false,
}) => {
  // Real-time lifecycle stage indicator during loading
  const [loadingStage, setLoadingStage] = useState<number>(0);

  const stages = [
    'Scanning clinical text...',
    'Running 4-model PubMedBERT inference...',
    'Resolving entity boundaries & provenance...',
    'Finalizing structured extractions...',
  ];

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isLoading) {
      setLoadingStage(0);
      interval = setInterval(() => {
        setLoadingStage((prev) => (prev < stages.length - 1 ? prev + 1 : prev));
      }, 550);
    } else {
      setLoadingStage(0);
    }
    return () => clearInterval(interval);
  }, [isLoading]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (text.trim() && !isLoading && !disabled) {
        onAnalyze();
      }
    }
  };

  return (
    <div id="analyze" className="bg-white rounded-3xl border border-slate-200/90 shadow-sm p-6 sm:p-8 transition-all">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="inline-flex items-center space-x-1.5 text-xs font-mono font-bold text-indigo-600 uppercase tracking-wider">
            <Binary className="w-3.5 h-3.5" />
            <span>Interactive Inference Console</span>
          </div>
          <h3 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight mt-0.5">
            Analyze Clinical Text
          </h3>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Enter unstructured biomedical text and inspect the entities detected by the multi-model pipeline.
          </p>
        </div>

        <div className="flex items-center space-x-2 self-start sm:self-auto">
          <button
            type="button"
            onClick={onLoadDefaultExample}
            disabled={isLoading || disabled}
            className="inline-flex items-center space-x-1.5 text-xs font-semibold px-3 py-2 rounded-xl bg-indigo-50 text-indigo-700 hover:bg-indigo-100 disabled:opacity-50 transition-colors border border-indigo-100"
          >
            <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
            <span>Load Quick Demo</span>
          </button>

          {text && (
            <button
              type="button"
              onClick={onClear}
              disabled={isLoading || disabled}
              className="inline-flex items-center space-x-1 text-xs font-medium px-3 py-2 rounded-xl text-slate-600 hover:bg-slate-100 disabled:opacity-50 transition-colors border border-slate-200"
              title="Clear input and results"
            >
              <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
              <span>Clear</span>
            </button>
          )}
        </div>
      </div>

      {/* Editor Textarea */}
      <div className="relative">
        <textarea
          id="clinical-text-input"
          value={text}
          onChange={(e) => onTextChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading || disabled}
          placeholder="Enter clinical notes, pathology findings, patient discharge summaries, or biomedical literature abstracts..."
          rows={5}
          className="w-full rounded-2xl border border-slate-200/90 p-4 sm:p-5 text-sm sm:text-base text-slate-800 placeholder-slate-400 focus:outline-hidden focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 disabled:bg-slate-50 disabled:text-slate-500 transition-all font-sans leading-relaxed resize-y shadow-2xs"
        />

        {/* Floating shortcut badge on desktop */}
        <div className="absolute bottom-3 right-3 hidden sm:flex items-center space-x-1 text-[10px] font-mono text-slate-400 bg-white/90 backdrop-blur-xs px-2 py-0.5 rounded border border-slate-200 pointer-events-none">
          <span>Press</span>
          <kbd className="font-semibold text-slate-600">Ctrl + Enter</kbd>
          <span>to run</span>
        </div>
      </div>

      {/* Bottom Bar: Character metrics, Lifecycle Status & Submit CTA */}
      <div className="mt-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pt-2">
        <div className="flex items-center space-x-3 text-xs font-mono text-slate-400">
          <span>{text.length} characters</span>
          <span>•</span>
          <span>{text.trim() ? text.trim().split(/\s+/).length : 0} words</span>
        </div>

        {/* Live Lifecycle Stage Indicator */}
        {isLoading && (
          <div className="flex items-center space-x-2 text-xs font-mono text-indigo-700 bg-indigo-50 px-3 py-1.5 rounded-lg border border-indigo-100 animate-pulse">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-600" />
            <span>{stages[loadingStage]}</span>
          </div>
        )}

        <button
          type="button"
          onClick={onAnalyze}
          disabled={!text.trim() || isLoading || disabled}
          className="w-full sm:w-auto inline-flex items-center justify-center space-x-2 px-7 py-3 rounded-xl bg-indigo-600 text-white font-semibold text-sm hover:bg-indigo-700 active:bg-indigo-800 focus:outline-hidden focus:ring-2 focus:ring-indigo-500/40 disabled:bg-slate-200 disabled:text-slate-400 disabled:cursor-not-allowed shadow-xs transition-all"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-white" />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              <span>Analyze Clinical Text</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};
