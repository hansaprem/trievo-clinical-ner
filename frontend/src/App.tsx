import React, { useEffect, useState } from 'react';
import { Header } from './components/Header';
import { Hero } from './components/Hero';
import { PipelineFlow } from './components/PipelineFlow';
import { ClinicalInput } from './components/ClinicalInput';
import { ExampleLibrary, EXAMPLES } from './components/ExampleLibrary';
import { EntityHighlightedText } from './components/EntityHighlightedText';
import { StatisticsCards } from './components/StatisticsCards';
import { EntityTable } from './components/EntityTable';
import { EntityTypeExplorer } from './components/EntityTypeExplorer';
import { ModelArchitecture } from './components/ModelArchitecture';
import { BenchmarkSection } from './components/BenchmarkSection';
import { TrustStatus } from './components/TrustStatus';
import { Footer } from './components/Footer';
import { checkHealth, predictClinicalText, ApiError } from './services/api';
import { Entity, HealthResponse, PredictResponse } from './types/ner';
import { AlertCircle, FileSearch, X } from 'lucide-react';

export const App: React.FC = () => {
  // Clinical Text & Result States
  const [inputText, setInputText] = useState<string>('');
  const [analyzedText, setAnalyzedText] = useState<string>('');
  const [entities, setEntities] = useState<Entity[] | null>(null);
  const [selectedEntityIndex, setSelectedEntityIndex] = useState<number | null>(null);
  const [activeFilter, setActiveFilter] = useState<string | null>(null);

  // Network & System States
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);

  // Live Health Fetching
  const fetchHealth = async () => {
    setHealthLoading(true);
    try {
      const data = await checkHealth();
      setHealth(data);
      if (errorMessage && errorMessage.includes('Unable to connect')) {
        setErrorMessage(null);
      }
    } catch {
      setHealth({
        status: 'unhealthy',
        models_loaded: false,
        models: [],
      });
    } finally {
      setHealthLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  // Inference Execution
  const executeInference = async (rawText: string) => {
    const trimmed = rawText.trim();
    if (!trimmed) {
      setErrorMessage('Input text cannot be empty or whitespace-only.');
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    setSelectedEntityIndex(null);

    try {
      const response: PredictResponse = await predictClinicalText(trimmed);
      setAnalyzedText(response.text);
      setEntities(response.entities);

      // Background health check refresh if previously marked degraded
      if (!health?.models_loaded) {
        fetchHealth();
      }

      // Smooth scroll down to results section on mobile/desktop
      setTimeout(() => {
        const resultsEl = document.getElementById('results-section');
        if (resultsEl) {
          resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('An unexpected inference error occurred.');
      }
      setEntities(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnalyze = () => {
    executeInference(inputText);
  };

  // Example Selection Callback
  const handleSelectExample = (text: string, autoAnalyze = false) => {
    setInputText(text);
    if (autoAnalyze) {
      executeInference(text);
    }
  };

  const handleLoadDefaultExample = () => {
    const defaultEx = EXAMPLES[0].text;
    setInputText(defaultEx);
  };

  // Reset State
  const handleClear = () => {
    setInputText('');
    setAnalyzedText('');
    setEntities(null);
    setSelectedEntityIndex(null);
    setActiveFilter(null);
    setErrorMessage(null);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50/70 text-slate-900 font-sans selection:bg-indigo-100 selection:text-indigo-900">
      {/* Sticky Header with Dynamic Status */}
      <Header
        health={health}
        healthLoading={healthLoading}
        onRefreshHealth={fetchHealth}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 space-y-12 py-6">
        {/* Hero Section with 3D Core */}
        <Hero />

        {/* 5-Stage Multi-Model Architecture Flow */}
        <PipelineFlow />

        {/* Global Error Banner */}
        {errorMessage && (
          <div className="p-4 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 flex items-start justify-between shadow-xs animate-in fade-in duration-200">
            <div className="flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
              <div>
                <h4 className="text-sm font-bold text-rose-900">Inference Error</h4>
                <p className="text-xs text-rose-700 mt-0.5 leading-relaxed">{errorMessage}</p>
              </div>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-rose-400 hover:text-rose-600 p-1 transition-colors"
              aria-label="Dismiss error"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Clinical Text Editor Console */}
        <ClinicalInput
          text={inputText}
          onTextChange={setInputText}
          onAnalyze={handleAnalyze}
          onClear={handleClear}
          onLoadDefaultExample={handleLoadDefaultExample}
          isLoading={isLoading}
        />

        {/* 8-Category Example Library */}
        <ExampleLibrary
          onSelectExample={handleSelectExample}
          isLoading={isLoading}
        />

        {/* Live Results Display Area */}
        <div id="results-section">
          {entities !== null ? (
            <div className="space-y-6 animate-in fade-in duration-300">
              {/* Top Factual Counts */}
              <StatisticsCards entities={entities} />

              {/* Exact Character Highlight Annotation */}
              <EntityHighlightedText
                text={analyzedText}
                entities={entities}
                selectedEntityIndex={selectedEntityIndex}
                onSelectEntity={setSelectedEntityIndex}
              />

              {/* Searchable, Filterable Entity Table */}
              <EntityTable
                entities={entities}
                selectedEntityIndex={selectedEntityIndex}
                onSelectEntity={setSelectedEntityIndex}
                activeFilter={activeFilter}
                onFilterChange={setActiveFilter}
              />
            </div>
          ) : (
            /* Clean Empty State */
            <div className="bg-white rounded-3xl border border-dashed border-slate-300 p-10 text-center shadow-2xs">
              <div className="w-12 h-12 rounded-2xl bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-500 mx-auto mb-3">
                <FileSearch className="w-6 h-6" />
              </div>
              <h3 className="text-base font-bold text-slate-800">
                Inference Console Ready
              </h3>
              <p className="text-xs text-slate-500 max-w-md mx-auto mt-1 leading-relaxed">
                Enter clinical text in the editor above or click any card in the <strong>Example Library</strong> to
                stream real extractions from the four fine-tuned PubMedBERT checkpoints.
              </p>
            </div>
          )}
        </div>

        {/* 8 Biomedical Entity Types Explorer */}
        <EntityTypeExplorer
          activeFilter={activeFilter}
          onSelectFilter={setActiveFilter}
        />

        {/* Multi-Model Ensemble Architecture Details */}
        <ModelArchitecture />

        {/* Verified 900-Sentence Benchmark & Error Analysis */}
        <BenchmarkSection />

        {/* System Status & Verified Technical Capabilities */}
        <TrustStatus />
      </main>

      {/* Academic Disclaimer Footer */}
      <Footer />
    </div>
  );
};
