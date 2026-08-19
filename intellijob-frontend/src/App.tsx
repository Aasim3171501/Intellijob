import { useState, useCallback } from 'react';
import { FileText, Search, RefreshCw } from 'lucide-react';
import { api } from '@/lib/api';
import type { AnalyzeResponse } from '@/types/api';
import {
  Dropzone,
  SkillPills,
  MatchCard,
  RoadmapView,
  LoadingSpinner,
  CareerPathways,
} from '@/components';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [targetTitle, setTargetTitle] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | undefined>(undefined);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [selectedPathwayKey, setSelectedPathwayKey] = useState<string | null>(null);

  const handleFileSelect = useCallback((file: File) => {
    if (file.name === 'removed' || file.name === 'invalid') {
      setSelectedFile(null);
    } else {
      setSelectedFile(file);
    }
    setError(undefined);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedFile) {
      setError('Please select a PDF resume');
      return;
    }

    setIsLoading(true);
    setError(undefined);

    try {
      const response = await api.analyzeResume({
        file: selectedFile,
        target_title: targetTitle.trim(),
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze resume');
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setTargetTitle('');
    setResult(null);
    setError(undefined);
  };

  const isDiscovery =
    result?.career_mode === 'discovery' && (result.career_pathways?.length ?? 0) > 0;
  // If the selected key no longer exists in a fresh result, fall back to the
  // #1 pathway instead of showing a stale selection.
  const activePathway =
    isDiscovery && result
      ? result.career_pathways.find((p) => p.key === selectedPathwayKey) ??
        result.career_pathways[0]
      : null;
  const viewMatches = activePathway ? activePathway.matches : (result?.matches ?? []);
  const viewRoadmap = activePathway ? activePathway.roadmap : result?.roadmap;

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-primary-600 flex items-center justify-center">
                <FileText className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">IntelliJob</h1>
                <p className="text-xs text-slate-500">Local Career Analytics Platform</p>
              </div>
            </div>
            <div className="hidden md:flex items-center gap-4 text-sm text-slate-600">
              <span className="px-2 py-1 rounded bg-slate-100 font-mono text-slate-700">
                100% Local
              </span>
              <span className="px-2 py-1 rounded bg-slate-100 font-mono text-slate-700">
                GDPR Compliant
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!result ? (
          <div className="max-w-2xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-slate-900 mb-3">
                Analyze Your Resume
              </h2>
              <p className="text-slate-600">
                Upload your PDF resume and target a specific role to get a personalized
                skill-gap roadmap powered by local AI.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
                <Dropzone
                  onFileSelect={handleFileSelect}
                  selectedFile={selectedFile}
                  isLoading={isLoading}
                  error={error}
                />
              </div>

              <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
                <label htmlFor="target-title" className="block text-sm font-medium text-slate-700 mb-2">
                  Target Job Title
                </label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                  <input
                    id="target-title"
                    type="text"
                    value={targetTitle}
                    onChange={(e) => setTargetTitle(e.target.value)}
                    placeholder="e.g., Senior Backend Engineer, Data Scientist, DevOps Engineer"
                    className="w-full pl-10 pr-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-colors"
                    disabled={isLoading}
                    aria-describedby="target-title-help"
                  />
                </div>
                <p id="target-title-help" className="mt-1 text-sm text-slate-500">
                  Optional — leave blank for <span className="font-medium text-primary-700">Career Discovery</span> and
                  we'll recommend the best-fit pathways for your resume automatically.
                </p>
              </div>

              {error && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm" role="alert">
                  {error}
                </div>
              )}

              <div className="flex gap-4">
                <button
                  type="submit"
                  disabled={isLoading || !selectedFile}
                  className="flex-1 py-3 px-6 bg-primary-600 text-white font-semibold rounded-lg hover:bg-primary-700 focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isLoading ? (
                    <>
                      <LoadingSpinner size="sm" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Search className="w-5 h-5" />
                      Analyze Resume
                    </>
                  )}
                </button>
                <button
                  type="button"
                  onClick={handleReset}
                  disabled={isLoading}
                  className="px-6 py-3 border border-slate-300 text-slate-700 font-semibold rounded-lg hover:bg-slate-50 focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 transition-colors disabled:opacity-50"
                >
                  Reset
                </button>
              </div>
            </form>
          </div>
        ) : (
          <div className="space-y-8 animate-fade-in">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div>
                <h2 className="text-2xl font-bold text-slate-900">Analysis Results</h2>
                {activePathway ? (
                  <p className="text-slate-600 mt-1">
                    <span className="font-medium text-primary-700">Career Discovery</span> — exploring{' '}
                    <span className="font-medium text-slate-900">{activePathway.name}</span>
                  </p>
                ) : (
                  <p className="text-slate-600 mt-1">
                    Target: <span className="font-medium text-primary-700">{targetTitle}</span>
                  </p>
                )}
              </div>
              <button
                onClick={handleReset}
                className="flex items-center gap-2 px-4 py-2 border border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                New Analysis
              </button>
            </div>

            {isDiscovery && result && (
              <CareerPathways
                pathways={result.career_pathways}
                activeKey={activePathway?.key ?? selectedPathwayKey}
                onSelect={setSelectedPathwayKey}
              />
            )}

            <div className="grid lg:grid-cols-3 gap-6">
              <div className="lg:col-span-1 space-y-6">
                <SkillPills skills={result.extracted_skills} />

                <div className="bg-white rounded-xl border border-slate-200 p-6">
                  <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-primary-500"></span>
                    Match Summary
                  </h3>
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600">Jobs Analyzed</span>
                      <span className="font-semibold text-slate-900">{viewMatches.length}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600">Top Match Score</span>
                      <span className="font-semibold text-primary-700">
                        {viewMatches[0]
                          ? `${Math.round(viewMatches[0].similarity_score * 100)}%`
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600">Skills Extracted</span>
                      <span className="font-semibold text-slate-900">{result.extracted_skills.length}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="lg:col-span-2 space-y-6">
                <section>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-primary-500"></span>
                      {activePathway ? `Matches in ${activePathway.name}` : 'Top Job Matches'}
                    </h3>
                    <span className="text-sm text-slate-500">
                      {viewMatches.length} matches found
                    </span>
                  </div>
                  <div className="space-y-4">
                    {viewMatches.map((match, index) => (
                      <MatchCard
                        key={match.id}
                        match={match}
                        index={index}
                      />
                    ))}
                    {viewMatches.length === 0 && (
                      <div className="text-center py-12 text-slate-500">
                        <p>No matching jobs found for the given criteria.</p>
                      </div>
                    )}
                  </div>
                </section>

                {viewRoadmap && (
                  <section>
                    <RoadmapView roadmap={viewRoadmap} />
                  </section>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="bg-white border-t border-slate-200 mt-12">
        <div className="max-w-6xl mx-auto px-4 py-6 text-center text-sm text-slate-500">
          <p>
            IntelliJob — Local Career Analytics Platform | Built for MSc IT+ Dissertation
          </p>
          <p className="mt-1">
            Powered by PyMuPDF, spaCy, SentenceTransformers, and local Ollama LLMs
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;