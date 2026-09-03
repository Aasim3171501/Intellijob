import { Compass, TrendingUp, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { CareerPathway } from '@/types/api';

interface CareerPathwaysProps {
  pathways: CareerPathway[];
  activeKey: string | null;
  onSelect: (key: string) => void;
  loadingKeys?: Set<string>;
  isAnyLoading?: boolean;
}

const MEDALS = ['bg-primary-600 text-white', 'bg-slate-500 text-white', 'bg-slate-400 text-white'];

export function CareerPathways({ pathways, activeKey, onSelect, loadingKeys, isAnyLoading }: CareerPathwaysProps) {
  if (!pathways.length) return null;

  return (
    <section className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
      <div className="flex items-center gap-3 mb-1">
        <div className="w-9 h-9 rounded-lg bg-primary-100 flex items-center justify-center">
          <Compass className="w-5 h-5 text-primary-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-slate-900">Recommended Career Pathways</h3>
          <p className="text-sm text-slate-500">
            Ranked against the skills on your resume — select one to explore.
          </p>
        </div>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 mt-4">
        {pathways.map((pathway, index) => {
          const active = pathway.key === activeKey;
          const isLoading = loadingKeys?.has(pathway.key);
          // Disable all other cards when any pathway is loading
          const isDisabled = isLoading || (isAnyLoading && !active && !isLoading);
          const scorePercent = Math.round((pathway.score ?? pathway.similarity_score) * 100);
          const matched = pathway.matched_skills ?? [];
          return (
            <button
              key={pathway.key}
              type="button"
              onClick={() => onSelect(pathway.key)}
              aria-pressed={active}
              disabled={isDisabled}
              className={cn(
                'text-left rounded-xl border p-4 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary-500',
                active
                  ? 'border-primary-600 ring-2 ring-primary-500 bg-primary-50'
                  : 'border-slate-200 hover:border-primary-300 bg-white hover:bg-slate-50 hover:shadow-xl hover:scale-[1.02]',
                isDisabled && 'opacity-50 cursor-not-allowed',
                isLoading && 'opacity-60 cursor-wait'
              )}
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      'flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold',
                      MEDALS[index] ?? 'bg-slate-200 text-slate-700'
                    )}
                  >
                    {index + 1}
                  </span>
                  <span className="font-semibold text-slate-900">{pathway.name}</span>
                  {isLoading && (
                    <Loader2 className="w-4 h-4 text-primary-600 animate-spin" aria-label="Loading roadmap" />
                  )}
                  {isAnyLoading && !isLoading && !active && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-500">Loading...</span>
                  )}
                </div>
                <span
                  className={cn(
                    'flex items-center gap-1 text-sm font-bold',
                    active ? 'text-primary-700' : 'text-slate-600'
                  )}
                >
                  <TrendingUp className="w-4 h-4" />
                  {scorePercent}%
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-1 line-clamp-2">{pathway.description}</p>
              <p className="text-xs text-slate-400 mt-2">
                {pathway.match_count} roles analysed
              </p>
              {matched.length > 0 && (
                <div className="mt-2 border-t border-slate-100 pt-2">
                  <p className="text-[11px] font-semibold text-slate-500">
                    Overlaps {matched.length} of your skills
                  </p>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {matched.slice(0, 3).map((skill) => (
                      <span key={skill} className="text-[11px] px-1.5 py-0.5 rounded bg-primary-50 text-primary-700">
                        {skill}
                      </span>
                    ))}
                    {matched.length > 3 && (
                      <span className="text-[11px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-500">
                        +{matched.length - 3}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </button>
          );
        })}
      </div>
    </section>
  );
}