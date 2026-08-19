import { Compass, TrendingUp } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { CareerPathway } from '@/types/api';

interface CareerPathwaysProps {
  pathways: CareerPathway[];
  activeKey: string | null;
  onSelect: (key: string) => void;
}

const MEDALS = ['bg-primary-600 text-white', 'bg-slate-500 text-white', 'bg-slate-400 text-white'];

export function CareerPathways({ pathways, activeKey, onSelect }: CareerPathwaysProps) {
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
            Your resume was matched against every pathway in the dataset — select one to explore.
          </p>
        </div>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 mt-4">
        {pathways.map((pathway, index) => {
          const active = pathway.key === activeKey;
          const scorePercent = Math.round(pathway.similarity_score * 100);
          return (
            <button
              key={pathway.key}
              type="button"
              onClick={() => onSelect(pathway.key)}
              aria-pressed={active}
              className={cn(
                'text-left rounded-xl border p-4 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500',
                active
                  ? 'border-primary-600 ring-2 ring-primary-500 bg-primary-50'
                  : 'border-slate-200 hover:border-primary-300 bg-white hover:bg-slate-50'
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
            </button>
          );
        })}
      </div>
    </section>
  );
}