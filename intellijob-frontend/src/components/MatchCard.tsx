import { Building2, MapPin, Star, ExternalLink } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { JobMatch } from '@/types/api';

interface MatchCardProps {
  match: JobMatch;
  index: number;
}

const getScoreColor = (score: number): string => {
  if (score >= 0.8) return 'text-green-600 bg-green-100';
  if (score >= 0.6) return 'text-blue-600 bg-blue-100';
  if (score >= 0.4) return 'text-yellow-600 bg-yellow-100';
  return 'text-slate-600 bg-slate-100';
};

const getScoreLabel = (score: number): string => {
  if (score >= 0.8) return 'Excellent Match';
  if (score >= 0.6) return 'Good Match';
  if (score >= 0.4) return 'Fair Match';
  return 'Weak Match';
};

export function MatchCard({ match, index }: MatchCardProps) {
  const scorePercent = Math.round(match.similarity_score * 100);
  const scoreColor = getScoreColor(match.similarity_score);
  const scoreLabel = getScoreLabel(match.similarity_score);

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-medium text-slate-500">#{index + 1}</span>
            <h4 className="text-lg font-semibold text-slate-900 truncate">
              {match.title}
            </h4>
          </div>
          {match.company && (
            <p className="flex items-center gap-1.5 text-sm text-slate-600">
              <Building2 className="w-4 h-4" />
              {match.company}
            </p>
          )}
          {match.location_display && (
            <p className="flex items-center gap-1.5 text-sm text-slate-500 mt-1">
              <MapPin className="w-4 h-4" />
              {match.location_display}
            </p>
          )}
        </div>
        <div className="flex flex-col items-end gap-2 shrink-0">
          <div className={cn('match-score', scoreColor)}>
            {scorePercent}%
          </div>
          <span className="text-xs font-medium px-2 py-1 rounded-full bg-slate-100 text-slate-600">
            {scoreLabel}
          </span>
        </div>
      </div>

      {match.description_excerpt && (
        <div className="pt-4 border-t border-slate-100">
          <p className="text-sm text-slate-600 line-clamp-3">
            {match.description_excerpt}
          </p>
        </div>
      )}

      <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between">
        <Star className="w-4 h-4 text-primary-500" />
        <a
          href="#"
          className="text-sm font-medium text-primary-600 hover:text-primary-700 flex items-center gap-1"
          onClick={(e) => e.preventDefault()}
        >
          View Similar Skills
          <ExternalLink className="w-3.5 h-3.5" />
        </a>
      </div>
    </div>
  );
}