import { AlertTriangle, TrendingUp, CheckCircle, Info, Flag, ChevronRight, PoundSterling } from 'lucide-react';
import { useState } from 'react';
import type { Roadmap, CareerPhase } from '@/types/api';
import { LearningSteps } from './LearningSteps';
import { PhaseDetailModal } from './PhaseDetailModal';

interface RoadmapViewProps {
  roadmap: Roadmap;
  skills?: string[];
  targetTitle?: string;
  matchedJobs?: string[];
  showTrajectoryOnly?: boolean;
}

export function RoadmapView({ roadmap, skills, targetTitle, matchedJobs, showTrajectoryOnly = false }: RoadmapViewProps) {
  const isFallback = roadmap.status === 'fallback';
  const [selectedPhase, setSelectedPhase] = useState<CareerPhase | null>(null);

  if (roadmap.status === 'pending') {
    return (
      <div className="text-center py-12">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary-100 mb-4">
          <TrendingUp className="w-8 h-8 text-primary-600 animate-spin" />
        </div>
        <h3 className="text-lg font-semibold text-slate-700 mb-2">Generating Roadmap...</h3>
        <p className="text-slate-500">
          Analyzing skill gaps and creating your personalized learning plan.
        </p>
      </div>
    );
  }

  if (
    !roadmap.learning_steps.length &&
    !(roadmap.career_trajectory?.length ?? 0)
  ) {
    return (
      <div className="text-center py-12">
        <Info className="mx-auto w-12 h-12 text-slate-400 mb-4" />
        <h3 className="text-lg font-semibold text-slate-700 mb-2">No Roadmap Available</h3>
        <p className="text-slate-500">
          {roadmap.note || 'Unable to generate a roadmap at this time.'}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {isFallback && (
        <div className="flex items-center gap-3 p-4 bg-amber-50 border border-amber-200 rounded-lg">
          <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0" />
          <div className="text-sm text-amber-800">
            <span className="font-medium">Deterministic Mode:</span>{' '}
            No AI roadmap model is currently configured, so this roadmap was
            generated deterministically from your skill-gap analysis.
          </div>
        </div>
      )}

      {/* Learning Steps (only when not trajectory-only mode) */}
      {!showTrajectoryOnly && roadmap.learning_steps.length > 0 && (
        <section className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-green-100 flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Learning Steps</h3>
              <p className="text-sm text-slate-500">
                Actionable milestones to close the gaps
              </p>
            </div>
          </div>
          <LearningSteps steps={roadmap.learning_steps} />
        </section>
      )}

      {/* Career Trajectory (only when trajectory-only mode) */}
      {showTrajectoryOnly && (roadmap.career_trajectory?.length ?? 0) > 0 && (
        <StrategicTrajectory phases={roadmap.career_trajectory} onSelect={setSelectedPhase} />
      )}

      {selectedPhase && (
        <PhaseDetailModal
          phase={selectedPhase}
          onClose={() => setSelectedPhase(null)}
          skills={skills ?? []}
          targetTitle={targetTitle ?? ''}
          matchedJobs={matchedJobs ?? []}
        />
      )}
    </div>
  );
}

function StrategicTrajectory({ phases, onSelect }: { phases: CareerPhase[]; onSelect: (p: CareerPhase) => void }) {
  return (
    <section className="bg-white rounded-xl border border-slate-200 p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-lg bg-primary-100 flex items-center justify-center">
          <Flag className="w-5 h-5 text-primary-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-slate-900">Strategic Career Trajectory</h3>
          <p className="text-sm text-slate-500">
            A long-term, multi-phase plan from entry-level readiness to a senior role
          </p>
        </div>
      </div>

      <div className="space-y-6">
        {phases.map((phase, index) => (
          <div
            key={`${phase.name}-${index}`}
            className="relative pl-10 cursor-pointer hover:bg-slate-50 hover:shadow-xl hover:scale-[1.02] rounded-xl transition-all duration-300"
            onClick={() => onSelect(phase)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(phase); } }}
          >
            <div className="absolute left-0 top-0 flex flex-col items-center">
              <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary-600 text-white text-sm font-bold">
                {index + 1}
              </div>
              {index < phases.length - 1 && (
                <div className="w-px flex-1 bg-primary-200 mt-2" aria-hidden="true" />
              )}
            </div>

            <div className="bg-slate-50 rounded-xl border border-slate-100 p-5">
              <div className="flex items-center justify-between flex-wrap gap-2 mb-1">
                <h4 className="text-base font-semibold text-slate-900">{phase.name}</h4>
                <span className="px-2.5 py-1 text-xs font-semibold text-primary-700 bg-primary-100 rounded-full">
                  Phase {index + 1}
                </span>
              </div>
              <p className="text-sm text-slate-600">{phase.focus}</p>
              {phase.salary_range_gbp && (
                <div className="flex items-center gap-1.5 mt-2 text-xs text-slate-500">
                  <PoundSterling className="w-3.5 h-3.5" />
                  <span>{phase.salary_range_gbp}</span>
                </div>
              )}
              {(phase.objectives?.length ?? 0) > 0 && (
                <ul className="mt-3 space-y-1.5">
                  {phase.objectives.map((objective, i) => (
                    <li key={`${objective}-${i}`} className="flex items-start gap-2 text-sm text-slate-700">
                      <CheckCircle className="w-4 h-4 text-primary-500 mt-0.5 flex-shrink-0" />
                      <span>{objective}</span>
                    </li>
                  ))}
                </ul>
              )}
              <div className="mt-3 flex items-center gap-2 text-xs text-slate-500">
                <ChevronRight className="w-3.5 h-3.5" />
                <span>Click to explore phase details →</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}