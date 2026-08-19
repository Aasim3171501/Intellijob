import { AlertTriangle, Clock, TrendingUp, Target, CheckCircle, AlertCircle, Info, Flag } from 'lucide-react';
import type { Roadmap, CareerPhase } from '@/types/api';

interface RoadmapViewProps {
  roadmap: Roadmap;
}

export function RoadmapView({ roadmap }: RoadmapViewProps) {
  const isFallback = roadmap.status === 'fallback';

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

  if (!roadmap.skill_gaps.length && !roadmap.learning_steps.length && !Object.keys(roadmap.estimated_timeline_weeks).length) {
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
            <span className="font-medium">Fallback Mode:</span>{' '}
            The roadmap was generated using a deterministic fallback because the local
            LLM service is unavailable. Results are based on skill-gap analysis only.
          </div>
        </div>
      )}

      {(roadmap.career_trajectory?.length ?? 0) > 0 && <StrategicTrajectory phases={roadmap.career_trajectory} />}

      {roadmap.skill_gaps.length > 0 && (
        <section className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-red-100 flex items-center justify-center">
              <AlertCircle className="w-5 h-5 text-red-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Skill Gaps</h3>
              <p className="text-sm text-slate-500">
                Missing competencies required for your target role
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {roadmap.skill_gaps.map((gap, index) => (
              <span
                key={`${gap}-${index}`}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-50 text-red-700 border border-red-200 text-sm font-medium"
              >
                <Target className="w-3.5 h-3.5" />
                {gap}
              </span>
            ))}
          </div>
        </section>
      )}

      {roadmap.learning_steps.length > 0 && (
        <section className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-green-100 flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Learning Steps</h3>
              <p className="text-sm text-slate-500">
                Actionable milestones to close the skill gaps
              </p>
            </div>
          </div>
          <ol className="space-y-4">
            {roadmap.learning_steps.map((step, index) => (
              <li key={`${step}-${index}`} className="relative pl-10">
                <div className="absolute left-0 top-1 flex items-center justify-center w-6 h-6 rounded-full bg-primary-100 text-primary-600 text-sm font-bold">
                  {index + 1}
                </div>
                <div className="bg-slate-50 rounded-lg p-4 border border-slate-100">
                  <p className="text-slate-700">{step}</p>
                </div>
              </li>
            ))}
          </ol>
        </section>
      )}

      {Object.keys(roadmap.estimated_timeline_weeks).length > 0 && (
        <section className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center">
              <Clock className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Estimated Timeline</h3>
              <p className="text-sm text-slate-500">
                Suggested duration for each learning phase
              </p>
            </div>
          </div>
          <div className="space-y-3">
            {Object.entries(roadmap.estimated_timeline_weeks).map(([phase, weeks], index) => (
              <div
                key={`${phase}-${index}`}
                className="flex items-center justify-between p-4 bg-slate-50 rounded-lg border border-slate-100"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center">
                    <CheckCircle className="w-4 h-4 text-primary-600" />
                  </div>
                  <span className="font-medium text-slate-900">{phase}</span>
                </div>
                <span className="px-3 py-1 text-sm font-semibold text-primary-700 bg-primary-100 rounded-full">
                  {weeks}
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

      {(roadmap.matched_jobs?.length ?? 0) > 0 && (
        <section className="bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center">
              <Info className="w-5 h-5 text-slate-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Based on Matched Jobs</h3>
              <p className="text-sm text-slate-500">
                These job specifications informed the roadmap
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {roadmap.matched_jobs.map((job, index) => (
              <span
                key={`${job}-${index}`}
                className="px-3 py-1 text-sm bg-slate-100 text-slate-700 rounded-lg border border-slate-200"
              >
                {job}
              </span>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function StrategicTrajectory({ phases }: { phases: CareerPhase[] }) {
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
          <div key={`${phase.name}-${index}`} className="relative pl-10">
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
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}