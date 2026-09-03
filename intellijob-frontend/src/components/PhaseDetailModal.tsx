import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  AlertTriangle,
  ArrowRight,
  Briefcase,
  CheckCircle,
  GraduationCap,
  Loader2,
  MapPin,
  PoundSterling,
  Sparkles,
  Target,
  TrendingUp,
  X,
} from 'lucide-react';
import { api } from '@/lib/api';
import { getLearningResources, type LearningResource } from '@/data/learningResources';
import type { CareerPhase, PhasePlan } from '@/types/api';

interface PhaseDetailModalProps {
  phase: CareerPhase;
  onClose: () => void;
  skills: string[];
  targetTitle: string;
  matchedJobs: string[];
}

const SKILL_KIND_STYLES: Record<string, { icon: typeof GraduationCap; label: string; classes: string }> = {
  documentation: { icon: GraduationCap, label: 'Documentation', classes: 'bg-blue-50 text-blue-600' },
  tutorial: { icon: GraduationCap, label: 'Tutorial / Guide', classes: 'bg-amber-50 text-amber-600' },
  interactive: { icon: MapPin, label: 'Interactive Path', classes: 'bg-violet-50 text-violet-600' },
};

function defaultResources(skill: string): LearningResource[] {
  const query = encodeURIComponent(skill.trim());
  return [
    { title: `W3Schools search for ${skill}`, url: `https://www.w3schools.com/search/search.asp?query=${query}`, domain: 'w3schools.com', kind: 'tutorial' },
    { title: `MDN search for ${skill}`, url: `https://developer.mozilla.org/search?q=${query}`, domain: 'developer.mozilla.org', kind: 'documentation' },
    { title: `roadmap.sh search for ${skill}`, url: `https://roadmap.sh/search?q=${query}`, domain: 'roadmap.sh', kind: 'interactive' },
  ];
}

async function fetchResources(skill: string): Promise<LearningResource[]> {
  try {
    const apiResources = await api.getLearningResources(skill);
    if (Array.isArray(apiResources) && apiResources.length > 0) return apiResources;
  } catch { /* fall through */ }
  try {
    const local = getLearningResources(skill);
    if (Array.isArray(local) && local.length > 0) return local;
  } catch { /* fall through */ }
  return defaultResources(skill);
}

function ResourceCard({ resource }: { resource: LearningResource }) {
  const { icon: Icon, label, classes } = SKILL_KIND_STYLES[resource.kind] || SKILL_KIND_STYLES.tutorial;
  return (
    <a
      href={resource.url}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex items-center gap-3 p-4 rounded-xl border border-slate-200 bg-white hover:border-primary-300 hover:bg-primary-50/50 transition-colors"
    >
      <div className={`flex items-center justify-center w-10 h-10 rounded-lg ${classes} flex-shrink-0`}>
        <Icon className="w-5 h-5" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-slate-900 truncate">{resource.title}</p>
        <p className="text-xs text-slate-500">{label} · {resource.domain}</p>
      </div>
    </a>
  );
}

export function PhaseDetailModal({ phase, onClose, skills, targetTitle, matchedJobs }: PhaseDetailModalProps) {
  const [resourceMap, setResourceMap] = useState<Record<string, LearningResource[]>>({});
  const [loadingSkills, setLoadingSkills] = useState<Set<string>>(new Set());
  const [plan, setPlan] = useState<PhasePlan | null>(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);
  const [isFallback, setIsFallback] = useState(false);

  useEffect(() => {
    const skillsToLoad = [...new Set([...phase.required_skills, ...phase.next_phase_unlocks])];
    skillsToLoad.forEach(skill => setLoadingSkills(prev => new Set(prev).add(skill)));

    skillsToLoad.forEach(async (skill) => {
      try {
        const resources = await fetchResources(skill);
        setResourceMap(prev => ({ ...prev, [skill]: resources }));
      } finally {
        setLoadingSkills(prev => { const next = new Set(prev); next.delete(skill); return next; });
      }
    });
  }, [phase]);

  const doGeneratePlan = async () => {
    setPlanLoading(true);
    setPlanError(null);
    try {
      const result = await api.getPhasePlan({
        skills,
        target_title: targetTitle,
        phase_name: phase.name,
        phase_focus: phase.focus,
        matched_jobs: matchedJobs,
      });
      setPlan(result);
      setIsFallback(false);
    } catch (err) {
      setPlanError(err instanceof Error ? err.message : 'Failed to generate plan');
    } finally {
      setPlanLoading(false);
    }
  };

  const handleGeneratePlan = () => {
    doGeneratePlan();
  };

  const handleRetryWithAI = () => {
    doGeneratePlan();
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handleKeyDown);
    document.body.style.overflow = 'hidden';
    return () => { document.removeEventListener('keydown', handleKeyDown); document.body.style.overflow = ''; };
  }, [onClose]);

  if (typeof document === 'undefined') return null;

  return createPortal(
    <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="relative w-full max-w-4xl bg-white rounded-2xl p-6 shadow-2xl border border-slate-200 z-10 max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        <div className="flex items-start justify-between gap-4 mb-6">
          <div className="pr-8">
            <h3 id="phase-modal-title" className="text-2xl font-bold text-slate-900">{phase.name}</h3>
            <p className="text-slate-600 mt-1">{phase.focus}</p>
            {phase.salary_range_gbp && (
              <div className="flex items-center gap-2 mt-2 text-sm text-slate-600">
                <PoundSterling className="w-4 h-4" />
                <span>{phase.salary_range_gbp}</span>
              </div>
            )}
          </div>
          <button onClick={onClose} aria-label="Close" className="flex items-center justify-center w-9 h-9 rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-700 transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Generate Plan Button / Loading UI */}
        {!plan && !planLoading && (
          <div className="text-center py-8">
            <button
              onClick={handleGeneratePlan}
              className="inline-flex items-center gap-2 px-6 py-3 bg-primary-600 text-white font-semibold rounded-lg hover:bg-primary-700 transition-colors"
            >
              <Sparkles className="w-5 h-5" />
              Generate My Personalized Plan
            </button>
            {planError && (
              <p className="mt-3 text-sm text-red-600">{planError}</p>
            )}
          </div>
        )}

        {/* Loading UI */}
        {!plan && planLoading && (
          <div className="text-center py-12 space-y-4">
            <div className="flex justify-center">
              <Loader2 className="w-12 h-12 animate-spin text-primary-600" />
            </div>
            <div className="space-y-2">
              <p className="text-slate-700 font-medium">Generating your personalized plan...</p>
              <p className="text-sm text-slate-500">This may take 30–90 seconds. We're crafting a detailed week-by-week plan with Tavily resources.</p>
            </div>
            <div className="w-72 mx-auto bg-slate-200 rounded-full h-2.5 overflow-hidden">
              <div className="bg-primary-600 h-full rounded-full animate-pulse" style={{ width: '60%' }} />
            </div>
            <p className="text-xs text-slate-400">Please wait...</p>
          </div>
        )}

        {planError && !planLoading && (
          <div className="text-center py-8">
            <button
              onClick={handleRetryWithAI}
              className="inline-flex items-center gap-2 px-6 py-3 bg-primary-600 text-white font-semibold rounded-lg hover:bg-primary-700 transition-colors"
            >
              <Sparkles className="w-5 h-5" />
              Try Again
            </button>
            <p className="mt-3 text-sm text-red-600">{planError}</p>
          </div>
        )}

        {/* Generated Plan Display */}
        {plan && (
          <div className="space-y-6">
            {/* Fallback warning banner */}
            {isFallback ? (
              <div className="flex items-start gap-3 p-4 bg-amber-50 border border-amber-200 rounded-lg mb-4">
                <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-amber-800 flex-1">
                  <span className="font-medium">Deterministic Mode:</span>{' '}
                  AI generation fell back to deterministic plan (rate limit or timeout).
                </div>
                <button
                  onClick={handleRetryWithAI}
                  disabled={planLoading}
                  className="ml-auto flex-shrink-0 px-4 py-2 bg-amber-600 text-white text-sm font-medium rounded-lg hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {planLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    'Retry with AI'
                  )}
                </button>
              </div>
            ) : null}

            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-slate-900">Your {plan.total_weeks}-Week Action Plan</h4>
              <button
                onClick={() => setPlan(null)}
                className="text-sm text-primary-600 hover:underline"
              >
                ← Back to phase overview
              </button>
            </div>

            {/* Key Projects */}
            {plan.key_projects.length > 0 && (
              <section>
                <h4 className="flex items-center gap-2 text-sm font-semibold text-slate-900 uppercase tracking-wide mb-3">
                  <CheckCircle className="w-4 h-4 text-primary-500" />
                  Key Portfolio Projects
                </h4>
                <ul className="space-y-2">
                  {plan.key_projects.map((proj, i) => (
                    <li key={i} className="flex items-start gap-3 text-sm text-slate-700 p-3 bg-slate-50 rounded-lg">
                      <CheckCircle className="w-4 h-4 text-primary-500 mt-0.5 flex-shrink-0" />
                      <span>{proj}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {/* Common Pitfalls */}
            {plan.common_pitfalls.length > 0 && (
              <section>
                <h4 className="flex items-center gap-2 text-sm font-semibold text-slate-900 uppercase tracking-wide mb-3">
                  <Target className="w-4 h-4 text-red-500" />
                  Common Pitfalls to Avoid
                </h4>
                <ul className="space-y-2">
                  {plan.common_pitfalls.map((pitfall, i) => (
                    <li key={i} className="flex items-start gap-3 text-sm text-slate-700 p-3 bg-red-50 rounded-lg border border-red-100">
                      <Target className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                      <span>{pitfall}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {/* Week-by-Week Plan */}
            <section>
              <h4 className="flex items-center gap-2 text-sm font-semibold text-slate-900 uppercase tracking-wide mb-3">
                <TrendingUp className="w-4 h-4 text-primary-500" />
                Weekly Milestones
              </h4>
              <div className="space-y-4">
                {plan.weeks.map((week) => (
                  <div key={week.week} className="bg-slate-50 rounded-xl border border-slate-200 p-5">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary-600 text-white text-sm font-bold flex-shrink-0">
                        {week.week}
                      </div>
                      <h5 className="font-semibold text-slate-900">{week.theme}</h5>
                    </div>
                    
                    {week.project && (
                      <div className="mb-3 p-3 bg-primary-50 rounded-lg border border-primary-100">
                        <p className="text-xs font-medium text-primary-700 mb-1">Portfolio Project:</p>
                        <p className="text-sm text-primary-600">{week.project}</p>
                      </div>
                    )}

                    {week.milestones.length > 0 && (
                      <div className="mb-3">
                        <p className="text-xs font-medium text-slate-600 mb-2">Milestones:</p>
                        <ul className="space-y-1">
                          {week.milestones.map((ms, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                              <ArrowRight className="w-3.5 h-3.5 text-primary-500 mt-0.5 flex-shrink-0" />
                              <span>{ms}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {(week.skills.length > 0 || (plan.resources && Object.keys(plan.resources).length > 0)) && (
                      <div>
                        <p className="text-xs font-medium text-slate-600 mb-2">Skills to focus on:</p>
                        <div className="flex flex-wrap gap-2">
                          {week.skills.map((skill) => (
                            <span key={skill} className="px-2.5 py-1 text-xs bg-white text-slate-700 rounded-full border border-slate-200">
                              {skill}
                            </span>
                          ))}
                        </div>
                        {week.skills.length > 0 && plan.resources && (
                          <div className="mt-3 space-y-2">
                            {week.skills
                              .filter(s => plan.resources?.[s]?.length)
                              .map((skill) => (
                                <div key={skill}>
                                  <p className="text-xs font-medium text-slate-600 mb-1">{skill} resources:</p>
                                  <div className="flex flex-wrap gap-2">
                                    {(plan.resources[skill] || []).map((res) => (
                                      <ResourceCard key={res.url} resource={res} />
                                    ))}
                                  </div>
                                </div>
                              ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>

            {/* Resource Priorities */}
            {plan.resource_priorities.length > 0 && (
              <section>
                <h4 className="flex items-center gap-2 text-sm font-semibold text-slate-900 uppercase tracking-wide mb-3">
                  <GraduationCap className="w-4 h-4 text-primary-500" />
                  Priority Learning Order
                </h4>
                <p className="text-sm text-slate-600 mb-3">
                  Focus on these first for maximum impact:
                </p>
                <div className="flex flex-wrap gap-2">
                  {plan.resource_priorities.map((skill, i) => (
                    <span key={i} className="px-3 py-1.5 text-sm bg-primary-50 text-primary-700 rounded-full border border-primary-200">
                      {i + 1}. {skill}
                    </span>
                  ))}
                </div>
              </section>
            )}
          </div>
        )}

        {/* Static Phase Info (collapsible) */}
        <details className="mt-8 border-t border-slate-200 pt-6">
          <summary className="cursor-pointer font-medium text-slate-700">
            Show phase overview (typical titles, objectives, required skills)
          </summary>
          <div className="mt-4 space-y-6">
            {phase.typical_titles.length > 0 && (
              <section>
                <h4 className="flex items-center gap-2 text-sm font-semibold text-slate-900 uppercase tracking-wide mb-3">
                  <Briefcase className="w-4 h-4 text-primary-500" />
                  Typical Job Titles
                </h4>
                <div className="flex flex-wrap gap-2">
                  {phase.typical_titles.map((title, i) => (
                    <span key={i} className="px-3 py-1.5 text-sm bg-primary-50 text-primary-700 rounded-full border border-primary-200">
                      {title}
                    </span>
                  ))}
                </div>
              </section>
            )}

            {phase.objectives.length > 0 && (
              <section>
                <h4 className="flex items-center gap-2 text-sm font-semibold text-slate-900 uppercase tracking-wide mb-3">
                  <Target className="w-4 h-4 text-primary-500" />
                  Objectives
                </h4>
                <ul className="space-y-2">
                  {phase.objectives.map((obj, i) => (
                    <li key={i} className="flex items-start gap-3 text-sm text-slate-700">
                      <TrendingUp className="w-4 h-4 text-primary-500 mt-0.5 flex-shrink-0" />
                      <span>{obj}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {phase.required_skills.length > 0 && (
              <section>
                <h4 className="flex items-center gap-2 text-sm font-semibold text-slate-900 uppercase tracking-wide mb-3">
                  <Sparkles className="w-4 h-4 text-primary-500" />
                  Required Skills
                </h4>
                <div className="flex flex-wrap gap-2 mb-4">
                  {phase.required_skills.map((skill, i) => (
                    <span key={i} className="px-3 py-1.5 text-sm bg-slate-100 text-slate-700 rounded-lg border border-slate-200">
                      {skill}
                    </span>
                  ))}
                </div>
                <div className="space-y-3">
                  {phase.required_skills.map((skill) => (
                    <div key={skill}>
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="font-medium text-slate-700">{skill}</span>
                        {loadingSkills.has(skill) && <Loader2 className="w-4 h-4 text-primary-500 animate-spin" />}
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {(resourceMap[skill] || []).map((res) => (
                          <ResourceCard key={res.url} resource={res} />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {phase.next_phase_unlocks.length > 0 && (
              <section>
                <h4 className="flex items-center gap-2 text-sm font-semibold text-slate-900 uppercase tracking-wide mb-3">
                  <GraduationCap className="w-4 h-4 text-primary-500" />
                  Unlocks Next Phase: {phase.name.includes('Entry') ? 'Mid-Level' : phase.name.includes('Mid') ? 'Senior' : 'Leadership'}
                </h4>
                <p className="text-sm text-slate-600 mb-3">
                  Master these to progress to the next career phase:
                </p>
                <div className="flex flex-wrap gap-2 mb-4">
                  {phase.next_phase_unlocks.map((skill, i) => (
                    <span key={i} className="px-3 py-1.5 text-sm bg-amber-50 text-amber-700 rounded-full border border-amber-200">
                      {skill}
                    </span>
                  ))}
                </div>
                <div className="space-y-3">
                  {phase.next_phase_unlocks.map((skill) => (
                    <div key={skill}>
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="font-medium text-slate-700">{skill}</span>
                        {loadingSkills.has(skill) && <Loader2 className="w-4 h-4 text-primary-500 animate-spin" />}
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {(resourceMap[skill] || []).map((res) => (
                          <ResourceCard key={res.url} resource={res} />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>
        </details>
      </div>
    </div>,
    document.body,
  );
}