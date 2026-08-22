import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  BookOpen,
  Clock,
  ExternalLink,
  GraduationCap,
  Loader2,
  Map,
  Sparkles,
  X,
} from 'lucide-react';
import { getKeyTopics, getLearningResources, type LearningResource, type ResourceKind } from '@/data/learningResources';
import { api } from '@/lib/api';
import type { LearningStep } from '@/types/api';

interface StepDetailModalProps {
  step: LearningStep;
  onClose: () => void;
}

const KIND_STYLES: Record<ResourceKind, { icon: typeof BookOpen; label: string; classes: string }> = {
  documentation: { icon: BookOpen, label: 'Documentation', classes: 'bg-blue-50 text-blue-600' },
  tutorial: { icon: GraduationCap, label: 'Tutorial / Guide', classes: 'bg-amber-50 text-amber-600' },
  interactive: { icon: Map, label: 'Interactive Path', classes: 'bg-violet-50 text-violet-600' },
};

const GENERIC_TOPICS = ['Hands-on Practice', 'Real-World Project', 'Best Practices'];

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

function safeTopics(skill: string): string[] {
  try {
    const topics = getKeyTopics(skill);
    if (Array.isArray(topics) && topics.length > 0) return topics;
  } catch { /* fall through */ }
  return GENERIC_TOPICS;
}

function ResourceCard({ resource }: { resource: LearningResource }) {
  const { icon: Icon, label, classes } = KIND_STYLES[resource.kind] || KIND_STYLES.tutorial;
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
      <span className="flex items-center gap-1 px-3 py-1.5 text-xs font-semibold text-primary-700 bg-primary-100 rounded-lg group-hover:bg-primary-600 group-hover:text-white transition-colors">
        <ExternalLink className="w-3.5 h-3.5" />
        Open Link
      </span>
    </a>
  );
}

export function StepDetailModal({ step, onClose }: StepDetailModalProps) {
  const [resources, setResources] = useState<LearningResource[]>([]);
  const [loading, setLoading] = useState(true);
  const topics = safeTopics(step.primary_skill);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const data = await fetchResources(step.primary_skill);
        if (!cancelled) setResources(data);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [step.primary_skill]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handleKeyDown);
    document.body.style.overflow = 'hidden';
    return () => { document.removeEventListener('keydown', handleKeyDown); document.body.style.overflow = ''; };
  }, [onClose]);

  if (typeof document === 'undefined') return null;

  return createPortal(
    <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm" role="dialog" aria-modal="true" aria-labelledby="step-modal-title" onClick={onClose}>
      <div className="relative w-full max-w-2xl bg-white rounded-2xl p-6 shadow-2xl border border-slate-200 z-10 max-h-[85vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between gap-4 mb-6">
          <div className="pr-8">
            <div className="flex flex-wrap items-center gap-2 mb-2">
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-primary-100 text-primary-700 text-xs font-semibold">
                <Sparkles className="w-3.5 h-3.5" />
                {step.primary_skill}
              </span>
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 text-xs font-semibold">
                <Clock className="w-3.5 h-3.5" />
                ~{step.estimated_hours} hours
              </span>
            </div>
            <h3 id="step-modal-title" className="text-xl font-bold text-slate-900 leading-snug">
              {step.title}
            </h3>
          </div>
          <button type="button" onClick={onClose} aria-label="Close" className="flex items-center justify-center w-9 h-9 rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-700 transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-6">
          <section>
            <h4 className="text-sm font-semibold text-slate-900 uppercase tracking-wide mb-2">Overview</h4>
            <p className="text-sm text-slate-600 leading-relaxed">{step.overview}</p>
          </section>

          {topics.length > 0 && (
            <section>
              <h4 className="text-sm font-semibold text-slate-900 uppercase tracking-wide mb-2">Key Sub-Topics</h4>
              <div className="flex flex-wrap gap-2">
                {topics.map((topic) => (
                  <span key={topic} className="px-3 py-1.5 text-sm text-slate-700 bg-slate-100 border border-slate-200 rounded-lg">
                    {topic}
                  </span>
                ))}
              </div>
            </section>
          )}

          <section>
            <h4 className="text-sm font-semibold text-slate-900 uppercase tracking-wide mb-1">Recommended Learning Resources</h4>
            <p className="text-xs text-slate-500 mb-3">Curated links for {step.primary_skill} — official docs, tutorials, and learning paths.</p>
            <div className="space-y-2.5">
              {loading ? (
                <div className="flex justify-center py-4">
                  <Loader2 className="w-5 h-5 text-primary-500 animate-spin" />
                </div>
              ) : (
                resources.map((resource) => (
                  <ResourceCard key={resource.url} resource={resource} />
                ))
              )}
            </div>
          </section>
        </div>
      </div>
    </div>,
    document.body,
  );
}