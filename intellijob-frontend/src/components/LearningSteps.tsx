import { useState } from 'react';
import { ChevronRight, TrendingUp } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { LearningStep } from '@/types/api';
import { StepDetailModal } from './StepDetailModal';

interface LearningStepsProps {
  steps: LearningStep[];
}

export function LearningSteps({ steps }: LearningStepsProps) {
  const [selectedStep, setSelectedStep] = useState<LearningStep | null>(null);

  if (!steps.length) return null;

  return (
    <>
      <section className="bg-white rounded-xl border border-slate-200 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-lg bg-green-100 flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-green-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-900">Learning Steps</h3>
            <p className="text-sm text-slate-500">
              Click any milestone to explore resources and a deep-dive
            </p>
          </div>
        </div>

        <ol className="space-y-3">
          {steps.map((step, index) => (
            <li key={step.id ?? `${step.title}-${index}`}>
              <div className="relative group">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedStep(step);
                  }}
                  className={cn(
                    'w-full flex items-center gap-4 text-left rounded-xl border bg-slate-50 p-4 transition-all duration-300',
                    'border-slate-100 hover:border-primary-300 hover:bg-white hover:shadow-xl hover:scale-[1.02]',
                    'focus:outline-none focus:ring-2 focus:ring-primary-500'
                  )}
                >
                  <span className="flex items-center justify-center w-7 h-7 rounded-full bg-primary-100 text-primary-600 text-sm font-bold flex-shrink-0">
                    {index + 1}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-slate-700 font-medium group-hover:text-primary-700 transition-colors">
                      {step.title}
                    </span>
                    <span className="block text-xs text-slate-500 mt-0.5">
                      ~{step.estimated_hours} hours · {step.primary_skill}
                    </span>
                  </span>
                  <ChevronRight className="w-5 h-5 text-slate-400 group-hover:text-primary-600 group-hover:translate-x-0.5 transition-all flex-shrink-0" />
                </button>

                <span className="pointer-events-none absolute right-14 top-1/2 -translate-y-1/2 px-2.5 py-1 text-xs font-medium text-white bg-slate-900 rounded-md shadow-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
                  Explore Step
                </span>
              </div>
            </li>
          ))}
        </ol>
      </section>

      {selectedStep && (
        <StepDetailModal step={selectedStep} onClose={() => setSelectedStep(null)} />
      )}
    </>
  );
}