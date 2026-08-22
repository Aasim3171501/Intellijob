import { useState } from 'react';

interface SkillPillsProps {
  skills: string[];
  title?: string;
  maxDisplay?: number;
}

export function SkillPills({ skills, title = 'Extracted Skills', maxDisplay = 20 }: SkillPillsProps) {
  const [expanded, setExpanded] = useState(false);

  if (!skills || skills.length === 0) {
    return (
      <div className="text-center py-8 text-slate-500">
        <p>No skills extracted from the resume.</p>
      </div>
    );
  }

  const showAll = expanded || skills.length <= maxDisplay;
  const visibleSkills = showAll ? skills : skills.slice(0, maxDisplay);
  const hidden = skills.length - visibleSkills.length;

  return (
    <div>
      <h3 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-primary-500"></span>
        {title} ({skills.length})
      </h3>
      <div className="flex flex-wrap gap-2">
        {visibleSkills.map((skill, index) => (
          <span
            key={`${skill}-${index}`}
            className="skill-badge"
          >
            {skill}
          </span>
        ))}
        {hidden > 0 && (
          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            className="skill-badge bg-slate-100 text-slate-600 hover:bg-slate-200 transition-colors cursor-pointer"
          >
            {expanded ? 'Show less' : `+${hidden} more`}
          </button>
        )}
      </div>
    </div>
  );
}