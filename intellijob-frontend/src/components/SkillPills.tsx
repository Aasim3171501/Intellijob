interface SkillPillsProps {
  skills: string[];
  title?: string;
  maxDisplay?: number;
}

export function SkillPills({ skills, title = 'Extracted Skills', maxDisplay = 20 }: SkillPillsProps) {
  if (!skills || skills.length === 0) {
    return (
      <div className="text-center py-8 text-slate-500">
        <p>No skills extracted from the resume.</p>
      </div>
    );
  }

  const displaySkills = skills.slice(0, maxDisplay);
  const remaining = skills.length - maxDisplay;

  return (
    <div>
      <h3 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-primary-500"></span>
        {title} ({skills.length})
      </h3>
      <div className="flex flex-wrap gap-2">
        {displaySkills.map((skill, index) => (
          <span
            key={`${skill}-${index}`}
            className="skill-badge"
          >
            {skill}
          </span>
        ))}
        {remaining > 0 && (
          <span className="skill-badge bg-slate-100 text-slate-600">
            +{remaining} more
          </span>
        )}
      </div>
    </div>
  );
}