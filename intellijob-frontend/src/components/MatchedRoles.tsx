interface MatchedRolesProps {
  titles: string[];
}

export function MatchedRoles({ titles }: MatchedRolesProps) {
  const unique = [...new Set(titles.filter(Boolean))];
  if (unique.length === 0) return null;

  return (
    <section>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-primary-500"></span>
          Matched Roles
        </h3>
        <span className="text-sm text-slate-500">{unique.length} roles</span>
      </div>
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <div className="flex flex-wrap gap-2">
          {unique.map((title) => (
            <span
              key={title}
              className="px-3 py-1.5 text-sm font-medium text-slate-700 bg-slate-100 rounded-full"
            >
              {title}
            </span>
          ))}
        </div>
        <p className="mt-3 text-xs text-slate-500">
          Current-market roles used to shape your skill-gap analysis and roadmap.
        </p>
      </div>
    </section>
  );
}