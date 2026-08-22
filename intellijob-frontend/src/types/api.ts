export interface Skill {
  name: string;
}

export interface JobMatch {
  id: string;
  title: string;
  company: string;
  location_display: string;
  similarity_score: number;
  description_excerpt: string;
}

export interface CareerPhase {
  name: string;
  focus: string;
  objectives: string[];
  typical_titles: string[];
  salary_range_gbp: string;
  required_skills: string[];
  next_phase_unlocks: string[];
}

export interface LearningResource {
  title: string;
  url: string;
  domain: string;
  kind: 'documentation' | 'tutorial' | 'interactive';
}

export interface LearningStep {
  id: string;
  title: string;
  overview: string;
  primary_skill: string;
  estimated_hours: number;
}

export interface Roadmap {
  status: 'generated' | 'fallback' | 'pending';
  skill_gaps: string[];
  learning_steps: LearningStep[];
  estimated_timeline_weeks: Record<string, string>;
  career_trajectory: CareerPhase[];
  matched_jobs: string[];
  extracted_skill_count: number;
  note?: string;
}

export interface CareerPathway {
  key: string;
  name: string;
  description: string;
  similarity_score: number;
  score: number;
  coverage: number;
  matched_skills: string[];
  match_count: number;
  matches: JobMatch[];
  roadmap: Roadmap;
}

export interface AnalyzeResponse {
  extracted_skills: string[];
  career_mode: 'targeted' | 'discovery';
  career_pathways: CareerPathway[];
  matches: JobMatch[];
  roadmap: Roadmap;
}

export interface AnalyzeRequest {
  file: File;
  target_title: string;
}

export interface PhaseWeek {
  week: number;
  theme: string;
  milestones: string[];
  skills: string[];
  project: string;
}

export interface PhasePlan {
  phase_name: string;
  phase_focus: string;
  total_weeks: number;
  weeks: PhaseWeek[];
  key_projects: string[];
  common_pitfalls: string[];
  resource_priorities: string[];
  resources: Record<string, LearningResource[]>;
}