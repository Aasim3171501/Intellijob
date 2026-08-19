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
}

export interface Roadmap {
  status: 'generated' | 'fallback' | 'pending';
  skill_gaps: string[];
  learning_steps: string[];
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