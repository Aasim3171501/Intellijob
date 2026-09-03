import axios from 'axios';
import type { AxiosInstance, AxiosError } from 'axios';
import type { AnalyzeResponse, AnalyzeRequest, LearningResource, PhasePlan, RoadmapResponse } from '@/types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 300000, // 5 min headroom for parallel roadmap generation (discovery mode)
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        const data = error.response?.data as { error?: string } | undefined;
        const message = data?.error || error.message || 'An unexpected error occurred';
        return Promise.reject(new Error(message));
      }
    );
  }

  async analyzeResume(request: AnalyzeRequest): Promise<AnalyzeResponse> {
    const formData = new FormData();
    formData.append('file', request.file);
    formData.append('target_title', request.target_title);

    const response = await this.client.post<AnalyzeResponse>('/analyze/', formData);
    return response.data;
  }

  async getLearningResources(skill: string): Promise<LearningResource[]> {
    const response = await this.client.get<{ resources: LearningResource[] }>('/learning-resources/', {
      params: { skill },
      timeout: 10000,
    });
    return response.data.resources;
  }

  async getPhasePlan(data: {
    skills: string[];
    target_title: string;
    phase_name: string;
    phase_focus: string;
    matched_jobs: string[];
  }): Promise<PhasePlan> {
    const response = await this.client.post<PhasePlan>('/phase-plan/', data, {
      timeout: 200000, // 200s - must exceed backend LLM_TIMEOUT_SECONDS (180s)
      headers: { 'Content-Type': 'application/json' },
    });
    return response.data;
  }

  async getPathwayRoadmap(data: {
    skills: string[];
    pathway_key: string;
    target_title: string;
  }): Promise<RoadmapResponse> {
    const response = await this.client.post<RoadmapResponse>('/pathway-roadmap/', data, {
      timeout: 200000, // 200s - must exceed backend LLM_TIMEOUT_SECONDS (180s)
      headers: { 'Content-Type': 'application/json' },
    });
    return response.data;
  }
}

export const api = new ApiClient();