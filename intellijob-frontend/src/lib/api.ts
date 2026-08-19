import axios from 'axios';
import type { AxiosInstance, AxiosError } from 'axios';
import type { AnalyzeResponse, AnalyzeRequest } from '@/types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 300000, // 5 minutes for LLM generation (discovery mode: 3x Ollama roadmaps)
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
}

export const api = new ApiClient();