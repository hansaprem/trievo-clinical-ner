/**
 * API Service for interacting with the FastAPI Clinical NER backend.
 */

import { HealthResponse, PredictRequest, PredictResponse } from '../types/ner';

const PRODUCTION_BACKEND_URL = 'https://parents-acting-explanation-realty.trycloudflare.com';

const API_BASE_URL = (
  import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD ? PRODUCTION_BACKEND_URL : 'http://127.0.0.1:8000')
).replace(/\/+$/, '');

export class ApiError extends Error {
  statusCode?: number;

  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
  }
}

/**
 * Checks the operational health and loaded models status of the backend API.
 */
export async function checkHealth(timeoutMs = 5000): Promise<HealthResponse> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
      signal: controller.signal,
    });

    clearTimeout(timer);

    if (!res.ok) {
      throw new ApiError(`Health check failed with status: ${res.status}`, res.status);
    }

    const data: HealthResponse = await res.json();
    return data;
  } catch (err: unknown) {
    clearTimeout(timer);
    if (err instanceof ApiError) {
      throw err;
    }
    if (err instanceof Error && err.name === 'AbortError') {
      throw new ApiError('Connection to Clinical NER API timed out.', 408);
    }
    throw new ApiError(
      'Unable to connect to the Clinical NER API. Please verify the backend server is running on ' + API_BASE_URL,
      0
    );
  }
}

/**
 * Sends clinical text to the backend for multi-model entity extraction.
 */
export async function predictClinicalText(
  text: string,
  timeoutMs = 30000
): Promise<PredictResponse> {
  if (!text || !text.trim()) {
    throw new ApiError('Input text cannot be empty or whitespace-only.', 400);
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const payload: PredictRequest = { text };
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    clearTimeout(timer);

    if (!res.ok) {
      let errorDetail = `Inference failed with status ${res.status}`;
      try {
        const errorJson = await res.json();
        if (errorJson && errorJson.detail) {
          errorDetail = errorJson.detail;
        }
      } catch {
        // Fallback to default message
      }

      if (res.status === 400) {
        throw new ApiError(errorDetail, 400);
      } else if (res.status === 422) {
        throw new ApiError('Malformed request payload. Please check input text format.', 422);
      } else if (res.status >= 500) {
        throw new ApiError(
          'An internal inference error occurred while processing the request.',
          500
        );
      }
      throw new ApiError(errorDetail, res.status);
    }

    const data: PredictResponse = await res.json();
    return data;
  } catch (err: unknown) {
    clearTimeout(timer);
    if (err instanceof ApiError) {
      throw err;
    }
    if (err instanceof Error && err.name === 'AbortError') {
      throw new ApiError('Inference request timed out after 30 seconds.', 408);
    }
    throw new ApiError(
      'Unable to connect to the Clinical NER API. Please make sure the backend server is running.',
      0
    );
  }
}
