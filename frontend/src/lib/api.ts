import type {
  CorpusInfo, NormInfo, ArcAggregated, ArcTexts, CorpusArc, GenreArc,
  TrajectoryResponse, PassageResponse
} from './types';

// Use same hostname as the page, port 1709 for the API (Johnson's DOB)
const API_BASE = typeof window !== 'undefined'
  ? `http://${window.location.hostname}:1709/api`
  : 'http://localhost:1709/api';

async function fetchJson<T>(path: string, params?: Record<string, string | string[]>): Promise<T> {
  const url = new URL(`${API_BASE}${path}`);
  if (params) {
    for (const [key, val] of Object.entries(params)) {
      if (Array.isArray(val)) {
        for (const v of val) url.searchParams.append(key, v);
      } else if (val !== undefined && val !== '') {
        url.searchParams.set(key, val);
      }
    }
  }
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

// Meta
export const fetchCorpora = () => fetchJson<CorpusInfo[]>('/meta/corpora');
export const fetchNorms = () => fetchJson<NormInfo[]>('/meta/norms');
export const fetchGenres = () => fetchJson<string[]>('/meta/genres');

// Arc
export function fetchArcAggregated(params: {
  col?: string; genre?: string[]; corpus?: string[];
  year_min?: string; year_max?: string; bin_size?: string;
}) {
  return fetchJson<ArcAggregated>('/arc/aggregated', params);
}

export function fetchArcByGenre(params: {
  col?: string; genre?: string[]; corpus?: string[];
  year_min?: string; year_max?: string;
  period_matched?: string; loess_span?: string;
}) {
  return fetchJson<GenreArc[]>('/arc/by-genre', params);
}

export function fetchArcByCorpus(params: {
  col?: string; genre?: string[]; corpus?: string[];
  year_min?: string; year_max?: string; bin_size?: string;
}) {
  return fetchJson<CorpusArc[]>('/arc/by-corpus', params);
}

export function fetchArcTexts(params: {
  col?: string; genre?: string[]; corpus?: string[];
  year_min?: string; year_max?: string;
  page?: string; page_size?: string;
}) {
  return fetchJson<ArcTexts>('/arc/texts', params);
}

// Trajectory
export function fetchTrajectory(corpus: string, textId: string, params?: {
  col?: string; chunk_size?: string;
}) {
  return fetchJson<TrajectoryResponse>(
    `/trajectory/${corpus}/${textId}`, params
  );
}

// Passage
export function fetchPassage(corpus: string, textId: string, chunkIndex: number, params?: {
  col?: string; chunk_size?: string;
}) {
  return fetchJson<PassageResponse>(
    `/passage/${corpus}/${textId}/${chunkIndex}`, params
  );
}

export async function scoreText(text: string, col?: string): Promise<PassageResponse> {
  const res = await fetch(`${API_BASE}/passage/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, col: col || 'Abs-Conc.Median.median' }),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
