import type {
  CorpusInfo, NormInfo, ArcTexts, GenreArc, AggGenreArc,
  TrajectoryResponse, PassageResponse
} from './types';

// API base URL. Override with VITE_API_BASE (absolute like
// "https://example.org/api", or relative like "/api" behind a reverse proxy).
// Default: same protocol+hostname as the page, port 1709 (Johnson's DOB).
export const API_BASE: string =
  import.meta.env.VITE_API_BASE ??
  (typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.hostname}:1709/api`
    : 'http://localhost:1709/api');

async function fetchJson<T>(path: string, params?: Record<string, string | string[]>): Promise<T> {
  // Second arg makes relative API_BASE values (e.g. "/api") resolve against the page origin.
  const url = new URL(
    `${API_BASE}${path}`,
    typeof window !== 'undefined' ? window.location.origin : 'http://localhost'
  );
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
export const fetchRawCorpora = () => fetchJson<{id: string; label: string; lang: string; n_texts: number}[]>('/meta/raw-corpora');

// Arc
export function fetchArcAggregate(params: {
  col?: string; genre?: string[]; corpus?: string[];
  year_min?: string; year_max?: string;
  period_matched?: string; loess_span?: string; bin_size?: string;
  split_by?: string; is_translated?: string;
  dedup?: string;
}) {
  return fetchJson<AggGenreArc[]>('/arc/aggregate', params);
}

export function fetchArcByGenre(params: {
  col?: string; genre?: string[]; corpus?: string[];
  year_min?: string; year_max?: string;
  period_matched?: string; corpus_adjusted?: string; loess_span?: string; model?: string; bin_size?: string;
  is_translated?: string;
  dedup?: string;
}) {
  return fetchJson<GenreArc[]>('/arc/by-genre', params);
}

export function fetchArcTexts(params: {
  col?: string; genre?: string[]; corpus?: string[];
  year_min?: string; year_max?: string;
  genre_raw?: string;
  page?: string; page_size?: string;
  period_matched?: string;
  dedup?: string;
}) {
  return fetchJson<ArcTexts>('/arc/texts', params);
}

// Trajectory
export function fetchTrajectory(corpus: string, textId: string, params?: {
  col?: string; chunk_size?: string; period_matched?: string;
}) {
  return fetchJson<TrajectoryResponse>(
    `/trajectory/${corpus}/${textId}`, params
  );
}

// Passage
export function fetchPassage(corpus: string, textId: string, chunkIndex: number, params?: {
  col?: string; chunk_size?: string; period_matched?: string;
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
