import { writable } from 'svelte/store';
import type { CorpusInfo, NormInfo } from './types';

export const norm = writable('Abs-Conc.Median.median');
export const selectedGenres = writable<string[]>(['arc_fiction']);
export const selectedCorpora = writable<string[]>([]);
export const yearRange = writable<[number, number]>([1565, 2020]);
export const chunkSize = writable(500);
export const periodMatched = writable(true);
export const globalLoading = writable(false);
export const loessSpan = writable(0.2);
export const adjustModel = writable('quadratic');
export const corpusAdjusted = writable(false);
export const binSize = writable(10);
export const corpusSmoothed = writable(false);
export const splitBy = writable('');
export const translatedFilter = writable('');  // '' = all, 'true' = only translated, 'false' = only originals

// Cached metadata (loaded once)
export const corporaList = writable<CorpusInfo[]>([]);
export const normsList = writable<NormInfo[]>([]);
export const genresList = writable<string[]>([]);
