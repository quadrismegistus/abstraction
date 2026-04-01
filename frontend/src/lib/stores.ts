import { writable } from 'svelte/store';
import type { CorpusInfo, NormInfo } from './types';

export const norm = writable('Abs-Conc.Median.median');
export const selectedGenres = writable<string[]>(['Fiction', 'Poetry', 'Periodical']);
export const selectedCorpora = writable<string[]>([]);
export const yearRange = writable<[number, number]>([1580, 2020]);
export const chunkSize = writable(500);
export const periodMatched = writable(true);
export const globalLoading = writable(false);
export const loessSpan = writable(0.2);

// Cached metadata (loaded once)
export const corporaList = writable<CorpusInfo[]>([]);
export const normsList = writable<NormInfo[]>([]);
export const genresList = writable<string[]>([]);
