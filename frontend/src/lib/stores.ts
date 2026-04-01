import { writable } from 'svelte/store';
import type { CorpusInfo, NormInfo } from './types';

export const norm = writable('Abs-Conc.Median.median');
export const selectedGenres = writable<string[]>([]);
export const selectedCorpora = writable<string[]>([]);
export const yearRange = writable<[number, number]>([1500, 2020]);
export const chunkSize = writable(500);

// Cached metadata (loaded once)
export const corporaList = writable<CorpusInfo[]>([]);
export const normsList = writable<NormInfo[]>([]);
export const genresList = writable<string[]>([]);
