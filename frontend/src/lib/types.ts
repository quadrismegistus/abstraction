export interface CorpusInfo {
  name: string;
  n_texts: number;
  year_min: number | null;
  year_max: number | null;
  genres: string[];
}

export interface NormInfo {
  col: string;
  source: string;
  period: string;
  label: string;
}

export interface ArcBin {
  decade: number;
  mean: number | null;
  median: number | null;
  q25: number | null;
  q75: number | null;
  n: number;
}

export interface ArcAggregated {
  bins: ArcBin[];
  total: number;
}

export interface ArcText {
  id: string;
  corpus: string;
  year: number | null;
  author: string | null;
  title: string | null;
  genre: string | null;
  score: number | null;
}

export interface ArcTexts {
  texts: ArcText[];
  total: number;
  page: number;
  page_size: number;
}

export interface TrajectoryChunk {
  index: number;
  score: number | null;
  n_words: number;
  start_word: number;
}

export interface TrajectoryResponse {
  metadata: Record<string, any>;
  chunks: TrajectoryChunk[];
  overall_score: number | null;
}

export interface ScoredWord {
  position: number;
  word: string;
  score: number | null;
  is_abstract: boolean;
  is_concrete: boolean;
}

export interface PassageResponse {
  text: string;
  words: ScoredWord[];
  html: string;
}
