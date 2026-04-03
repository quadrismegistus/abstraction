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

export interface CorpusArcBin {
  decade: number;
  mean: number | null;
  n: number;
}

export interface CorpusArc {
  corpus: string;
  genre: string | null;
  n_texts: number;
  bins: CorpusArcBin[];
}

export interface ArcText {
  id: string;
  corpus: string;
  year: number | null;
  author: string | null;
  title: string | null;
  genre: string | null;
  genre_raw: string | null;
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

export interface AdjustedPoint {
  year: number;
  score: number;
  adjusted: number;
  n_texts: number;
  corpus: string | null;
}

export interface LoessPoint {
  year: number;
  fitted: number;
  se_lo: number;
  se_hi: number;
}

export interface ArcStats {
  n_texts: number;
  n_corpora: number;
  breakpoint: number | null;
  rise_slope: number | null;
  fall_slope: number | null;
  rise_slope_p: number | null;
  fall_slope_p: number | null;
  r2: number | null;
  peak_year: number | null;
  peak_score: number | null;
  start_score: number | null;
  end_score: number | null;
  change_sd: number | null;
}

export interface GenreArc {
  genre: string;
  points: AdjustedPoint[];
  loess: LoessPoint[];
  loess_raw: LoessPoint[];
  loess_aggregate: LoessPoint[];
  stats: ArcStats;
  n_texts_total: number;
  n_corpora: number;
}

export interface PassageResponse {
  text: string;
  body_html: string;         // Color background HTML fragment (for web)
  print_body_html: string;   // Grayscale HTML fragment (for inline print preview)
  print_html: string;        // Full grayscale HTML document (for export/new window)
  n_abstract: number;
  n_concrete: number;
  n_neutral: number;
}
