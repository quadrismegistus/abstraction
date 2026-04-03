"""Pydantic response schemas for the API."""

from pydantic import BaseModel


class CorpusInfo(BaseModel):
    name: str
    n_texts: int
    year_min: float | None
    year_max: float | None
    genres: list[str]


class NormInfo(BaseModel):
    col: str
    source: str
    period: str
    label: str


class ArcBin(BaseModel):
    decade: int
    mean: float | None
    median: float | None
    q25: float | None
    q75: float | None
    n: int


class ArcAggregated(BaseModel):
    bins: list[ArcBin]
    total: int


class CorpusArcBin(BaseModel):
    decade: int
    mean: float | None
    n: int


class CorpusArc(BaseModel):
    corpus: str
    genre: str | None
    n_texts: int
    bins: list[CorpusArcBin]


class ArcText(BaseModel):
    id: str
    corpus: str
    year: float | None
    author: str | None
    title: str | None
    genre: str | None
    score: float | None


class ArcTexts(BaseModel):
    texts: list[ArcText]
    total: int
    page: int
    page_size: int


class TrajectoryChunk(BaseModel):
    index: int
    score: float | None
    n_words: int
    start_word: int


class TrajectoryResponse(BaseModel):
    metadata: dict
    chunks: list[TrajectoryChunk]
    overall_score: float | None


class PassageResponse(BaseModel):
    text: str
    body_html: str           # Color-styled HTML fragment (for web display)
    print_body_html: str     # Grayscale-styled HTML fragment (for inline print preview)
    print_html: str          # Full grayscale HTML document (for export/new window)
    n_abstract: int = 0
    n_concrete: int = 0
    n_neutral: int = 0


class AdjustedPoint(BaseModel):
    year: float
    score: float
    adjusted: float
    n_texts: int
    corpus: str | None = None


class LoessPoint(BaseModel):
    year: float
    fitted: float
    se_lo: float
    se_hi: float


class ArcStats(BaseModel):
    n_texts: int
    n_corpora: int
    breakpoint: int | None = None
    rise_slope: float | None = None        # per decade
    fall_slope: float | None = None        # per decade
    rise_slope_p: float | None = None
    fall_slope_p: float | None = None
    r2: float | None = None
    peak_year: int | None = None
    peak_score: float | None = None
    start_score: float | None = None       # score at earliest decade
    end_score: float | None = None         # score at latest decade
    change_sd: float | None = None         # peak-to-end in SD units


class GenreArc(BaseModel):
    genre: str
    points: list[AdjustedPoint]
    loess: list[LoessPoint]
    loess_raw: list[LoessPoint]       # LOESS on unadjusted scores
    loess_aggregate: list[LoessPoint]  # LOESS on text-weighted aggregate bins
    stats: ArcStats
    n_texts_total: int
    n_corpora: int


class ScoreRequest(BaseModel):
    text: str
    col: str = "Abs-Conc.Median.median"
