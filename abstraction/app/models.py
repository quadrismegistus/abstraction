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
    body_html: str       # HTML fragment with data-z attrs, no inline styles (for web)
    print_html: str      # Full HTML document with grayscale inline styles (for export)
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


class GenreArc(BaseModel):
    genre: str
    points: list[AdjustedPoint]
    loess: list[LoessPoint]
    n_texts_total: int
    n_corpora: int


class ScoreRequest(BaseModel):
    text: str
    col: str = "Abs-Conc.Median.median"
