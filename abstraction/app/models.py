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


class PassageToken(BaseModel):
    """A token in a passage — either a scored word or punctuation/whitespace."""
    text: str
    is_punct: bool
    score: float | None = None
    is_abstract: bool = False
    is_concrete: bool = False


class PassageResponse(BaseModel):
    text: str
    tokens: list[PassageToken]
    html: str
    n_abstract: int = 0
    n_concrete: int = 0
    n_neutral: int = 0


class ScoreRequest(BaseModel):
    text: str
    col: str = "Abs-Conc.Median.median"
