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


class ScoredWord(BaseModel):
    position: int
    word: str
    score: float | None
    is_abstract: bool
    is_concrete: bool


class PassageResponse(BaseModel):
    text: str
    words: list[ScoredWord]
    html: str


class ScoreRequest(BaseModel):
    text: str
    col: str = "Abs-Conc.Median.median"
