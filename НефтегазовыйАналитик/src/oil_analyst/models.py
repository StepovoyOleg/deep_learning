from datetime import date as Date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class DocumentMetadata(BaseModel):
    document_name: str
    document_title: str | None = None
    organization: str
    report_name: str
    report_date: Date | None = None
    source: str
    page: int = Field(ge=1)
    date: Date | None = None
    source_type: Literal["report"] = "report"
    chunk_id: str
    section_title: str | None = None
    table_title: str | None = None
    content_type: Literal["text", "table", "graph_caption"] = "text"


class RetrievedDocument(BaseModel):
    text: str
    metadata: DocumentMetadata
    score: float = Field(ge=0)


class WebResult(BaseModel):
    title: str
    url: HttpUrl
    content: str
    published_date: Date | None = None
    score: float = 0.0


class ForecastPoint(BaseModel):
    period: str
    value: float
    lower_bound: float
    upper_bound: float


class ForecastResult(BaseModel):
    instrument: str
    forecast_horizon: int
    method: Literal["sarima", "exponential_smoothing"]
    forecast: list[ForecastPoint]
    lower_bound: list[float]
    upper_bound: list[float]
    metrics: dict[str, float | None]
    interpretation: str
    assumptions: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    in_scope: bool = True
    needs_web: bool = False
    needs_forecast: bool = False
    web_only: bool = False
    instrument: str | None = None
    horizon: int | None = None
    reasons: list[str] = Field(default_factory=list)
    metadata_filters: "MetadataFilters" = Field(default_factory=lambda: MetadataFilters())


class MetadataFilters(BaseModel):
    organization: Literal["OPEC", "EIA"] | None = None
    report_date: str | None = None
    report_name: str | None = None

    def active(self) -> bool:
        return any((self.organization, self.report_date, self.report_name))


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1)
    source_refs: list[str] = Field(default_factory=list)
    uncertainty: str | None = None


class GeneratedNarrative(BaseModel):
    """LLM-owned fields; provenance is attached by the backend."""
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1)
    uncertainty: str | None = None


class Source(BaseModel):
    kind: Literal["report", "web"]
    citation: str
    url: HttpUrl | None = None


class AnalystResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    forecast: ForecastResult | None = None
    route: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)
    metadata_filters: MetadataFilters = Field(default_factory=MetadataFilters)
    uncertainty: str | None = None
