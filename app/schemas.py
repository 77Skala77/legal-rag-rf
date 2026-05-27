from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str
    database: str


class IngestFileRequest(BaseModel):
    path: str | None = Field(
        default=None, description="Path to a UTF-8 .txt or .md file available to the API process."
    )
    text: str | None = Field(default=None, min_length=1, description="Document plain text.")
    title: str = Field(min_length=1, max_length=500)
    source_url: str | None = Field(default=None, max_length=2048)
    doc_type: str = Field(default="legal_document", min_length=1, max_length=100)
    jurisdiction: str = Field(default="RF", min_length=1, max_length=50)
    version_date: date | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ensure_single_content_source(self) -> "IngestFileRequest":
        if (self.path is None) == (self.text is None):
            raise ValueError("Provide exactly one of 'path' or 'text'.")
        return self


class IngestResponse(BaseModel):
    document_id: str
    chunk_count: int
    title: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=8, ge=1, le=20)


class SourceReference(BaseModel):
    title: str
    source_url: str | None = None
    doc_type: str
    version_date: date | None = None
    chunk_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    confidence: Literal["high", "medium", "low"]
    warnings: list[str] = Field(default_factory=list)
