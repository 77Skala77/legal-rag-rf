from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.answering import (
    LEGAL_WARNINGS,
    AnsweringError,
    confidence_for_sources,
    generate_grounded_answer,
)
from app.config import get_settings
from app.db import database_is_available, get_db
from app.embeddings import EmbeddingError, OpenAIEmbeddingService
from app.ingest import IngestionError, InvalidDocumentError, ingest_document
from app.retrieval import semantic_search
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestFileRequest,
    IngestResponse,
    SourceReference,
)
from app.source_adapters.base import SourceAdapterError


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    database = "connected" if database_is_available(db) else "unavailable"
    return HealthResponse(status="ok" if database == "connected" else "degraded", database=database)


@app.post("/ingest/file", response_model=IngestResponse, status_code=201)
def ingest_file(request: IngestFileRequest, db: Session = Depends(get_db)) -> IngestResponse:
    try:
        return ingest_document(db, request)
    except (SourceAdapterError, InvalidDocumentError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except IngestionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, db: Session = Depends(get_db)) -> AskResponse:
    try:
        embedder = OpenAIEmbeddingService(settings=settings)
        sources = semantic_search(db, request.question, embedder, request.top_k)
        answer = generate_grounded_answer(request.question, sources, settings=settings)
    except (EmbeddingError, AnsweringError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database search failed.") from exc

    source_references = [
        SourceReference(
            title=source.title,
            source_url=source.source_url,
            doc_type=source.doc_type,
            version_date=source.version_date,
            chunk_id=source.chunk_id,
            excerpt=_excerpt(source.text),
        )
        for source in sources
    ]
    warnings = list(LEGAL_WARNINGS)
    if not sources:
        warnings.append("В базе не найдено релевантных фрагментов для ответа.")
    return AskResponse(
        answer=answer,
        sources=source_references,
        confidence=confidence_for_sources(sources),
        warnings=warnings,
    )


def _excerpt(text: str, limit: int = 450) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."
