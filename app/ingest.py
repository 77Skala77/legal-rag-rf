from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.chunking import LegalChunk, chunk_legal_text
from app.config import Settings, get_settings
from app.embeddings import OpenAIEmbeddingService
from app.models import Chunk, Document
from app.schemas import IngestFileRequest, IngestResponse
from app.source_adapters.local_file import LocalFileAdapter


class IngestionError(RuntimeError):
    """Raised when a document cannot be persisted after preparation."""


class InvalidDocumentError(IngestionError):
    """Raised when supplied text has no ingestible content."""


def ingest_document(
    db: Session,
    request: IngestFileRequest,
    embedder: OpenAIEmbeddingService | None = None,
    settings: Settings | None = None,
) -> IngestResponse:
    app_settings = settings or get_settings()
    text, source_metadata = _resolve_source_text(request)
    chunks = chunk_legal_text(
        text, max_chars=app_settings.chunk_size, overlap=app_settings.chunk_overlap
    )
    if not chunks:
        raise InvalidDocumentError("Document is empty after normalization.")

    embedding_service = embedder or OpenAIEmbeddingService(settings=app_settings)
    embeddings = embedding_service.embed_texts([chunk.text for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise IngestionError("Each chunk must receive one embedding.")

    document = Document(
        title=request.title,
        source_url=request.source_url,
        doc_type=request.doc_type,
        jurisdiction=request.jurisdiction,
        version_date=request.version_date,
        metadata_json={**request.metadata, **source_metadata},
    )
    try:
        db.add(document)
        db.flush()
        db.add_all(
            [
                _build_db_chunk(document, chunk, embedding)
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]
        )
        db.commit()
        db.refresh(document)
    except SQLAlchemyError as exc:
        db.rollback()
        raise IngestionError(f"Unable to persist document chunks: {exc}") from exc

    return IngestResponse(
        document_id=str(document.id), chunk_count=len(chunks), title=document.title
    )


def _resolve_source_text(request: IngestFileRequest) -> tuple[str, dict[str, str]]:
    if request.path:
        adapter = LocalFileAdapter(request.path)
        text = adapter.normalize(adapter.fetch())
        return text, adapter.get_metadata()
    return (request.text or "").strip(), {"adapter": "inline_text"}


def _build_db_chunk(
    document: Document, chunk: LegalChunk, embedding: list[float]
) -> Chunk:
    return Chunk(
        document_id=document.id,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        embedding=embedding,
        article_number=chunk.article_number,
        section_title=chunk.section_title,
    )
