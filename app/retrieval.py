import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings import EmbeddingProvider
from app.models import Chunk, Document


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    title: str
    source_url: str | None
    doc_type: str
    version_date: date | None
    article_number: str | None
    section_title: str | None
    similarity: float


@dataclass(frozen=True)
class VectorCandidate:
    identifier: str
    text: str
    embedding: list[float]


def semantic_search(
    db: Session, question: str, embedder: EmbeddingProvider, top_k: int
) -> list[RetrievedChunk]:
    question_vector = embedder.embed_query(question)
    distance = Chunk.embedding.cosine_distance(question_vector)
    query = (
        select(Chunk, Document, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)
        .limit(top_k)
    )
    results: list[RetrievedChunk] = []
    for chunk, document, cosine_distance in db.execute(query):
        similarity = max(-1.0, min(1.0, 1.0 - float(cosine_distance)))
        results.append(
            RetrievedChunk(
                chunk_id=str(chunk.id),
                text=chunk.text,
                title=document.title,
                source_url=document.source_url,
                doc_type=document.doc_type,
                version_date=document.version_date,
                article_number=chunk.article_number,
                section_title=chunk.section_title,
                similarity=similarity,
            )
        )
    return results


def rank_candidates(
    question: str,
    candidates: Sequence[VectorCandidate],
    embedder: EmbeddingProvider,
    top_k: int,
) -> list[tuple[VectorCandidate, float]]:
    """In-memory cosine ranking for deterministic tests and small diagnostics."""
    query_vector = embedder.embed_query(question)
    ranked = [
        (candidate, cosine_similarity(query_vector, candidate.embedding))
        for candidate in candidates
    ]
    return sorted(ranked, key=lambda item: item[1], reverse=True)[:top_k]


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Vectors must be non-empty and have equal dimensions.")
    dot_product = sum(a * b for a, b in zip(first, second, strict=True))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return dot_product / (first_norm * second_norm)
