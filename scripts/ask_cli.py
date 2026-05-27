import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.answering import (  # noqa: E402
    LEGAL_WARNINGS,
    AnsweringError,
    confidence_for_sources,
    generate_grounded_answer,
)
from app.config import get_settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.embeddings import EmbeddingError, OpenAIEmbeddingService  # noqa: E402
from app.retrieval import semantic_search  # noqa: E402
from app.schemas import AskResponse, SourceReference  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a grounded legal question.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = get_settings()
    if args.top_k < 1 or args.top_k > settings.max_top_k:
        print(f"--top-k must be between 1 and {settings.max_top_k}.", file=sys.stderr)
        return 2
    try:
        embedder = OpenAIEmbeddingService(settings=settings)
        with SessionLocal() as db:
            sources = semantic_search(db, args.question, embedder, args.top_k)
        answer = generate_grounded_answer(args.question, sources, settings=settings)
    except (EmbeddingError, AnsweringError) as exc:
        print(f"Question failed: {exc}", file=sys.stderr)
        return 1

    warnings = list(LEGAL_WARNINGS)
    if not sources:
        warnings.append("В базе не найдено релевантных фрагментов для ответа.")
    response = AskResponse(
        answer=answer,
        sources=[
            SourceReference(
                title=source.title,
                source_url=source.source_url,
                doc_type=source.doc_type,
                version_date=source.version_date,
                chunk_id=source.chunk_id,
                excerpt=_excerpt(source.text),
            )
            for source in sources
        ],
        confidence=confidence_for_sources(sources),
        warnings=warnings,
    )
    print(response.model_dump_json(indent=2))
    return 0


def _excerpt(text: str, limit: int = 450) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


if __name__ == "__main__":
    raise SystemExit(main())
