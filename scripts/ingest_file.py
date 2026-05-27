import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db import SessionLocal  # noqa: E402
from app.embeddings import EmbeddingError  # noqa: E402
from app.ingest import IngestionError, ingest_document  # noqa: E402
from app.schemas import IngestFileRequest  # noqa: E402
from app.source_adapters.base import SourceAdapterError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a local legal text file.")
    parser.add_argument("path", help="Path to a UTF-8 .txt or .md file.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--source-url", default="local-upload")
    parser.add_argument("--doc-type", default="legal_document")
    parser.add_argument("--jurisdiction", default="RF")
    parser.add_argument("--version-date", type=date.fromisoformat)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    request = IngestFileRequest(
        path=args.path,
        title=args.title,
        source_url=args.source_url,
        doc_type=args.doc_type,
        jurisdiction=args.jurisdiction,
        version_date=args.version_date,
    )
    try:
        with SessionLocal() as db:
            result = ingest_document(db, request)
    except (EmbeddingError, IngestionError, SourceAdapterError) as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
