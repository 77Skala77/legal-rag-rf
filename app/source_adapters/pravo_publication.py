from typing import Any

from app.source_adapters.base import BaseSourceAdapter, SourceAdapterError


class PravoPublicationAdapter(BaseSourceAdapter):
    """Placeholder for an authorized official-source integration."""

    def __init__(self, publication_identifier: str) -> None:
        self.publication_identifier = publication_identifier

    def fetch(self) -> str:
        raise SourceAdapterError(
            "Official pravo.gov.ru ingestion is not configured. "
            "Connect an authorized API or permitted official export before use."
        )

    def normalize(self, content: str) -> str:
        return content.strip()

    def get_metadata(self) -> dict[str, Any]:
        return {
            "adapter": "pravo_publication",
            "publication_identifier": self.publication_identifier,
            "status": "stub_requires_authorized_integration",
        }
