from abc import ABC, abstractmethod
from typing import Any


class SourceAdapterError(RuntimeError):
    """Raised when an external or local source cannot be read safely."""


class BaseSourceAdapter(ABC):
    @abstractmethod
    def fetch(self) -> str:
        """Retrieve source text without modifying it."""

    @abstractmethod
    def normalize(self, content: str) -> str:
        """Normalize retrieved source content for ingestion."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return adapter-specific provenance metadata."""
