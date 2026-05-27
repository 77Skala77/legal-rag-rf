from pathlib import Path
from typing import Any

from app.source_adapters.base import BaseSourceAdapter, SourceAdapterError


class LocalFileAdapter(BaseSourceAdapter):
    SUPPORTED_SUFFIXES = {".txt", ".md"}

    def __init__(self, path: str) -> None:
        self.path = Path(path).expanduser()

    def fetch(self) -> str:
        if self.path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise SourceAdapterError("Only local .txt and .md files are supported.")
        if not self.path.is_file():
            raise SourceAdapterError(f"Local source file does not exist: {self.path}")
        try:
            return self.path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SourceAdapterError("Local source file must be UTF-8 text.") from exc
        except OSError as exc:
            raise SourceAdapterError(f"Unable to read local source file: {exc}") from exc

    def normalize(self, content: str) -> str:
        return content.replace("\r\n", "\n").replace("\r", "\n").strip()

    def get_metadata(self) -> dict[str, Any]:
        return {"adapter": "local_file", "local_path": str(self.path.resolve())}
