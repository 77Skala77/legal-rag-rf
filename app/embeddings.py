from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI

from app.config import Settings, get_settings


class EmbeddingError(RuntimeError):
    """Raised when embedding generation cannot be completed."""


class EmbeddingProvider(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbeddingService:
    def __init__(
        self, settings: Settings | None = None, client: OpenAI | None = None
    ) -> None:
        self.settings = settings or get_settings()
        if client is None:
            if not self.settings.openai_api_key:
                raise EmbeddingError("OPENAI_API_KEY is not configured.")
            client = OpenAI(api_key=self.settings.openai_api_key)
        self.client = client

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_texts([text])
        return embeddings[0]

    def embed_texts(self, texts: Sequence[str], batch_size: int = 100) -> list[list[float]]:
        clean_texts = [text.strip() for text in texts]
        if not clean_texts or any(not text for text in clean_texts):
            raise EmbeddingError("Embedding input must contain non-empty text.")

        vectors: list[list[float]] = []
        try:
            for start in range(0, len(clean_texts), batch_size):
                batch = clean_texts[start : start + batch_size]
                response = self.client.embeddings.create(
                    model=self.settings.openai_embedding_model,
                    input=batch,
                    dimensions=self.settings.openai_embedding_dimensions,
                    encoding_format="float",
                )
                vectors.extend(item.embedding for item in response.data)
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embeddings request failed: {exc}") from exc

        if len(vectors) != len(clean_texts):
            raise EmbeddingError("OpenAI returned an unexpected number of embeddings.")
        if any(
            len(vector) != self.settings.openai_embedding_dimensions for vector in vectors
        ):
            raise EmbeddingError(
                "Embedding dimension does not match OPENAI_EMBEDDING_DIMENSIONS."
            )
        return vectors
