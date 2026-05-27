from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and an optional .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Юридический GPT-помощник по законодательству РФ"
    database_url: str = (
        "postgresql+psycopg://legal_rag:legal_rag_password@localhost:5432/legal_rag"
    )

    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = Field(default=1536, gt=0)
    openai_chat_model: str = "gpt-4.1-mini"

    chunk_size: int = Field(default=1800, ge=200)
    chunk_overlap: int = Field(default=200, ge=0)
    max_top_k: int = Field(default=20, ge=1, le=100)
    max_answer_tokens: int = Field(default=2200, ge=256)


@lru_cache
def get_settings() -> Settings:
    return Settings()
