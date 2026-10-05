import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_path: str = os.getenv("DATABASE_PATH", "data/rag.sqlite3")
    corpus_path: str = os.getenv("CORPUS_PATH", "data/corpus.json")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    small_model: str = os.getenv("SMALL_MODEL", "ollama/llama3.2:3b")
    large_model: str = os.getenv("LARGE_MODEL", "ollama/qwen3:8b")
    cache_threshold: float = float(os.getenv("CACHE_THRESHOLD", "0.92"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
    small_input_usd_per_1m: float = float(os.getenv("SMALL_INPUT_USD_PER_1M", "0"))
    small_output_usd_per_1m: float = float(os.getenv("SMALL_OUTPUT_USD_PER_1M", "0"))
    large_input_usd_per_1m: float = float(os.getenv("LARGE_INPUT_USD_PER_1M", "0"))
    large_output_usd_per_1m: float = float(os.getenv("LARGE_OUTPUT_USD_PER_1M", "0"))
    ollama_api_base: str = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    def __post_init__(self) -> None:
        if not 0.0 <= self.cache_threshold <= 1.0:
            raise ValueError("CACHE_THRESHOLD must be between 0 and 1")
        if self.cache_ttl_seconds <= 0:
            raise ValueError("CACHE_TTL_SECONDS must be positive")
