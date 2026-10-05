import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class SQLiteStore:
    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO metadata (key, value)
                    VALUES ('corpus_version', '0');
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    corpus_version INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS cache_version_expiry
                    ON semantic_cache (corpus_version, expires_at);
                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    query TEXT NOT NULL,
                    model TEXT NOT NULL,
                    route TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    latency_ms REAL NOT NULL
                );
                """
            )

    def get_chunks(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM chunks ORDER BY id").fetchall()
        return [
            {
                "id": row["id"],
                "source": row["source"],
                "text": row["text"],
                "embedding": json.loads(row["embedding"]),
            }
            for row in rows
        ]

    def replace_chunks(self, chunks: list[dict[str, Any]]) -> int:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.executemany(
                "INSERT INTO chunks (id, source, text, embedding) VALUES (?, ?, ?, ?)",
                [
                    (
                        chunk["id"],
                        chunk["source"],
                        chunk["text"],
                        json.dumps(chunk["embedding"]),
                    )
                    for chunk in chunks
                ],
            )
            connection.execute(
                "UPDATE metadata SET value = CAST(value AS INTEGER) + 1 "
                "WHERE key = 'corpus_version'"
            )
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'corpus_version'"
            ).fetchone()
        return int(row["value"])

    def corpus_version(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'corpus_version'"
            ).fetchone()
        return int(row["value"])

    def find_cache_hit(
        self, embedding: list[float], threshold: float, now: float | None = None
    ) -> dict[str, Any] | None:
        current_time = now if now is not None else time.time()
        version = self.corpus_version()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT query, embedding, answer, sources
                FROM semantic_cache
                WHERE corpus_version = ? AND expires_at > ?
                """,
                (version, current_time),
            ).fetchall()
        best: tuple[float, sqlite3.Row] | None = None
        for row in rows:
            score = cosine_similarity(embedding, json.loads(row["embedding"]))
            if best is None or score > best[0]:
                best = (score, row)
        if best is None or best[0] < threshold:
            return None
        return {
            "score": best[0],
            "query": best[1]["query"],
            "answer": best[1]["answer"],
            "sources": json.loads(best[1]["sources"]),
        }

    def save_cache(
        self,
        query: str,
        embedding: list[float],
        answer: str,
        sources: list[dict[str, Any]],
        ttl_seconds: int,
    ) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO semantic_cache
                    (query, embedding, answer, sources, corpus_version,
                     created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query,
                    json.dumps(embedding),
                    answer,
                    json.dumps(sources),
                    self.corpus_version(),
                    now,
                    now + ttl_seconds,
                ),
            )

    def log_request(self, entry: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO request_logs
                    (created_at, query, model, route, strategy, cache_hit,
                     input_tokens, output_tokens, cost_usd, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    entry["query"],
                    entry["model"],
                    entry["route"],
                    entry["strategy"],
                    int(entry["cache_hit"]),
                    entry["input_tokens"],
                    entry["output_tokens"],
                    entry["cost_usd"],
                    entry["latency_ms"],
                ),
            )

    def recent_logs(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM request_logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
