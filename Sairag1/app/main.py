from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import LiteLLMClient, RAGService, SentenceTransformerEncoder, load_corpus
from app.settings import Settings
from app.storage import SQLiteStore

app = FastAPI(
    title="Sairag1: Semantic Cache and Cost-Aware RAG",
    description="A measured RAG demo with retrieval citations, semantic caching, and model routing.",
    version="1.0.0",
)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_store() -> SQLiteStore:
    return SQLiteStore(get_settings().database_path)


@lru_cache
def get_service() -> RAGService:
    settings = get_settings()
    return RAGService(
        settings=settings,
        store=get_store(),
        encoder=SentenceTransformerEncoder(settings.embedding_model),
        llm=LiteLLMClient(settings),
        corpus=load_corpus(settings.corpus_path),
    )


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    strategy: Literal["optimized", "baseline"] = "optimized"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(
    request: AskRequest, service: RAGService = Depends(get_service)
) -> dict[str, object]:
    try:
        return service.answer(request.query, strategy=request.strategy)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/metrics")
def metrics(store: SQLiteStore = Depends(get_store)) -> dict[str, object]:
    logs = store.recent_logs()
    if not logs:
        return {
            "requests": 0,
            "cache_hit_rate": 0.0,
            "cost_per_request_usd": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "route_split": {},
        }
    latencies = sorted(float(row["latency_ms"]) for row in logs)
    hit_count = sum(int(row["cache_hit"]) for row in logs)
    total_cost = sum(float(row["cost_usd"]) for row in logs)

    def percentile(fraction: float) -> float:
        index = max(0, int((len(latencies) - 1) * fraction))
        return round(latencies[index], 2)

    routes: dict[str, int] = {}
    for row in logs:
        routes[row["route"]] = routes.get(row["route"], 0) + 1
    return {
        "requests": len(logs),
        "cache_hit_rate": round(hit_count / len(logs), 4),
        "cost_per_request_usd": round(total_cost / len(logs), 8),
        "p50_latency_ms": percentile(0.50),
        "p95_latency_ms": percentile(0.95),
        "route_split": routes,
    }
