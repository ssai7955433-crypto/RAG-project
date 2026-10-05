from pathlib import Path

from app.rag import RAGService, chunk_text, normalize_query
from app.settings import Settings
from app.storage import SQLiteStore


class FakeEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 16
            for character in text.lower():
                vector[ord(character) % len(vector)] += 1.0
            vectors.append(vector)
        return vectors


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, prompt: str, model: str) -> tuple[str, int, int]:
        self.calls.append(model)
        source = prompt.split("Source [", 1)[1].split("]", 1)[0]
        return f"Employees receive 20 leave days [{source}].", 12, 8


class EscalatingLLM(FakeLLM):
    def complete(self, prompt: str, model: str) -> tuple[str, int, int]:
        self.calls.append(model)
        if len(self.calls) == 1:
            return "I don't know.", 12, 8
        return "Employees receive 20 leave days [handbook].", 12, 8


def make_service(tmp_path: Path) -> tuple[RAGService, FakeLLM]:
    settings = Settings(
        database_path=str(tmp_path / "test.sqlite3"),
        corpus_path="unused.json",
        cache_threshold=0.99,
        cache_ttl_seconds=3600,
        small_model="small",
        large_model="large",
        small_input_usd_per_1m=1.0,
        small_output_usd_per_1m=2.0,
        large_input_usd_per_1m=10.0,
        large_output_usd_per_1m=20.0,
    )
    llm = FakeLLM()
    corpus = [
        {
            "id": "handbook",
            "text": "Full-time employees receive 20 paid leave days every year.",
        }
    ]
    return (
        RAGService(
            settings=settings,
            store=SQLiteStore(settings.database_path),
            encoder=FakeEncoder(),
            llm=llm,
            corpus=corpus,
        ),
        llm,
    )


def test_query_normalization_and_chunk_overlap() -> None:
    assert normalize_query("  Password RESET, please! ") == "password reset please"
    pieces = chunk_text(" ".join(f"w{i}" for i in range(12)), max_words=5, overlap_words=2)
    assert pieces == ["w0 w1 w2 w3 w4", "w3 w4 w5 w6 w7", "w6 w7 w8 w9 w10", "w9 w10 w11"]


def test_semantic_cache_hit_skips_model_and_logs_cost(tmp_path: Path) -> None:
    service, llm = make_service(tmp_path)
    first = service.answer("How many paid leave days?", strategy="optimized")
    second = service.answer("How many paid leave days?", strategy="optimized")

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert len(llm.calls) == 1
    assert second["sources"][0]["id"] == "handbook"
    logs = service.store.recent_logs()
    assert logs[0]["cache_hit"] == 1
    assert logs[0]["cost_usd"] == 0.0


def test_baseline_always_uses_large_model_without_cache(tmp_path: Path) -> None:
    service, llm = make_service(tmp_path)
    first = service.answer("How many paid leave days?", strategy="baseline")
    second = service.answer("How many paid leave days?", strategy="baseline")

    assert first["route"] == "large"
    assert second["cache_hit"] is False
    assert llm.calls == ["large", "large"]
    assert first["cost_usd"] > 0


def test_complex_question_uses_large_model(tmp_path: Path) -> None:
    service, llm = make_service(tmp_path)

    result = service.answer(
        "Explain how employees receive paid leave benefits across departments",
        strategy="optimized",
    )

    assert result["route"] == "complex"
    assert result["model"] == "large"
    assert llm.calls == ["large"]


def test_weak_small_model_answer_escalates_and_counts_both_costs(
    tmp_path: Path,
) -> None:
    service, _ = make_service(tmp_path)
    llm = EscalatingLLM()
    service.llm = llm

    result = service.answer("How many paid leave days?", strategy="optimized")

    assert result["route"] == "escalated"
    assert result["model"] == "small,large"
    assert llm.calls == ["small", "large"]
    assert result["cost_usd"] == 0.000308


def test_reingestion_invalidates_old_cache(tmp_path: Path) -> None:
    service, llm = make_service(tmp_path)
    service.answer("How many paid leave days?", strategy="optimized")
    old_version = service.store.corpus_version()

    new_version = service.ingest(
        [{"id": "new-handbook", "text": "Employees receive 25 paid leave days."}]
    )
    result = service.answer("How many paid leave days?", strategy="optimized")

    assert new_version == old_version + 1
    assert result["cache_hit"] is False
    assert len(llm.calls) == 2
