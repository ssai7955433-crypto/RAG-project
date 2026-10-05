import json
import re
import string
import time
from pathlib import Path
from typing import Any, Protocol

from app.settings import Settings
from app.storage import SQLiteStore, cosine_similarity


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class CompletionClient(Protocol):
    def complete(self, prompt: str, model: str) -> tuple[str, int, int]: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


class LiteLLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def complete(self, prompt: str, model: str) -> tuple[str, int, int]:
        from litellm import completion

        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_base=self.settings.ollama_api_base if model.startswith("ollama/") else None,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Model {model} returned an empty answer")
        usage = response.usage
        if usage and usage.prompt_tokens is not None and usage.completion_tokens is not None:
            input_tokens, output_tokens = usage.prompt_tokens, usage.completion_tokens
        else:
            from litellm import token_counter

            input_tokens = token_counter(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            output_tokens = token_counter(model=model, text=content)
        return content.strip(), int(input_tokens), int(output_tokens)


def normalize_query(query: str) -> str:
    lowered = query.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", lowered).strip()


def chunk_text(text: str, max_words: int = 350, overlap_words: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


class RAGService:
    def __init__(
        self,
        settings: Settings,
        store: SQLiteStore,
        encoder: Encoder,
        llm: CompletionClient,
        corpus: list[dict[str, str]],
    ) -> None:
        self.settings = settings
        self.store = store
        self.encoder = encoder
        self.llm = llm
        self._ensure_corpus(corpus)

    def _ensure_corpus(self, corpus: list[dict[str, str]]) -> None:
        if self.store.get_chunks():
            return
        chunks: list[dict[str, Any]] = []
        texts_to_embed: list[str] = []
        for document in corpus:
            pieces = chunk_text(document["text"])
            for index, piece in enumerate(pieces):
                chunks.append(
                    {
                        "id": f'{document["id"]}-{index + 1}',
                        "source": document["id"],
                        "text": piece,
                    }
                )
                texts_to_embed.append(piece)
        if not chunks:
            raise ValueError("Corpus has no text to index")
        vectors = self.encoder.encode(texts_to_embed)
        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector
        self.store.replace_chunks(chunks)

    def ingest(self, corpus: list[dict[str, str]]) -> int:
        chunks: list[dict[str, Any]] = []
        texts_to_embed: list[str] = []
        for document in corpus:
            for index, piece in enumerate(chunk_text(document["text"])):
                chunks.append(
                    {
                        "id": f'{document["id"]}-{index + 1}',
                        "source": document["id"],
                        "text": piece,
                    }
                )
                texts_to_embed.append(piece)
        if not chunks:
            raise ValueError("Corpus has no text to index")
        vectors = self.encoder.encode(texts_to_embed)
        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector
        return self.store.replace_chunks(chunks)

    def answer(self, query: str, strategy: str = "optimized") -> dict[str, Any]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if strategy not in {"optimized", "baseline"}:
            raise ValueError("strategy must be 'optimized' or 'baseline'")
        started = time.perf_counter()
        normalized = normalize_query(query)
        query_vector = self.encoder.encode([normalized])[0]
        if strategy == "optimized":
            hit = self.store.find_cache_hit(
                query_vector, self.settings.cache_threshold
            )
            if hit:
                elapsed_ms = (time.perf_counter() - started) * 1000
                self.store.log_request(
                    {
                        "query": query,
                        "model": "cache",
                        "route": "cache",
                        "strategy": strategy,
                        "cache_hit": True,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "latency_ms": elapsed_ms,
                    }
                )
                return {
                    "answer": hit["answer"],
                    "sources": hit["sources"],
                    "cache_hit": True,
                    "route": "cache",
                    "model": "cache",
                    "latency_ms": round(elapsed_ms, 2),
                    "cost_usd": 0.0,
                }

        retrieved = self._retrieve(query_vector)
        route = "large" if strategy == "baseline" else self._classify(query, retrieved)
        model = (
            self.settings.large_model
            if route in {"large", "complex"}
            else self.settings.small_model
        )
        answer, input_tokens, output_tokens = self._generate(query, retrieved, model)
        total_input, total_output = input_tokens, output_tokens
        models_used = [model]
        cost = self._cost_attempt(model, input_tokens, output_tokens)
        if strategy == "optimized" and route == "simple" and self._needs_escalation(
            answer, retrieved
        ):
            model = self.settings.large_model
            answer, input_tokens, output_tokens = self._generate(query, retrieved, model)
            total_input += input_tokens
            total_output += output_tokens
            models_used.append(model)
            cost += self._cost_attempt(model, input_tokens, output_tokens)
            route = "escalated"

        elapsed_ms = (time.perf_counter() - started) * 1000
        sources = [
            {
                "id": item["source"],
                "chunk_id": item["id"],
                "score": round(item["score"], 4),
                "text": item["text"],
            }
            for item in retrieved
        ]
        if strategy == "optimized" and answer.strip() and sources:
            self.store.save_cache(
                normalized,
                query_vector,
                answer,
                sources,
                self.settings.cache_ttl_seconds,
            )
        self.store.log_request(
            {
                "query": query,
                "model": ",".join(models_used),
                "route": route,
                "strategy": strategy,
                "cache_hit": False,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cost_usd": cost,
                "latency_ms": elapsed_ms,
            }
        )
        return {
            "answer": answer,
            "sources": sources,
            "cache_hit": False,
            "route": route,
            "model": ",".join(models_used),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost_usd": round(cost, 8),
            "latency_ms": round(elapsed_ms, 2),
        }

    def _retrieve(self, query_vector: list[float]) -> list[dict[str, Any]]:
        candidates = self.store.get_chunks()
        for item in candidates:
            item["score"] = cosine_similarity(query_vector, item["embedding"])
        return sorted(candidates, key=lambda item: item["score"], reverse=True)[:5]

    @staticmethod
    def _classify(query: str, retrieved: list[dict[str, Any]]) -> str:
        complex_markers = (
            "compare",
            "why",
            "explain",
            "difference",
            "across",
            "relationship",
            "multiple",
        )
        words = query.lower().split()
        if any(marker in query.lower() for marker in complex_markers) or len(words) > 22:
            return "complex"
        if len(retrieved) >= 2:
            score_gap = retrieved[0]["score"] - retrieved[-1]["score"]
            if score_gap < 0.04:
                return "complex"
        return "simple"

    @staticmethod
    def _needs_escalation(answer: str, retrieved: list[dict[str, Any]]) -> bool:
        normalized = answer.lower()
        uncertainty = ("i don't know", "not enough information", "cannot answer")
        weak = any(phrase in normalized for phrase in uncertainty)
        missing_citation = not re.search(r"\[[\w.-]+\]", answer)
        low_retrieval = not retrieved or retrieved[0]["score"] < 0.25
        return weak or missing_citation or low_retrieval

    def _generate(
        self, query: str, retrieved: list[dict[str, Any]], model: str
    ) -> tuple[str, int, int]:
        context = "\n\n".join(
            f'Source [{item["source"]}]: {item["text"]}' for item in retrieved
        )
        prompt = (
            "Answer the question using only the source passages below. "
            "Cite factual claims with source IDs in square brackets. "
            "If the passages do not answer the question, say so.\n\n"
            f"Passages:\n{context}\n\nQuestion: {query}"
        )
        return self.llm.complete(prompt, model)

    def _cost_attempt(self, model: str, input_tokens: int, output_tokens: int) -> float:
        input_rate, output_rate = self._rates(model)
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000

    def _rates(self, model: str) -> tuple[float, float]:
        if model == self.settings.small_model:
            return (
                self.settings.small_input_usd_per_1m,
                self.settings.small_output_usd_per_1m,
            )
        return (
            self.settings.large_input_usd_per_1m,
            self.settings.large_output_usd_per_1m,
        )


def load_corpus(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as corpus_file:
        documents = json.load(corpus_file)
    if not isinstance(documents, list) or any(
        not isinstance(document, dict)
        or not isinstance(document.get("id"), str)
        or not isinstance(document.get("text"), str)
        for document in documents
    ):
        raise ValueError("Corpus JSON must be a list of documents with id and text")
    return documents
