import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sairag1 RAG", page_icon="🔎", layout="wide")
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

for key in (
    "GROQ_API_KEY",
    "SMALL_MODEL",
    "LARGE_MODEL",
    "CACHE_THRESHOLD",
    "CACHE_TTL_SECONDS",
    "SMALL_INPUT_USD_PER_1M",
    "SMALL_OUTPUT_USD_PER_1M",
    "LARGE_INPUT_USD_PER_1M",
    "LARGE_OUTPUT_USD_PER_1M",
):
    if key in st.secrets:
        os.environ[key] = str(st.secrets[key])

os.environ.setdefault("SMALL_MODEL", "groq/openai/gpt-oss-20b")
os.environ.setdefault("LARGE_MODEL", "groq/openai/gpt-oss-120b")
os.environ.setdefault("CORPUS_PATH", str(PROJECT_DIR / "data" / "corpus.json"))
os.environ.setdefault("DATABASE_PATH", str(PROJECT_DIR / "data" / "rag.sqlite3"))

from app.rag import LiteLLMClient, RAGService, SentenceTransformerEncoder, load_corpus
from app.settings import Settings
from app.storage import SQLiteStore


settings = Settings()
if (
    any(
        model.startswith("groq/")
        for model in (settings.small_model, settings.large_model)
    )
    and not os.getenv("GROQ_API_KEY")
):
    st.error(
        "Add GROQ_API_KEY to Streamlit Cloud → App settings → Secrets (or to local "
        ".env) before asking questions. Do not put API keys in source code."
    )
    st.stop()


@st.cache_resource
def get_service() -> RAGService:
    return RAGService(
        settings=settings,
        store=SQLiteStore(settings.database_path),
        encoder=SentenceTransformerEncoder(settings.embedding_model),
        llm=LiteLLMClient(settings),
        corpus=load_corpus(settings.corpus_path),
    )


def render_metrics(logs: pd.DataFrame) -> None:
    if logs.empty:
        st.info("Ask a question to start collecting request metrics.")
        return

    cards = st.columns(4)
    cards[0].metric("Requests", len(logs))
    cards[1].metric("Semantic-cache hit rate", f"{logs['cache_hit'].mean():.1%}")
    cards[2].metric("Mean configured cost", f"${logs['cost_usd'].mean():.6f}")
    cards[3].metric(
        "p50 / p95 latency",
        f"{logs['latency_ms'].quantile(0.50):.0f} / "
        f"{logs['latency_ms'].quantile(0.95):.0f} ms",
    )
    if not logs["cost_usd"].any():
        st.caption(
            "Cost is shown as $0 until you add the provider's current per-million-token "
            "rates to Streamlit Secrets."
        )
    st.subheader("Requests by route")
    st.bar_chart(logs.groupby("route").size())
    st.subheader("Cost and latency over time")
    chart_logs = logs.copy()
    chart_logs["created_at"] = pd.to_datetime(chart_logs["created_at"], unit="s")
    st.line_chart(
        chart_logs.set_index("created_at")[["cost_usd", "latency_ms"]]
    )


st.title("Sairag1: Semantic Cache and Cost-Aware RAG")
st.caption(
    "Ask questions about the sample handbook. Answers use retrieved passages and "
    "include source citations."
)

service = get_service()
with st.sidebar:
    st.header("Request settings")
    strategy_label = st.radio(
        "Answer strategy",
        ("Optimized (cache + routing)", "Baseline (large model)"),
    )
    strategy = "optimized" if strategy_label.startswith("Optimized") else "baseline"
    st.caption(
        "The cache is shared across users and should only be used with this "
        "non-personal demo corpus."
    )
    st.divider()
    st.caption(f"Indexed corpus version: {service.store.corpus_version()}")
    st.caption("Embeddings run locally in the Streamlit app; model answers use Groq.")

chat_tab, metrics_tab = st.tabs(["Ask the handbook", "Metrics"])

with chat_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("result"):
                result = message["result"]
                st.caption(
                    f"Route: {result['route']} · "
                    f"Cache hit: {'yes' if result['cache_hit'] else 'no'} · "
                    f"Latency: {result['latency_ms']:.0f} ms"
                )
                for source in result["sources"]:
                    with st.expander(
                        f"[{source['id']}] · relevance {source['score']:.2f}"
                    ):
                        st.write(source["text"])

    question = st.chat_input("Ask a question about the handbook")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving passages and generating an answer..."):
                answer: dict[str, Any] = service.answer(question, strategy=strategy)
            st.markdown(answer["answer"])
            st.caption(
                f"Route: {answer['route']} · "
                f"Cache hit: {'yes' if answer['cache_hit'] else 'no'} · "
                f"Latency: {answer['latency_ms']:.0f} ms"
            )
            for source in answer["sources"]:
                with st.expander(
                    f"[{source['id']}] · relevance {source['score']:.2f}"
                ):
                    st.write(source["text"])
        st.session_state.messages.append(
            {"role": "assistant", "content": answer["answer"], "result": answer}
        )

with metrics_tab:
    logs_path = Path(settings.database_path)
    if logs_path.exists():
        with sqlite3.connect(logs_path) as connection:
            request_logs = pd.read_sql_query(
                "SELECT * FROM request_logs ORDER BY id", connection
            )
    else:
        request_logs = pd.DataFrame()
    render_metrics(request_logs)
    results_path = PROJECT_DIR / "eval_results.csv"
    if results_path.exists():
        evaluation = pd.read_csv(results_path)
        if not evaluation.empty:
            st.subheader("Evaluation answer overlap by route")
            evaluation["evaluated_at"] = pd.to_datetime(evaluation["evaluated_at"])
            quality = evaluation.pivot_table(
                index="evaluated_at",
                columns=["strategy", "route"],
                values="answer_overlap",
                aggfunc="mean",
            )
            st.line_chart(quality)
            st.caption(
                "Lexical overlap is a quick proxy, not a human or LLM quality judgment."
            )
