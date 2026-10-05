import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from app.settings import Settings

st.set_page_config(page_title="Sairag1 RAG Metrics", layout="wide")
st.title("Semantic Cache and Cost-Aware RAG")
database_path = Path(Settings().database_path)

if not database_path.exists():
    st.info("No request log exists yet. Start the API and run the evaluation.")
    st.stop()

with sqlite3.connect(database_path) as connection:
    logs = pd.read_sql_query("SELECT * FROM request_logs ORDER BY id", connection)

if logs.empty:
    st.info("No requests have been logged yet.")
    st.stop()

cache_rate = float(logs["cache_hit"].mean())
cost_per_request = float(logs["cost_usd"].mean())
p50 = float(logs["latency_ms"].quantile(0.50))
p95 = float(logs["latency_ms"].quantile(0.95))
cards = st.columns(4)
cards[0].metric("Requests", len(logs))
cards[1].metric("Cache hit rate", f"{cache_rate:.1%}")
cards[2].metric("Mean cost / request", f"${cost_per_request:.6f}")
cards[3].metric("p50 / p95 latency", f"{p50:.0f} / {p95:.0f} ms")

st.subheader("Requests by route")
st.bar_chart(logs.groupby("route").size())
st.subheader("Cost and latency over time")
logs["created_at"] = pd.to_datetime(logs["created_at"], unit="s")
chart_data = logs.set_index("created_at")[["cost_usd", "latency_ms"]]
st.line_chart(chart_data)

evaluation_path = Path("eval_results.csv")
if evaluation_path.exists():
    evaluation = pd.read_csv(evaluation_path)
    if not evaluation.empty:
        st.subheader("Eval answer-overlap score by route")
        evaluation["evaluated_at"] = pd.to_datetime(evaluation["evaluated_at"])
        quality = evaluation.pivot_table(
            index="evaluated_at",
            columns=["strategy", "route"],
            values="answer_overlap",
            aggfunc="mean",
        )
        st.line_chart(quality)
        st.caption(
            "Lexical overlap is a lightweight proxy, not a human or LLM quality judgment."
        )
else:
    st.info("Run scripts/evaluate.py to show answer-overlap scores by route.")

st.subheader("Recent request log")
st.dataframe(logs.tail(100).sort_values("id", ascending=False), use_container_width=True)
