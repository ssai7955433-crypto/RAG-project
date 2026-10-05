# Sairag1: Semantic Cache and Cost-Aware RAG

A standalone RAG project based on the first optimization project in the brief. It includes a baseline mode, semantic answer caching, rule-based small/large-model routing, escalation, SQLite request logging, an evaluation replay script, and a Streamlit chat app with metrics.

The included fictional handbook corpus and six evaluation questions are a runnable demo, not a statistically meaningful benchmark. Expand them to 50–200 real documents and 50–100 manually checked questions before making performance claims.

## Stack

- FastAPI for the `/ask` API
- Sentence Transformers (`all-MiniLM-L6-v2`) for local query and document embeddings
- LiteLLM for model calls; local runs can use Ollama, while the hosted Streamlit app uses Groq
- SQLite for versioned cache entries, indexed chunks, and request logs
- Streamlit for the metrics dashboard

## Setup

From this folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2:3b
ollama pull qwen3:8b
```

Edit `.env` if your model names, Ollama endpoint, corpus location, or model prices differ. The price defaults are zero for local models; configure per-million input/output rates for paid hosted models. Costs in the logs are estimates using these configured rates.

## Run

```powershell
uvicorn app.main:app --reload
```

The first request downloads/loads the embedding model and indexes the bundled corpus into SQLite. Then:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"How many annual leave days do employees get?"}'
```

Use `"strategy":"baseline"` to always route to the large model and bypass cache reads/writes. The default `"optimized"` strategy uses semantic cache, routing, and escalation. `GET /health` checks API availability; `GET /metrics` reports recent aggregate metrics.

## Corpus updates

Edit `data/corpus.json`, keeping each document in `{ "id": "...", "text": "..." }` form. Re-ingest to replace the indexed chunks and increment the corpus version:

```powershell
python -m scripts.ingest
```

Old cache entries are ignored after a corpus-version change. Cache entries also expire after `CACHE_TTL_SECONDS`. This demo corpus is shared and non-personal; do not use the global cache for answers that depend on user identity or permissions.

## Evaluation

Start the API in one terminal, then replay both strategies with repeat traffic:

```powershell
python -m scripts.evaluate --repeats 3 --output eval_results.csv
```

The CSV contains one row per request, with answer-token overlap, expected-source retrieval hit, cache hit, route, cost, and latency. Answer overlap is only a lexical proxy; inspect answers manually or add an LLM judge before treating it as correctness. Compare baseline versus optimized on the same query/repeat sequence. The default six-row set demonstrates the workflow only; manually check a larger eval set before reporting resume metrics.

## Streamlit app

```powershell
streamlit run dashboard.py
```

The app answers questions from the bundled handbook and includes an aggregate metrics tab. The shared `.env.example` defaults to Ollama for the FastAPI workflow. For local Streamlit with Groq, set `GROQ_API_KEY` and change `SMALL_MODEL` and `LARGE_MODEL` to the Groq IDs in `.streamlit/secrets.toml.example`; alternatively keep the Ollama model IDs and run Ollama locally.

### Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with the GitHub account that can access `ssai7955433-crypto/RAG-project`.
2. Choose **Create app** and select repository `ssai7955433-crypto/RAG-project`, branch `main`, and main file path `Sairag1/dashboard.py`.
3. In the app's **Advanced settings / Secrets**, add a Groq API key and the model IDs. Use `.streamlit/secrets.toml.example` as a template, but replace the placeholder with your own key. Never commit the real key.
4. Deploy. The first startup downloads the embedding model and may take a few minutes. Streamlit Cloud will then show the public app URL.

Get a Groq API key from [Groq Console](https://console.groq.com/keys). The configured Groq models are `openai/gpt-oss-20b` and `openai/gpt-oss-120b`. Groq announced retirement of the previous Llama model IDs for free/developer accounts in June 2026; if your app has old `SMALL_MODEL` or `LARGE_MODEL` values in Streamlit Secrets, replace them with the IDs in `.streamlit/secrets.toml.example` and reboot the app. Streamlit Secrets override code defaults. The application uses local sentence-transformer embeddings, so only LLM calls require the Groq key. The SQLite cache and request log live on the app filesystem and may reset when the cloud app restarts; use this as a demo, not durable production storage. The shared cache is only appropriate for the bundled non-personal corpus.

For local Streamlit, Groq settings can be set in `.env`; `.streamlit/secrets.toml.example` shows the cloud Secrets format. Cost estimates are zero until current provider prices are configured per million tokens.

## Cache threshold tuning

Use a labeled file of paraphrase pairs and near-miss pairs (including changed regions, dates, products, and entities). Sweep the bundled example with:

```powershell
python -m scripts.sweep_cache_threshold
```

Add enough manually labeled pairs to represent your corpus before selecting the lowest threshold with zero false hits. The six bundled pairs are only a smoke test and do not validate the default threshold.

## Tests

```powershell
pytest
```

The tests use fake encoders and model responses, so they do not need a running Ollama server or a model download.
