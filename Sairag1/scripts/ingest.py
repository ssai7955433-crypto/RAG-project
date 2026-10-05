from app.rag import LiteLLMClient, RAGService, SentenceTransformerEncoder, load_corpus
from app.settings import Settings
from app.storage import SQLiteStore


def main() -> None:
    settings = Settings()
    store = SQLiteStore(settings.database_path)
    already_ingested = bool(store.get_chunks())
    service = RAGService(
        settings=settings,
        store=store,
        encoder=SentenceTransformerEncoder(settings.embedding_model),
        llm=LiteLLMClient(settings),
        corpus=load_corpus(settings.corpus_path),
    )
    if already_ingested:
        version = service.ingest(load_corpus(settings.corpus_path))
    else:
        version = service.store.corpus_version()
    print(f"Ingested corpus at version {version}.")


if __name__ == "__main__":
    main()
