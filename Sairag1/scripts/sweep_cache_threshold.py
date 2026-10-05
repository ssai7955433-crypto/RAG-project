import argparse
import csv
from pathlib import Path

from app.rag import SentenceTransformerEncoder, normalize_query
from app.settings import Settings
from app.storage import cosine_similarity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep semantic-cache thresholds over labeled query pairs."
    )
    parser.add_argument("--pairs", type=Path, default=Path("data/cache_pairs.csv"))
    args = parser.parse_args()
    with args.pairs.open(newline="", encoding="utf-8") as file:
        pairs = list(csv.DictReader(file))
    if not pairs:
        raise ValueError(f"No query pairs found in {args.pairs}")

    encoder = SentenceTransformerEncoder(Settings().embedding_model)
    normalized_texts = [
        normalize_query(text)
        for pair in pairs
        for text in (pair["query_a"], pair["query_b"])
    ]
    vectors = encoder.encode(normalized_texts)
    scored_pairs = [
        (
            cosine_similarity(vectors[index * 2], vectors[index * 2 + 1]),
            pair["should_match"] == "1",
        )
        for index, pair in enumerate(pairs)
    ]

    print("threshold,paraphrase_recall,false_hits,near_miss_count")
    near_miss_count = sum(not should_match for _, should_match in scored_pairs)
    for step in range(85, 98):
        threshold = step / 100
        positives = [(score, label) for score, label in scored_pairs if label]
        recall = (
            sum(score >= threshold for score, _ in positives) / len(positives)
            if positives
            else 0.0
        )
        false_hits = sum(
            score >= threshold for score, should_match in scored_pairs if not should_match
        )
        print(f"{threshold:.2f},{recall:.3f},{false_hits},{near_miss_count}")
    print("Pick the lowest threshold with zero false hits on a sufficiently large,")
    print("manually labeled near-miss set. The bundled pairs are only a smoke test.")


if __name__ == "__main__":
    main()
