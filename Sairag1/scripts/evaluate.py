import argparse
import csv
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def answer_overlap(answer: str, reference: str) -> float:
    tokens = lambda text: set(re.findall(r"\b[\w'-]+\b", text.lower()))
    expected = tokens(reference)
    if not expected:
        return 0.0
    return len(tokens(answer) & expected) / len(expected)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def run_evaluation(
    api_url: str, eval_path: Path, output_path: Path, repeats: int
) -> None:
    with eval_path.open(newline="", encoding="utf-8") as file:
        cases = list(csv.DictReader(file))
    if not cases:
        raise ValueError(f"No evaluation cases found in {eval_path}")

    results: list[dict[str, Any]] = []
    evaluated_at = datetime.now(timezone.utc).isoformat()
    for strategy in ("baseline", "optimized"):
        for case in cases:
            for repeat in range(repeats):
                response = requests.post(
                    f"{api_url.rstrip('/')}/ask",
                    json={"query": case["question"], "strategy": strategy},
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()
                sources = result.get("sources", [])
                source_hit = any(
                    source.get("id") == case["source_id"] for source in sources
                )
                results.append(
                    {
                        "strategy": strategy,
                        "evaluated_at": evaluated_at,
                        "question_id": case["id"],
                        "repeat": repeat + 1,
                        "answer_overlap": round(
                            answer_overlap(result["answer"], case["reference_answer"]),
                            4,
                        ),
                        "retrieval_hit": int(source_hit),
                        "cache_hit": int(result["cache_hit"]),
                        "route": result["route"],
                        "cost_usd": result["cost_usd"],
                        "latency_ms": result["latency_ms"],
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    for strategy in ("baseline", "optimized"):
        subset = [row for row in results if row["strategy"] == strategy]
        costs = [float(row["cost_usd"]) for row in subset]
        latencies = [float(row["latency_ms"]) for row in subset]
        print(
            f"{strategy}: requests={len(subset)}, "
            f"retrieval_hit_rate={statistics.mean(row['retrieval_hit'] for row in subset):.1%}, "
            f"answer_token_overlap={statistics.mean(row['answer_overlap'] for row in subset):.1%}, "
            f"cache_hit_rate={statistics.mean(row['cache_hit'] for row in subset):.1%}, "
            f"mean_cost_usd={statistics.mean(costs):.8f}, "
            f"p50_ms={percentile(latencies, 0.50):.2f}, "
            f"p95_ms={percentile(latencies, 0.95):.2f}"
        )
    print(f"Per-request results written to {output_path}")
    print("Answer overlap is a lexical proxy, not an LLM-judge correctness score.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the eval set through baseline and optimized RAG."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--eval", type=Path, default=Path("data/eval.csv"))
    parser.add_argument("--output", type=Path, default=Path("eval_results.csv"))
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Replay each question this many times to include repeat traffic.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    run_evaluation(args.api_url, args.eval, args.output, args.repeats)


if __name__ == "__main__":
    main()
