"""Summarize AlphaZero Showdown simulator diagnostics JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def short_error(text: Any, limit: int = 180) -> str:
    value = str(text or "").replace("\n", " | ")
    return value[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize simulator diagnostics JSONL")
    parser.add_argument("path", type=Path)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--stage",
        type=str,
        default="",
        help="Only detailed --examples whose error sample stage matches this value.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Print this many detailed error samples after the summary.",
    )
    args = parser.parse_args()

    rows = load_rows(args.path)
    if not rows:
        print(f"No diagnostics found in {args.path}")
        return 0

    stage_counts: Counter[str] = Counter()
    error_counts: Counter[tuple[str, str]] = Counter()
    own_choice_counts: Counter[str] = Counter()
    opponent_choice_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    key_counts: Counter[str] = Counter()
    request_state_counts: Counter[str] = Counter()
    active_counts: Counter[str] = Counter()
    detailed_examples: list[dict[str, Any]] = []
    total_errors = 0
    total_skipped = 0
    total_repairs = 0

    for row in rows:
        total_errors += int(row.get("simulator_errors") or 0)
        total_skipped += int(row.get("simulator_skipped_branches") or 0)
        total_repairs += int(row.get("simulator_repairs") or 0)
        selected = row.get("selected_message")
        if selected:
            selected_counts[str(selected)] += 1

        for stage, count in (row.get("stage_counts") or {}).items():
            stage_counts[str(stage)] += int(count)

        for sample in row.get("error_samples") or []:
            if not isinstance(sample, dict):
                continue
            stage = str(sample.get("stage") or "unknown")
            error_counts[(stage, short_error(sample.get("error")))] += 1
            own_choice = sample.get("ownChoice") or sample.get("choice") or sample.get("candidate")
            opponent_choice = sample.get("opponentChoice")
            if own_choice:
                own_choice_counts[str(own_choice)] += 1
            if opponent_choice:
                opponent_choice_counts[str(opponent_choice)] += 1
            key = sample.get("key")
            if key:
                key_counts[str(key)] += 1
            request_state = sample.get("requestState")
            if request_state:
                request_state_counts[str(request_state)] += 1
            p1_active = sample.get("p1Active")
            p2_active = sample.get("p2Active")
            if p1_active or p2_active:
                active_counts[f"p1={p1_active} | p2={p2_active}"] += 1
            if args.examples and (not args.stage or stage == args.stage):
                detailed_examples.append(
                    {
                        "iteration": row.get("iteration"),
                        "opponent": row.get("opponent"),
                        "battle_tag": row.get("battle_tag"),
                        "turn": row.get("turn"),
                        "selected_message": row.get("selected_message"),
                        "sample": sample,
                    }
                )

    print(f"Diagnostics rows: {len(rows)}")
    print(f"Total repairs: {total_repairs}")
    print(f"Total simulator_errors: {total_errors}")
    print(f"Total skipped_branches: {total_skipped}")

    print("\nTop stages:")
    for stage, count in stage_counts.most_common(args.top):
        print(f"  {stage}: {count}")

    print("\nTop error samples:")
    for (stage, error), count in error_counts.most_common(args.top):
        print(f"  [{stage}] x{count}: {error}")

    print("\nTop own/candidate choices in samples:")
    for choice, count in own_choice_counts.most_common(args.top):
        print(f"  x{count}: {choice}")

    print("\nTop opponent choices in samples:")
    for choice, count in opponent_choice_counts.most_common(args.top):
        print(f"  x{count}: {choice}")

    if key_counts:
        print("\nTop history/request keys in samples:")
        for key, count in key_counts.most_common(args.top):
            print(f"  x{count}: {key}")

    if request_state_counts:
        print("\nTop request states in samples:")
        for state, count in request_state_counts.most_common(args.top):
            print(f"  x{count}: {state}")

    if active_counts:
        print("\nTop active Pokemon contexts in samples:")
        for active, count in active_counts.most_common(args.top):
            print(f"  x{count}: {active}")

    print("\nTop selected messages:")
    for choice, count in selected_counts.most_common(args.top):
        print(f"  x{count}: {choice}")

    if args.examples:
        print("\nDetailed examples:")
        for index, example in enumerate(detailed_examples[: args.examples], start=1):
            print(f"\nExample {index}:")
            print(json.dumps(example, ensure_ascii=True, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
