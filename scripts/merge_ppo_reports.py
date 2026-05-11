"""Merge split recurrent PPO report.json files into one comparable report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.common import build_common_from_ppo_report, write_report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge recurrent PPO reports after a crashed/resumed run.")
    parser.add_argument("--reports", type=Path, nargs="+", required=True, help="report.json de cada tramo.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Carpeta destino para el reporte unido.")
    parser.add_argument(
        "--keep-empty-stages",
        action="store_true",
        help="Conservar stages sin duracion/episodios. Por defecto se descartan.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    report["_source_path"] = str(path)
    return report


def is_empty_stage(stage: dict[str, Any]) -> bool:
    return (
        int(stage.get("duration") or 0) <= 0
        and int(stage.get("n_episodes") or 0) <= 0
        and stage.get("final_value_loss") is None
        and stage.get("final_policy_loss") is None
        and not (stage.get("reward_breakdown") or {})
    )


def merge_reward_breakdowns(stages: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for stage in stages:
        for key, value in (stage.get("reward_breakdown") or {}).items():
            totals[key] += float(value or 0.0)
    return dict(sorted(totals.items()))


def merge_league(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for report in reports:
        source = report["_source_path"]
        for row in report.get("league") or []:
            merged = dict(row)
            merged["source_report"] = source
            key = (str(merged.get("path") or ""), str(merged.get("label") or ""), int(merged.get("timestep") or 0))
            if key in seen:
                continue
            seen.add(key)
            rows.append(merged)
    return sorted(rows, key=lambda item: int(item.get("timestep") or 0))


def merge_reports(reports: list[dict[str, Any]], *, keep_empty_stages: bool) -> dict[str, Any]:
    merged_stages: list[dict[str, Any]] = []
    ignored_stages: list[dict[str, Any]] = []
    for report_index, report in enumerate(reports, start=1):
        for stage in report.get("stages") or []:
            stage_copy = dict(stage)
            stage_copy["source_report"] = report["_source_path"]
            stage_copy["source_timestamp"] = report.get("timestamp")
            stage_copy["source_index"] = report_index
            if not keep_empty_stages and is_empty_stage(stage_copy):
                ignored_stages.append(stage_copy)
                continue
            merged_stages.append(stage_copy)

    merged_stages.sort(key=lambda stage: (int(stage.get("started") or 0), int(stage.get("ended") or 0)))

    gaps: list[dict[str, int]] = []
    overlaps: list[dict[str, int]] = []
    previous_end: int | None = None
    for stage in merged_stages:
        start = int(stage.get("started") or 0)
        end = int(stage.get("ended") or start)
        if previous_end is not None:
            if start > previous_end:
                gaps.append({"started": previous_end, "ended": start, "duration": start - previous_end})
            elif start < previous_end:
                overlaps.append({"started": start, "previous_end": previous_end, "overlap": previous_end - start})
        previous_end = max(previous_end or end, end)

    base_config = dict((reports[0].get("training_config") or {}) if reports else {})
    source_reports = [report["_source_path"] for report in reports]
    base_config.update(
        {
            "merged_from_reports": source_reports,
            "merge_ignored_empty_stages": len(ignored_stages),
            "merge_gap_timesteps": sum(gap["duration"] for gap in gaps),
            "merge_overlap_timesteps": sum(overlap["overlap"] for overlap in overlaps),
        }
    )

    merged = {
        "timestamp": "_".join(str(report.get("timestamp") or "unknown") for report in reports) + "_merged",
        "stages": merged_stages,
        "training_config": base_config,
        "extra_meta": {
            "algorithm": (reports[0].get("extra_meta") or {}).get("algorithm", "recurrent_ppo") if reports else "recurrent_ppo",
            "merge_kind": "ppo_resume_reports",
            "source_reports": source_reports,
            "ignored_empty_stages": [
                {
                    "stage": stage.get("stage"),
                    "started": stage.get("started"),
                    "ended": stage.get("ended"),
                    "source_report": stage.get("source_report"),
                }
                for stage in ignored_stages
            ],
            "timeline_gaps": gaps,
            "timeline_overlaps": overlaps,
            "note": "Los gaps se preservan como datos faltantes; no se inventan metricas para timesteps perdidos.",
        },
        "reward_breakdown": merge_reward_breakdowns(merged_stages),
        "league": merge_league(reports),
        "loss_history": [
            {
                "iteration": int(stage.get("stage") or index),
                "epoch": int(stage.get("stage") or index),
                "value_loss": stage.get("final_value_loss"),
                "policy_loss": stage.get("final_policy_loss"),
            }
            for index, stage in enumerate(merged_stages, start=1)
        ],
    }

    merged["common_metrics"] = build_common_from_ppo_report(merged, source=", ".join(source_reports))
    if merged_stages:
        timeline_start = int(merged_stages[0].get("started") or 0)
        timeline_end = max(int(stage.get("ended") or 0) for stage in merged_stages)
        accounted = sum(int(stage.get("duration") or 0) for stage in merged_stages)
        merged["common_metrics"].update(
            {
                "timeline_start": timeline_start,
                "timeline_end": timeline_end,
                "timeline_timesteps": timeline_end - timeline_start,
                "accounted_timesteps": accounted,
                "missing_timesteps": sum(gap["duration"] for gap in gaps),
                "ignored_empty_stages": len(ignored_stages),
            }
        )
    return merged


def main() -> int:
    args = parse_args()
    reports = [load_report(path) for path in args.reports]
    merged = merge_reports(reports, keep_empty_stages=args.keep_empty_stages)
    output = write_report_payload(merged, output_dir=args.output_dir, prefix="ppo_merged")
    print(f"Reporte PPO unido generado en: {output}")
    print(f"  · {output / 'report.html'}")
    print(f"  · {output / 'report.json'}")
    print(f"  · {output / 'common_metrics.json'}")
    if merged.get("extra_meta", {}).get("timeline_gaps"):
        print(f"  · gaps detectados: {merged['extra_meta']['timeline_gaps']}")
    if merged.get("extra_meta", {}).get("ignored_empty_stages"):
        print(f"  · stages vacios ignorados: {len(merged['extra_meta']['ignored_empty_stages'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
