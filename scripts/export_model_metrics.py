"""Export comparable metrics for PPO and AlphaZero runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.common import write_alphazero_report, write_ppo_common_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export common metrics for model comparison")
    sub = parser.add_subparsers(dest="kind", required=True)

    az = sub.add_parser("alphazero", help="Build report from AlphaZero JSONL logs")
    az.add_argument("--training-log-path", type=Path, required=True)
    az.add_argument("--rollout-path", type=Path, required=True)
    az.add_argument("--output-dir", type=Path, required=True)
    az.add_argument("--title", type=str, default="AlphaZero MCTS+PPO")

    ppo = sub.add_parser("ppo", help="Normalize an existing PPO report.json")
    ppo.add_argument("--report-json", type=Path, required=True)
    ppo.add_argument("--output-dir", type=Path, required=True)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.kind == "alphazero":
        out = write_alphazero_report(
            training_log_path=args.training_log_path,
            rollout_path=args.rollout_path,
            output_dir=args.output_dir,
            title=args.title,
        )
    else:
        out = write_ppo_common_report(
            report_json=args.report_json,
            output_dir=args.output_dir,
        )
    print(f"Reporte generado en: {out}")
    print(f"  - {out / 'report.json'}")
    print(f"  - {out / 'common_metrics.json'}")
    print(f"  - {out / 'league_stats.json'}")
    print(f"  - {out / 'plots'}")
    print(f"  - {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
