"""Regenerate a comparable report directory from an existing report.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.common import write_report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate report.html and plots from report.json")
    parser.add_argument("--report-json", type=Path, required=True, help="Path al report.json existente.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Carpeta nueva para el reporte regenerado.")
    parser.add_argument("--prefix", type=str, default="regenerated", help="Prefijo si se usa output_root internamente.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = json.loads(args.report_json.read_text(encoding="utf-8"))
    report.setdefault("extra_meta", {})
    report["extra_meta"]["regenerated_from"] = str(args.report_json)

    output = write_report_payload(report, output_dir=args.output_dir, prefix=args.prefix)
    print(f"Reporte regenerado en: {output}")
    print(f"  · {output / 'report.html'}")
    print(f"  · {output / 'report.json'}")
    print(f"  · {output / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
