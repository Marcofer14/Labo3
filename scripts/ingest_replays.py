"""
CLI wrapper for replay ingestion.

Examples:
  python scripts/ingest_replays.py --format gen9randombattle --include-default-bots --limit 10
  python scripts/ingest_replays.py --format gen9randombattle --top-ladder 10 --limit 5
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.replay_ingestion import main


if __name__ == "__main__":
    raise SystemExit(main())
