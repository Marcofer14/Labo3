"""
Replay ingestion utilities for Pokemon Showdown.

The storage policy is intentionally conservative:
- raw replay JSON is kept intact;
- parsed JSON is derived from raw;
- decision JSONL is derived from parsed;
- downloaded raw files are never overwritten or deleted by this module.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests


REPLAY_BASE_URL = "https://replay.pokemonshowdown.com"
LADDER_BASE_URL = "https://logs.pokemonshowdown.com/ladder"
SCHEMA_VERSION = "1.0"
DOUBLE_DECISION_SCHEMA_VERSION = "2.0"
DEFAULT_OUTPUT_DIR = Path("data/replays")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


@dataclass(frozen=True)
class ReplayPaths:
    raw_dir: Path
    parsed_dir: Path
    datasets_dir: Path
    index_path: Path


class LadderTableParser(HTMLParser):
    """Small table parser for logs.pokemonshowdown.com/ladder/<format>."""

    def __init__(self) -> None:
        super().__init__()
        self.in_td = False
        self.current_cell: list[str] = []
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "td":
            self.in_td = True
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.in_td:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "td" and self.in_td:
            self.current_row.append("".join(self.current_cell).strip())
            self.current_cell = []
            self.in_td = False
        elif tag == "tr" and self.current_row:
            self.rows.append(self.current_row)
            self.current_row = []


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Labo3ReplayIngestion/1.0 (+local research project)",
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        }
    )
    return session


def ensure_paths(output_dir: Path, format_id: str) -> ReplayPaths:
    paths = ReplayPaths(
        raw_dir=output_dir / "raw" / format_id,
        parsed_dir=output_dir / "parsed" / format_id,
        datasets_dir=output_dir / "datasets",
        index_path=output_dir / "index.jsonl",
    )
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.parsed_dir.mkdir(parents=True, exist_ok=True)
    paths.datasets_dir.mkdir(parents=True, exist_ok=True)
    paths.index_path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json_if_missing(path: Path, data: dict[str, Any]) -> bool:
    if path.exists():
        return False
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    return True


def write_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
        f.write("\n")


def load_indexed_ids(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()

    ids: set[str] = set()
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            battle_id = row.get("battle_id")
            if battle_id:
                ids.add(battle_id)
    return ids


def load_dataset_keys(dataset_path: Path) -> set[str]:
    if not dataset_path.exists():
        return set()

    keys: set[str] = set()
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("sample_id")
            if key:
                keys.add(key)
    return keys


def replay_format_from_id(battle_id: str) -> str:
    return battle_id.split("-", 1)[0]


def replay_url(battle_id: str) -> str:
    return f"{REPLAY_BASE_URL}/{battle_id}"


def replay_json_url(battle_id: str) -> str:
    return f"{replay_url(battle_id)}.json"


def safe_filename_id(battle_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", battle_id)


def fetch_json(session: requests.Session, url: str, params: dict[str, Any] | None = None) -> Any:
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_text(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def search_replays(
    session: requests.Session,
    *,
    user: str | None,
    user2: str | None,
    format_id: str,
    limit: int,
    pages: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    """Search public replay metadata with pagination."""
    found: list[dict[str, Any]] = []
    before: int | None = None

    for _ in range(pages):
        params: dict[str, Any] = {"format": format_id}
        if user:
            params["user"] = user
        if user2:
            params["user2"] = user2
        if before:
            params["before"] = before

        results = fetch_json(session, f"{REPLAY_BASE_URL}/search.json", params=params)
        if not isinstance(results, list) or not results:
            break

        page_items = results[:50] if len(results) > 50 else results
        for item in page_items:
            battle_id = item.get("id")
            if not battle_id:
                continue
            if not battle_id.startswith(f"{format_id}-"):
                continue
            found.append(item)
            if len(found) >= limit:
                return found

        if len(results) <= 50:
            break

        last_upload = page_items[-1].get("uploadtime")
        if not last_upload:
            break
        before = int(last_upload)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return found


def fetch_ladder_users(session: requests.Session, format_id: str, top: int) -> list[str]:
    """Fetch top ladder usernames from the public ladder page."""
    html = fetch_text(session, f"{LADDER_BASE_URL}/{format_id}")
    parser = LadderTableParser()
    parser.feed(html)

    users: list[str] = []
    for row in parser.rows:
        if len(row) < 2:
            continue
        rank = row[0]
        username = row[1]
        if not rank.isdigit() or not username:
            continue
        users.append(username)
        if len(users) >= top:
            break
    return users


def read_users_file(path: Path) -> list[str]:
    users: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            users.append(line)
    return users


def split_log(log: str) -> list[str]:
    return [line for line in log.splitlines() if line]


def parse_side_actor(actor: str) -> tuple[str | None, str | None, str]:
    """Return (side, slot, pokemon_name) from strings like 'p1a: Pikachu'."""
    side = actor[:2] if actor.startswith(("p1", "p2")) else None
    slot = actor[:3] if actor.startswith(("p1", "p2")) and len(actor) >= 3 else side
    name = actor.split(":", 1)[1].strip() if ":" in actor else actor
    return side, slot, name


def parse_log_line(raw: str, line_index: int) -> dict[str, Any]:
    parts = raw.split("|")
    if parts and parts[0] == "":
        parts = parts[1:]
    event_type = parts[0] if parts else ""
    return {
        "line_index": line_index,
        "type": event_type,
        "args": parts[1:],
        "raw": raw,
    }


def parse_action(event: dict[str, Any], turn: int) -> dict[str, Any] | None:
    event_type = event["type"]
    args = event["args"]
    if event_type == "move" and len(args) >= 2:
        side, slot, pokemon = parse_side_actor(args[0])
        target_side, target_slot, target = parse_side_actor(args[2]) if len(args) >= 3 else (None, None, "")
        return {
            "turn": turn,
            "line_index": event["line_index"],
            "type": "move",
            "side": side,
            "slot": slot,
            "pokemon": pokemon,
            "move": args[1],
            "target_side": target_side,
            "target_slot": target_slot,
            "target": target,
            "raw": event["raw"],
        }
    if event_type == "switch" and len(args) >= 2:
        side, slot, pokemon = parse_side_actor(args[0])
        return {
            "turn": turn,
            "line_index": event["line_index"],
            "type": "switch",
            "side": side,
            "slot": slot,
            "pokemon": pokemon,
            "details": args[1],
            "hp_status": args[2] if len(args) >= 3 else None,
            "raw": event["raw"],
        }
    return None


def parse_replay(raw_replay: dict[str, Any], source: str) -> dict[str, Any]:
    battle_id = raw_replay.get("id") or raw_replay.get("battle_id")
    if not battle_id:
        raise ValueError("Replay JSON has no id")

    format_id = raw_replay.get("formatid") or replay_format_from_id(battle_id)
    raw_log = raw_replay.get("log") or ""
    lines = split_log(raw_log)

    players: dict[str, dict[str, Any]] = {}
    rules: list[str] = []
    team_preview: dict[str, list[dict[str, Any]]] = {"p1": [], "p2": []}
    turns: list[dict[str, Any]] = []
    current_turn = 0
    current_events: list[dict[str, Any]] = []
    current_actions: list[dict[str, Any]] = []
    winner: str | None = None
    tied = False
    metadata: dict[str, Any] = {
        "format_id": format_id,
        "format_name": raw_replay.get("format"),
        "uploadtime": raw_replay.get("uploadtime"),
        "views": raw_replay.get("views"),
        "rating": raw_replay.get("rating"),
        "source": source,
    }

    def flush_turn() -> None:
        if current_events or current_actions or current_turn > 0:
            turns.append(
                {
                    "turn": current_turn,
                    "events": list(current_events),
                    "actions": list(current_actions),
                }
            )

    for line_index, raw_line in enumerate(lines):
        event = parse_log_line(raw_line, line_index)
        event_type = event["type"]
        args = event["args"]

        if event_type == "turn" and args:
            flush_turn()
            current_events = []
            current_actions = []
            try:
                current_turn = int(args[0])
            except ValueError:
                current_turn += 1
            current_events.append(event)
            continue

        current_events.append(event)
        action = parse_action(event, current_turn)
        if action:
            current_actions.append(action)

        if event_type == "player" and len(args) >= 2:
            side = args[0]
            username = args[1]
            if not username and side in players:
                continue
            players[side] = {
                "username": username,
                "avatar": args[2] if len(args) >= 3 else None,
                "rating": args[3] if len(args) >= 4 and args[3] != "" else None,
            }
        elif event_type == "poke" and len(args) >= 2:
            side = args[0]
            team_preview.setdefault(side, []).append(
                {
                    "details": args[1],
                    "item": args[2] if len(args) >= 3 and args[2] else None,
                    "raw": raw_line,
                }
            )
        elif event_type == "rule" and args:
            rules.append(args[0])
        elif event_type in {"gametype", "gen", "tier", "rated"} and args:
            metadata[event_type] = args[0] if len(args) == 1 else args
        elif event_type == "win" and args:
            winner = args[0]
        elif event_type == "tie":
            tied = True

    flush_turn()

    max_turn = max((turn["turn"] for turn in turns), default=0)
    side_results: dict[str, str] = {}
    for side, player in players.items():
        username = player.get("username")
        if tied:
            side_results[side] = "tie"
        elif winner and username == winner:
            side_results[side] = "win"
        elif winner:
            side_results[side] = "loss"
        else:
            side_results[side] = "unknown"

    parsed = {
        "schema_version": SCHEMA_VERSION,
        "battle_id": battle_id,
        "format_id": format_id,
        "replay_url": replay_url(battle_id),
        "metadata": metadata,
        "players": players,
        "rules": rules,
        "team_preview": team_preview,
        "result": {
            "winner": winner,
            "tied": tied,
            "turns": max_turn,
            "side_results": side_results,
        },
        "turns": turns,
        "raw_log": lines,
    }
    return parsed


def make_decision_samples(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    battle_id = parsed["battle_id"]
    result = parsed.get("result", {})
    side_results = result.get("side_results", {})

    for turn in parsed.get("turns", []):
        turn_number = turn.get("turn", 0)
        for action_index, action in enumerate(turn.get("actions", [])):
            side = action.get("side")
            if side not in {"p1", "p2"}:
                continue
            sample_id = f"{battle_id}:t{turn_number}:{side}:{action_index}:{action.get('line_index')}"
            samples.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "battle_id": battle_id,
                    "format_id": parsed["format_id"],
                    "turn": turn_number,
                    "player_side": side,
                    "player_username": parsed.get("players", {}).get(side, {}).get("username"),
                    "action": action,
                    "outcome": side_results.get(side, "unknown"),
                    "observation_ref": {
                        "type": "replay_log_turn_context",
                        "parsed_path_hint": f"parsed/{parsed['format_id']}/{battle_id}.json",
                        "turn": turn_number,
                        "events_available": len(turn.get("events", [])),
                    },
                }
            )
    return samples


def _action_sort_key(action: dict[str, Any]) -> tuple[str, int]:
    return (str(action.get("slot") or ""), int(action.get("line_index") or 0))


def _normalize_double_action(action: dict[str, Any]) -> dict[str, Any]:
    action_type = action.get("type")
    normalized = {
        "type": action_type,
        "slot": action.get("slot"),
        "side": action.get("side"),
        "pokemon": action.get("pokemon"),
        "line_index": action.get("line_index"),
        "raw": action.get("raw"),
    }
    if action_type == "move":
        normalized.update(
            {
                "move": action.get("move"),
                "target_side": action.get("target_side"),
                "target_slot": action.get("target_slot"),
                "target": action.get("target"),
            }
        )
    elif action_type == "switch":
        normalized.update(
            {
                "details": action.get("details"),
                "hp_status": action.get("hp_status"),
            }
        )
    return normalized


def _double_order_signature(actions: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for action in sorted(actions, key=_action_sort_key):
        slot = action.get("slot") or "slot?"
        if action.get("type") == "move":
            move = re.sub(r"[^a-z0-9]+", "", str(action.get("move") or "").lower())
            target = action.get("target_slot") or action.get("target") or "target?"
            parts.append(f"{slot}:move:{move}:{target}")
        elif action.get("type") == "switch":
            details = str(action.get("details") or action.get("pokemon") or "").split(",", 1)[0]
            pokemon = re.sub(r"[^a-z0-9]+", "", details.lower())
            parts.append(f"{slot}:switch:{pokemon}")
        else:
            parts.append(f"{slot}:{action.get('type') or 'unknown'}")
    return "|".join(parts)


def _revealed_moves_before(parsed: dict[str, Any], line_index: int) -> dict[str, dict[str, list[str]]]:
    revealed: dict[str, dict[str, set[str]]] = {"p1": {}, "p2": {}}
    for turn in parsed.get("turns", []):
        for event in turn.get("events", []):
            if int(event.get("line_index", 10**12)) >= line_index:
                moves: dict[str, dict[str, list[str]]] = {}
                for side, pokemon_moves in revealed.items():
                    moves[side] = {
                        pokemon: sorted(move_names)
                        for pokemon, move_names in pokemon_moves.items()
                    }
                return moves
            if event.get("type") != "move":
                continue
            args = event.get("args") or []
            if len(args) < 2:
                continue
            side, _, pokemon = parse_side_actor(args[0])
            if side not in {"p1", "p2"}:
                continue
            revealed.setdefault(side, {}).setdefault(pokemon, set()).add(args[1])

    return {
        side: {pokemon: sorted(move_names) for pokemon, move_names in pokemon_moves.items()}
        for side, pokemon_moves in revealed.items()
    }


def make_double_decision_samples(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Build side-level VGC double-decision samples from parsed replay actions.

    Replays expose resolved battle log events, not the original `/choose`
    message. This groups the first observed action for each side+slot in a
    turn. It is a conservative approximation of the double order and keeps
    enough line-index metadata to reconstruct public state later.
    """
    samples: list[dict[str, Any]] = []
    battle_id = parsed["battle_id"]
    result = parsed.get("result", {})
    side_results = result.get("side_results", {})
    players = parsed.get("players", {})
    team_preview = parsed.get("team_preview", {})

    for turn in parsed.get("turns", []):
        turn_number = int(turn.get("turn", 0) or 0)
        grouped: dict[str, dict[str, dict[str, Any]]] = {"p1": {}, "p2": {}}

        for action in sorted(turn.get("actions", []), key=lambda item: int(item.get("line_index") or 0)):
            side = action.get("side")
            slot = action.get("slot")
            if side not in {"p1", "p2"} or not slot:
                continue
            grouped.setdefault(side, {})
            if slot not in grouped[side]:
                grouped[side][slot] = action

        for side, slot_actions in grouped.items():
            if not slot_actions:
                continue
            actions = sorted(slot_actions.values(), key=_action_sort_key)
            first_line = min(int(action.get("line_index") or 0) for action in actions)
            last_line = max(int(action.get("line_index") or 0) for action in actions)
            decision_type = "lead" if turn_number == 0 else "turn"
            if all(action.get("type") == "switch" for action in actions):
                decision_type = "lead" if turn_number == 0 else "switch_only"
            elif any(action.get("type") == "move" for action in actions):
                decision_type = "turn_order"

            sample_id = f"{battle_id}:double:t{turn_number}:{side}:{first_line}-{last_line}"
            samples.append(
                {
                    "schema_version": DOUBLE_DECISION_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "battle_id": battle_id,
                    "format_id": parsed["format_id"],
                    "turn": turn_number,
                    "player_side": side,
                    "player_username": players.get(side, {}).get("username"),
                    "decision_type": decision_type,
                    "order_signature": _double_order_signature(actions),
                    "actions": [_normalize_double_action(action) for action in actions],
                    "observed_slots": sorted(slot_actions),
                    "outcome": side_results.get(side, "unknown"),
                    "team_preview": {
                        "own": team_preview.get(side, []),
                        "opp": team_preview.get("p2" if side == "p1" else "p1", []),
                    },
                    "revealed_moves": _revealed_moves_before(parsed, first_line),
                    "observation_ref": {
                        "type": "replay_log_before_double_decision",
                        "parsed_path_hint": f"parsed/{parsed['format_id']}/{battle_id}.json",
                        "turn": turn_number,
                        "first_action_line_index": first_line,
                        "last_action_line_index": last_line,
                    },
                }
            )
    return samples


def index_row(
    *,
    raw_replay: dict[str, Any],
    parsed: dict[str, Any],
    raw_path: Path,
    parsed_path: Path,
    source: str,
    searched_user: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "battle_id": parsed["battle_id"],
        "format_id": parsed["format_id"],
        "source": source,
        "searched_user": searched_user,
        "replay_url": replay_url(parsed["battle_id"]),
        "raw_path": str(raw_path.as_posix()),
        "parsed_path": str(parsed_path.as_posix()),
        "ingested_at": now_utc(),
        "uploadtime": raw_replay.get("uploadtime"),
        "players": raw_replay.get("players") or [
            player.get("username") for player in parsed.get("players", {}).values()
        ],
        "rating": raw_replay.get("rating"),
        "views": raw_replay.get("views"),
        "turns": parsed.get("result", {}).get("turns"),
        "winner": parsed.get("result", {}).get("winner"),
    }


def ingest_replay(
    session: requests.Session,
    *,
    battle_id: str,
    format_id: str,
    output_dir: Path,
    source: str,
    searched_user: str | None,
    reparse: bool,
    dry_run: bool,
    existing_indexed_ids: set[str] | None = None,
    existing_sample_ids: set[str] | None = None,
    existing_double_sample_ids: set[str] | None = None,
) -> tuple[bool, bool, int, int]:
    paths = ensure_paths(output_dir, format_id)
    raw_path = paths.raw_dir / f"{safe_filename_id(battle_id)}.json"
    parsed_path = paths.parsed_dir / f"{safe_filename_id(battle_id)}.json"
    dataset_path = paths.datasets_dir / f"{format_id}_decisions.jsonl"
    double_dataset_path = paths.datasets_dir / f"{format_id}_double_decisions.jsonl"

    if dry_run:
        print(f"DRY RUN: would ingest {battle_id}")
        return False, False, 0, 0

    downloaded = False
    if raw_path.exists():
        raw_replay = read_json(raw_path)
    else:
        raw_replay = fetch_json(session, replay_json_url(battle_id))
        write_json_if_missing(raw_path, raw_replay)
        downloaded = True

    parsed_exists = parsed_path.exists()
    if parsed_exists and not reparse:
        parsed = read_json(parsed_path)
        parsed_written = False
    else:
        parsed = parse_replay(raw_replay, source=source)
        write_json(parsed_path, parsed)
        parsed_written = True

    existing_ids = existing_indexed_ids if existing_indexed_ids is not None else load_indexed_ids(paths.index_path)
    if battle_id not in existing_ids:
        append_jsonl(
            paths.index_path,
            index_row(
                raw_replay=raw_replay,
                parsed=parsed,
                raw_path=raw_path,
                parsed_path=parsed_path,
                source=source,
                searched_user=searched_user,
            ),
        )
        existing_ids.add(battle_id)

    if existing_sample_ids is None:
        existing_sample_ids = load_dataset_keys(dataset_path)
    new_samples = 0
    for sample in make_decision_samples(parsed):
        if sample["sample_id"] in existing_sample_ids:
            continue
        append_jsonl(dataset_path, sample)
        existing_sample_ids.add(sample["sample_id"])
        new_samples += 1

    if existing_double_sample_ids is None:
        existing_double_sample_ids = load_dataset_keys(double_dataset_path)
    new_double_samples = 0
    for sample in make_double_decision_samples(parsed):
        if sample["sample_id"] in existing_double_sample_ids:
            continue
        append_jsonl(double_dataset_path, sample)
        existing_double_sample_ids.add(sample["sample_id"])
        new_double_samples += 1

    return downloaded, parsed_written, new_samples, new_double_samples


def rebuild_parsed(output_dir: Path, format_id: str, reparse: bool) -> None:
    paths = ensure_paths(output_dir, format_id)
    raw_files = sorted(paths.raw_dir.glob("*.json"))
    if not raw_files:
        print(f"No raw replays found in {paths.raw_dir}")
        return

    dataset_path = paths.datasets_dir / f"{format_id}_decisions.jsonl"
    double_dataset_path = paths.datasets_dir / f"{format_id}_double_decisions.jsonl"
    existing_indexed_ids = load_indexed_ids(paths.index_path)
    existing_sample_ids = load_dataset_keys(dataset_path)
    existing_double_sample_ids = load_dataset_keys(double_dataset_path)
    total_samples = 0
    total_double_samples = 0
    for raw_path in raw_files:
        raw_replay = read_json(raw_path)
        battle_id = raw_replay.get("id") or raw_path.stem
        _, parsed_written, samples, double_samples = ingest_replay(
            make_session(),
            battle_id=battle_id,
            format_id=format_id,
            output_dir=output_dir,
            source="local_raw_rebuild",
            searched_user=None,
            reparse=reparse,
            dry_run=False,
            existing_indexed_ids=existing_indexed_ids,
            existing_sample_ids=existing_sample_ids,
            existing_double_sample_ids=existing_double_sample_ids,
        )
        total_samples += samples
        total_double_samples += double_samples
        status = "parsed" if parsed_written else "kept"
        print(f"{status}: {battle_id} (+{samples} samples, +{double_samples} double samples)")
    print(
        "Rebuild complete: "
        f"{len(raw_files)} raw replays, {total_samples} new samples, "
        f"{total_double_samples} new double samples"
    )


def collect_target_users(args: argparse.Namespace, session: requests.Session) -> list[str]:
    users: list[str] = []
    users.extend(args.user or [])
    if args.users_file:
        users.extend(read_users_file(Path(args.users_file)))
    if args.include_default_bots:
        users.extend(["Laboratorio3IA", "Laboratorio3IA-B"])
    if args.top_ladder:
        ladder_users = fetch_ladder_users(session, args.format, args.top_ladder)
        print(f"Top ladder users found: {len(ladder_users)}")
        users.extend(ladder_users)

    deduped: list[str] = []
    seen: set[str] = set()
    for user in users:
        key = user.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(user)
    return deduped


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and parse Pokemon Showdown replays")
    parser.add_argument("--format", required=True, help="Showdown format id, e.g. gen9randombattle")
    parser.add_argument("--user", action="append", help="Username to search. Can be repeated.")
    parser.add_argument("--user2", help="Optional second username filter for challenge/self-play searches.")
    parser.add_argument("--users-file", help="Text file with one username per line.")
    parser.add_argument(
        "--include-default-bots",
        action="store_true",
        help="Include Laboratorio3IA and Laboratorio3IA-B in the search.",
    )
    parser.add_argument(
        "--top-ladder",
        type=int,
        default=0,
        help="Also search replay users from the top N ladder page entries.",
    )
    parser.add_argument("--replay-id", action="append", help="Specific replay id to ingest. Can be repeated.")
    parser.add_argument("--limit", type=int, default=25, help="Max replays per user/search source.")
    parser.add_argument("--pages", type=int, default=1, help="Search pages per user.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between API requests.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Replay dataset root.")
    parser.add_argument("--source", default="pokemon_showdown_replay", help="Source label for metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing files.")
    parser.add_argument("--reparse", action="store_true", help="Regenerate parsed files from existing raw files.")
    parser.add_argument(
        "--rebuild-parsed",
        action="store_true",
        help="Parse all existing raw files for --format and rebuild derived JSON/JSONL.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    session = make_session()

    if args.rebuild_parsed:
        rebuild_parsed(output_dir, args.format, reparse=args.reparse)
        return 0

    target_ids: list[tuple[str, str | None]] = []

    for replay_id in args.replay_id or []:
        target_ids.append((replay_id, None))

    users = collect_target_users(args, session)
    for user in users:
        print(f"Searching replays for user={user!r} format={args.format!r}")
        results = search_replays(
            session,
            user=user,
            user2=args.user2,
            format_id=args.format,
            limit=args.limit,
            pages=args.pages,
            sleep_seconds=args.sleep,
        )
        print(f"  found {len(results)} replay(s)")
        target_ids.extend((item["id"], user) for item in results)
        if args.sleep > 0:
            time.sleep(args.sleep)

    if not target_ids:
        print("No replay ids to ingest. Use --user, --users-file, --top-ladder, or --replay-id.")
        return 0

    deduped_targets: list[tuple[str, str | None]] = []
    seen_ids: set[str] = set()
    for battle_id, searched_user in target_ids:
        if battle_id in seen_ids:
            continue
        seen_ids.add(battle_id)
        deduped_targets.append((battle_id, searched_user))

    downloaded_count = 0
    parsed_count = 0
    sample_count = 0
    double_sample_count = 0
    paths = ensure_paths(output_dir, args.format)
    existing_indexed_ids = load_indexed_ids(paths.index_path)
    existing_sample_ids = load_dataset_keys(paths.datasets_dir / f"{args.format}_decisions.jsonl")
    existing_double_sample_ids = load_dataset_keys(
        paths.datasets_dir / f"{args.format}_double_decisions.jsonl"
    )
    for battle_id, searched_user in deduped_targets:
        format_id = replay_format_from_id(battle_id)
        if format_id != args.format:
            print(f"Skipping {battle_id}: expected format {args.format}, got {format_id}")
            continue
        try:
            downloaded, parsed_written, samples, double_samples = ingest_replay(
                session,
                battle_id=battle_id,
                format_id=args.format,
                output_dir=output_dir,
                source=args.source,
                searched_user=searched_user,
                reparse=args.reparse,
                dry_run=args.dry_run,
                existing_indexed_ids=existing_indexed_ids,
                existing_sample_ids=existing_sample_ids,
                existing_double_sample_ids=existing_double_sample_ids,
            )
        except requests.HTTPError as exc:
            print(f"HTTP error for {battle_id}: {exc}")
            continue
        except Exception as exc:
            print(f"Error ingesting {battle_id}: {exc}")
            continue

        downloaded_count += int(downloaded)
        parsed_count += int(parsed_written)
        sample_count += samples
        double_sample_count += double_samples
        raw_status = "downloaded" if downloaded else "cached"
        parsed_status = "parsed" if parsed_written else "parsed-cached"
        print(
            f"{raw_status}, {parsed_status}: {battle_id} "
            f"(+{samples} samples, +{double_samples} double samples)"
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    query = urlencode({"format": args.format})
    print("")
    print("Ingestion complete")
    print(f"  targets: {len(deduped_targets)}")
    print(f"  downloaded raw: {downloaded_count}")
    print(f"  parsed files written: {parsed_count}")
    print(f"  new decision samples: {sample_count}")
    print(f"  new double decision samples: {double_sample_count}")
    print(f"  output: {output_dir}")
    print(f"  replay search reference: {REPLAY_BASE_URL}/search.json?{query}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
