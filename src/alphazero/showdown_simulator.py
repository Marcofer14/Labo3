"""Client and input-log tracker for real Pokemon Showdown simulations."""

from __future__ import annotations

import inspect
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any


def choice_message(choice: Any) -> str:
    return getattr(choice, "message", None) or str(choice)


def normalize_choice(choice: str) -> str:
    text = str(choice or "").strip()
    if text.startswith("/choose "):
        return text[len("/choose ") :]
    if text.startswith("/team "):
        return "team " + text[len("/team ") :]
    if text == "/forfeit":
        return "forfeit"
    return text


def _last_request(battle: Any) -> dict[str, Any]:
    request = getattr(battle, "last_request", None) or {}
    return request if isinstance(request, dict) else {}


def _force_switch_flags(battle: Any) -> list[bool]:
    flags = getattr(battle, "force_switch", None)
    if isinstance(flags, (list, tuple)):
        return [bool(flag) for flag in flags]
    request_flags = _last_request(battle).get("forceSwitch", [])
    if isinstance(request_flags, (list, tuple)):
        return [bool(flag) for flag in request_flags]
    return []


def _choice_key(battle: Any, message: str) -> str:
    if message.startswith("/team ") or message.startswith("team "):
        return "team"
    request = _last_request(battle)
    rqid = request.get("rqid")
    turn = int(getattr(battle, "turn", 0) or 0)
    if any(_force_switch_flags(battle)):
        return f"force:{rqid if rqid is not None else turn}"
    if request.get("active"):
        return "move"
    if rqid is not None:
        return f"rqid:{rqid}"
    return f"turn:{turn}"


def _commit_immediately(battle: Any, message: str) -> bool:
    if any(_force_switch_flags(battle)):
        return True
    normalized = normalize_choice(message)
    return normalized == "forfeit"


@dataclass
class _TrackedBattle:
    committed: list[dict[str, str]] = field(default_factory=list)
    pending: dict[str, dict[str, str]] = field(default_factory=dict)
    completed_keys: set[str] = field(default_factory=set)


class ShowdownSimulationTracker:
    """Records completed choices as a replayable Showdown input log.

    Choices for the current simultaneous turn are kept pending until both sides
    have chosen, so a bot never gets to simulate using an opponent choice that
    would still be hidden in the real game.
    """

    def __init__(self, *, battle_format: str, team_text: str | None):
        self.battle_format = battle_format
        self.team_text = team_text
        self._battles: dict[str, _TrackedBattle] = {}

    def history_for(self, battle_tag: str) -> list[dict[str, str]]:
        return list(self._battles.get(battle_tag, _TrackedBattle()).committed)

    def record_choice(self, battle: Any, choice: Any) -> None:
        battle_tag = getattr(battle, "battle_tag", "")
        side = getattr(battle, "player_role", None)
        if not battle_tag or side not in {"p1", "p2"}:
            return

        message = choice_message(choice)
        normalized = normalize_choice(message)
        if not normalized:
            return

        tracked = self._battles.setdefault(battle_tag, _TrackedBattle())
        key = _choice_key(battle, message)
        if _commit_immediately(battle, message):
            tracked.committed.append({"side": side, "choice": normalized, "key": key})
            return

        reusable_key = key == "move"
        if not reusable_key and key in tracked.completed_keys:
            return

        pending = tracked.pending.setdefault(key, {})
        pending[side] = normalized

        if "p1" in pending and "p2" in pending:
            tracked.committed.append({"side": "p1", "choice": pending["p1"], "key": key})
            tracked.committed.append({"side": "p2", "choice": pending["p2"], "key": key})
            if not reusable_key:
                tracked.completed_keys.add(key)
            del tracked.pending[key]

    def payload_for(
        self,
        *,
        battle: Any,
        candidates: list[Any],
        depth: int,
        max_choices: int,
        opponent_policy: str,
        robust_worst_weight: float,
    ) -> dict[str, Any] | None:
        side = getattr(battle, "player_role", None)
        battle_tag = getattr(battle, "battle_tag", "")
        if side not in {"p1", "p2"} or not battle_tag or not self.team_text:
            return None
        return {
            "format": self.battle_format,
            "team_p1": self.team_text,
            "team_p2": self.team_text,
            "side": side,
            "depth": int(depth),
            "max_choices": int(max_choices),
            "opponent_policy": opponent_policy,
            "robust_worst_weight": float(robust_worst_weight),
            "history": self.history_for(battle_tag),
            "candidates": [choice_message(candidate) for candidate in candidates],
        }


class ShowdownSimulatorClient:
    """Small HTTP client for tools/showdown_sim_server.js."""

    def __init__(
        self,
        url: str | None = None,
        *,
        live_state_url: str | None = None,
        timeout: float = 10.0,
        max_choices: int = 12,
        opponent_policy: str = "robust",
        robust_worst_weight: float = 0.35,
    ):
        self.url = (url or os.environ.get("SHOWDOWN_SIMULATOR_URL") or "").rstrip("/")
        self.live_state_url = (
            live_state_url or os.environ.get("SHOWDOWN_LIVE_STATE_URL") or ""
        ).rstrip("/")
        self.timeout = timeout
        self.max_choices = max_choices
        self.opponent_policy = opponent_policy if opponent_policy in {"minimax", "mean", "robust"} else "robust"
        self.robust_worst_weight = max(0.0, min(float(robust_worst_weight), 1.0))
        self.last_used = False
        self.last_repairs = 0
        self.last_simulation_errors = 0
        self.last_skipped_branches = 0
        self.last_error_details: list[dict[str, Any]] = []
        self.last_error_stage_counts: dict[str, int] = {}
        self.last_error = ""
        self._warned = False

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    @property
    def live_state_enabled(self) -> bool:
        return bool(self.live_state_url)

    def _fetch_live_state(self, battle: Any) -> Any | None:
        battle_tag = getattr(battle, "battle_tag", "") or getattr(battle, "battle_id", "")
        if not battle_tag:
            self.last_error = "battle has no battle_tag for live state lookup"
            return None
        roomid = urllib.parse.quote(str(battle_tag), safe="")
        request = urllib.request.Request(
            f"{self.live_state_url}/battle-state?roomid={roomid}",
            headers={"accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = str(exc)
            self.last_error = f"live state HTTP {exc.code}: {detail[:1000]}"
            if not self._warned:
                print(f"  Aviso AlphaZero: estado vivo no disponible: {self.last_error}", flush=True)
                self._warned = True
            return None
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            self.last_error = f"live state unavailable: {exc}"
            if not self._warned:
                print(f"  Aviso AlphaZero: estado vivo no disponible ({exc}).", flush=True)
                self._warned = True
            return None
        if not result.get("ok"):
            self.last_error = str(result.get("error") or "live state returned ok=false")
            if not self._warned:
                print(f"  Aviso AlphaZero: estado vivo rechazado: {self.last_error}", flush=True)
                self._warned = True
            return None
        return result.get("state")

    def evaluate_candidates(
        self,
        *,
        tracker: ShowdownSimulationTracker | None,
        battle: Any,
        candidates: list[Any],
        depth: int,
    ) -> list[float] | None:
        self.last_used = False
        self.last_repairs = 0
        self.last_simulation_errors = 0
        self.last_skipped_branches = 0
        self.last_error_details = []
        self.last_error_stage_counts = {}
        self.last_error = ""
        if not self.enabled:
            self.last_error = "simulator disabled"
            return None

        endpoint = "/evaluate"
        if self.live_state_enabled:
            side = getattr(battle, "player_role", None)
            if side not in {"p1", "p2"}:
                self.last_error = "could not determine player side for live state"
                return None
            state = self._fetch_live_state(battle)
            if state is None:
                return None
            endpoint = "/offline/evaluate"
            payload = {
                "state": state,
                "side": side,
                "depth": int(depth),
                "max_choices": int(self.max_choices),
                "opponent_policy": self.opponent_policy,
                "robust_worst_weight": float(self.robust_worst_weight),
                "candidates": [choice_message(candidate) for candidate in candidates],
            }
        else:
            if tracker is None:
                self.last_error = "simulator missing tracker and live state bridge disabled"
                return None
            payload = tracker.payload_for(
                battle=battle,
                candidates=candidates,
                depth=depth,
                max_choices=self.max_choices,
                opponent_policy=self.opponent_policy,
                robust_worst_weight=self.robust_worst_weight,
            )
            if payload is None:
                self.last_error = "could not build simulator payload"
                return None

        try:
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                f"{self.url}{endpoint}",
                data=data,
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = str(exc)
            self.last_error = f"HTTP {exc.code}: {detail[:1000]}"
            if not self._warned:
                print(
                    f"  Aviso AlphaZero: simulador Showdown devolvio HTTP {exc.code}: "
                    f"{detail[:500]}",
                    flush=True,
                )
                self._warned = True
            return None
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            self.last_error = str(exc)
            if not self._warned:
                print(f"  Aviso AlphaZero: simulador Showdown no disponible ({exc}).", flush=True)
                self._warned = True
            return None

        self._store_diagnostics(result)
        if not result.get("ok"):
            self.last_error = str(result.get("error") or "simulator returned ok=false")
            if self.last_error_details:
                first = self.last_error_details[0]
                if isinstance(first, dict):
                    stage = first.get("stage") or "unknown"
                    detail = str(first.get("error") or "").replace("\n", " | ")[:300]
                    self.last_error = f"{self.last_error}; first {stage}: {detail}"
            if not self._warned:
                print(
                    f"  Aviso AlphaZero: simulador Showdown rechazo la evaluacion: "
                    f"{self.last_error}",
                    flush=True,
                )
                self._warned = True
            return None

        values = result.get("values")
        if not isinstance(values, list) or len(values) != len(candidates):
            self.last_error = (
                f"invalid simulator values: expected {len(candidates)}, "
                f"got {len(values) if isinstance(values, list) else type(values).__name__}"
            )
            return None
        self.last_used = True
        return [float(value) for value in values]

    def _store_diagnostics(self, result: dict[str, Any]) -> None:
        try:
            self.last_repairs = int(result.get("repairs", 0) or 0)
        except (TypeError, ValueError):
            self.last_repairs = 0
        try:
            self.last_simulation_errors = int(result.get("simulation_errors", 0) or 0)
        except (TypeError, ValueError):
            self.last_simulation_errors = 0
        try:
            self.last_skipped_branches = int(result.get("skipped_branches", 0) or 0)
        except (TypeError, ValueError):
            self.last_skipped_branches = 0
        errors = result.get("errors")
        self.last_error_details = errors if isinstance(errors, list) else []
        stage_counts = result.get("error_stage_counts")
        if isinstance(stage_counts, dict):
            self.last_error_stage_counts = {
                str(key): int(value)
                for key, value in stage_counts.items()
                if isinstance(value, (int, float))
            }
        else:
            self.last_error_stage_counts = {}


def attach_simulation_tracking(player: Any, tracker: ShowdownSimulationTracker) -> None:
    """Wraps any poke-env player so its real choices are recorded."""
    if getattr(player, "_showdown_sim_tracking", False):
        return
    setattr(player, "_showdown_sim_tracking", True)

    original_choose_move = player.choose_move

    def tracked_choose_move(battle):
        choice = original_choose_move(battle)
        if inspect.isawaitable(choice):

            async def record_async_choice():
                resolved = await choice
                tracker.record_choice(battle, resolved)
                return resolved

            return record_async_choice()

        tracker.record_choice(battle, choice)
        return choice

    player.choose_move = tracked_choose_move

    original_teampreview = getattr(player, "teampreview", None)
    if original_teampreview is None:
        return

    def tracked_teampreview(battle):
        choice = original_teampreview(battle)
        if inspect.isawaitable(choice):

            async def record_async_preview():
                resolved = await choice
                tracker.record_choice(battle, resolved)
                return resolved

            return record_async_preview()

        tracker.record_choice(battle, choice)
        return choice

    player.teampreview = tracked_teampreview
