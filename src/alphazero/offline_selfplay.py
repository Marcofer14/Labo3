"""Offline AlphaZero-style self-play using Pokemon Showdown's JS engine.

This path keeps the live battle state inside tools/showdown_sim_server.js and
therefore uses one controlled PRNG seed for the rollout and for MCTS branch
evaluation. Unlike websocket collection, it does not replay a public log to
reconstruct state, so it avoids RNG drift.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from login import load_team, should_use_team
from src.alphazero.features import (
    simulator_action_features,
    simulator_state_features,
)
from src.alphazero.network import AlphaZeroPolicyValueNet, load_checkpoint
from src.format_resolver import resolve_format


AppendJsonl = Callable[[Path, dict[str, Any]], None]
CompactDiagnostic = Callable[..., dict[str, Any]]


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    total = exp.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full_like(logits, 1.0 / len(logits), dtype=np.float64)
    return exp / total


class OfflineShowdownClient:
    def __init__(self, url: str | None = None, *, timeout: float = 60.0):
        self.url = (url or os.environ.get("SHOWDOWN_SIMULATOR_URL") or "").rstrip("/")
        self.timeout = timeout
        if not self.url:
            raise ValueError("SHOWDOWN_SIMULATOR_URL is required for offline self-play")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=data,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"offline simulator HTTP {exc.code}: {detail[:1000]}") from exc
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "offline simulator returned ok=false"))
        return result

    def start(
        self,
        *,
        battle_format: str,
        team_text: str,
        seed: list[int],
        team_choice_p1: str,
        team_choice_p2: str,
        max_choices: int,
    ) -> dict[str, Any]:
        result = self._post(
            "/offline/start",
            {
                "format": battle_format,
                "team_p1": team_text,
                "team_p2": team_text,
                "seed": seed,
                "team_choice_p1": team_choice_p1,
                "team_choice_p2": team_choice_p2,
                "max_choices": max_choices,
            },
        )
        return result["battle"]

    def choose(
        self,
        *,
        state: Any,
        choices: dict[str, str],
        max_choices: int,
    ) -> dict[str, Any]:
        result = self._post(
            "/offline/choose",
            {
                "state": state,
                "choices": choices,
                "max_choices": max_choices,
            },
        )
        return result["battle"]

    def evaluate(
        self,
        *,
        state: Any,
        side: str,
        candidates: list[str],
        depth: int,
        max_choices: int,
        opponent_policy: str,
        robust_worst_weight: float,
    ) -> dict[str, Any]:
        return self._post(
            "/offline/evaluate",
            {
                "state": state,
                "side": side,
                "candidates": candidates,
                "depth": depth,
                "max_choices": max_choices,
                "opponent_policy": opponent_policy,
                "robust_worst_weight": robust_worst_weight,
            },
        )


@dataclass
class OfflineSearchResult:
    choice: str
    row: dict[str, Any]


class OfflineAlphaZeroSearch:
    def __init__(
        self,
        *,
        model: AlphaZeroPolicyValueNet,
        client: OfflineShowdownClient,
        simulations: int,
        depth: int,
        max_candidates: int,
        cpuct: float,
        temperature: float,
        device: str,
        simulator_max_choices: int,
        simulator_opponent_policy: str,
        simulator_robust_worst_weight: float,
        require_simulator: bool,
    ):
        self.model = model
        self.client = client
        self.simulations = max(1, int(simulations))
        self.depth = max(1, int(depth))
        self.max_candidates = int(max_candidates or 0)
        self.cpuct = float(cpuct)
        self.temperature = float(temperature)
        self.device = torch.device(device)
        self.simulator_max_choices = max(1, int(simulator_max_choices))
        self.simulator_opponent_policy = simulator_opponent_policy
        self.simulator_robust_worst_weight = float(simulator_robust_worst_weight)
        self.require_simulator = bool(require_simulator)
        self.rng = np.random.default_rng()
        self.model.to(self.device)
        self.model.eval()

    def _model_eval(
        self,
        state_features: np.ndarray,
        action_features: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        states = torch.as_tensor(state_features, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(action_features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, values = self.model(states.unsqueeze(0), actions.unsqueeze(0))
        priors = _softmax(logits.squeeze(0).detach().cpu().numpy())
        value = float(values.squeeze(0).detach().cpu().item())
        return priors.astype(np.float32), value

    def search(self, snapshot: dict[str, Any], side: str) -> OfflineSearchResult | None:
        candidates = list((snapshot.get("legal") or {}).get(side) or [])
        if self.max_candidates > 0:
            candidates = candidates[: self.max_candidates]
        if not candidates:
            return None

        state_features = simulator_state_features(snapshot, side)
        action_features = np.stack(
            [simulator_action_features(candidate) for candidate in candidates]
        ).astype(np.float32)
        priors, state_value = self._model_eval(state_features, action_features)

        simulator_used = False
        simulator_errors = 0
        simulator_skipped = 0
        simulator_error_details: list[dict[str, Any]] = []
        simulator_error_stage_counts: dict[str, int] = {}
        search_values = np.full(len(candidates), state_value, dtype=np.float32)

        if self.depth >= 2:
            try:
                result = self.client.evaluate(
                    state=snapshot["state"],
                    side=side,
                    candidates=candidates,
                    depth=max(1, self.depth - 1),
                    max_choices=self.simulator_max_choices,
                    opponent_policy=self.simulator_opponent_policy,
                    robust_worst_weight=self.simulator_robust_worst_weight,
                )
                values = result.get("values") or []
                if len(values) == len(candidates):
                    search_values = np.asarray(values, dtype=np.float32)
                    simulator_used = True
                elif self.require_simulator:
                    raise RuntimeError(
                        f"offline simulator returned {len(values)} values for {len(candidates)} candidates"
                    )
                simulator_errors = int(result.get("simulation_errors") or 0)
                simulator_skipped = int(result.get("skipped_branches") or 0)
                details = result.get("errors")
                if isinstance(details, list):
                    simulator_error_details = details
                stages = result.get("error_stage_counts")
                if isinstance(stages, dict):
                    simulator_error_stage_counts = {
                        str(key): int(value)
                        for key, value in stages.items()
                        if isinstance(value, (int, float))
                    }
            except Exception:
                if self.require_simulator:
                    raise

        candidate_values = np.asarray(
            [np.tanh(0.25 * state_value + value) for value in search_values],
            dtype=np.float32,
        )

        visits = np.zeros(len(candidates), dtype=np.float32)
        value_sums = np.zeros(len(candidates), dtype=np.float32)
        for _ in range(self.simulations):
            total_visits = float(visits.sum())
            scale = np.sqrt(total_visits + 1.0)
            q_values = np.divide(
                value_sums,
                visits,
                out=np.zeros_like(value_sums),
                where=visits > 0,
            )
            scores = q_values + self.cpuct * priors * scale / (1.0 + visits)
            index = int(np.argmax(scores))
            visits[index] += 1.0
            value_sums[index] += float(candidate_values[index])

        visit_probs = visits / max(float(visits.sum()), 1e-8)
        if self.temperature > 0:
            adjusted = np.power(visit_probs, 1.0 / self.temperature)
            adjusted = adjusted / max(float(adjusted.sum()), 1e-8)
            selected_index = int(self.rng.choice(len(candidates), p=adjusted))
        else:
            selected_index = int(np.argmax(visit_probs))

        selected_value = float(value_sums[selected_index] / max(visits[selected_index], 1.0))
        row = {
            "battle_tag": "",
            "turn": int(snapshot.get("turn") or 0),
            "side": side,
            "candidate_count": len(candidates),
            "forced_decision": len(candidates) <= 1,
            "candidate_messages": candidates,
            "state_features": state_features.astype(np.float32).tolist(),
            "action_features": action_features.astype(np.float32).tolist(),
            "selected_index": selected_index,
            "visit_probs": visit_probs.astype(np.float32).tolist(),
            "priors": priors.astype(np.float32).tolist(),
            "candidate_values": candidate_values.astype(np.float32).tolist(),
            "old_logprob": float(np.log(max(float(priors[selected_index]), 1e-8))),
            "old_value": selected_value,
            "simulator_used": simulator_used,
            "simulator_repairs": 0,
            "simulator_errors": simulator_errors,
            "simulator_skipped_branches": simulator_skipped,
            "simulator_error_details": simulator_error_details,
            "simulator_error_stage_counts": simulator_error_stage_counts,
            "selected_message": candidates[selected_index],
        }
        return OfflineSearchResult(choice=candidates[selected_index], row=row)


def _checkpoint_for(args) -> Path | None:
    checkpoint = args.output_dir / "best.pt"
    if checkpoint.exists():
        return checkpoint
    if args.init_checkpoint and args.init_checkpoint.exists():
        return args.init_checkpoint
    return None


def _build_model(args) -> AlphaZeroPolicyValueNet:
    checkpoint = _checkpoint_for(args)
    if checkpoint is not None:
        print(f"Loaded rollout policy checkpoint: {checkpoint}")
        return load_checkpoint(checkpoint, device=args.device)
    print("Initialized new rollout policy model.")
    return AlphaZeroPolicyValueNet().to(args.device)


def _team_order(rng: random.Random, team_text: str, mode: str) -> str:
    members = [block for block in team_text.split("\n\n") if block.strip()]
    count = max(1, len(members))
    size = min(4, count)
    if mode == "random" and count >= size:
        order = rng.sample(range(1, count + 1), size)
    else:
        order = list(range(1, size + 1))
    return "team " + "".join(str(item) for item in order)


def _baseline_choice(kind: str, legal: list[str], rng: random.Random) -> str:
    if not legal:
        return ""
    if kind == "random":
        return rng.choice(legal)
    if kind == "greedy":
        return legal[0]
    raise ValueError(f"Unsupported offline baseline opponent: {kind}")


def _outcome_for_side(snapshot: dict[str, Any], side: str) -> float:
    winner = snapshot.get("winner_side") or ""
    if winner == side:
        return 1.0
    if winner in {"p1", "p2"}:
        return -1.0
    return 0.0


def collect_offline_rollouts(
    args,
    *,
    iteration: int,
    append_jsonl: AppendJsonl,
    diagnostics_path: Path | None,
    compact_diagnostic: CompactDiagnostic,
) -> int:
    battle_format = resolve_format(args.format)
    if not should_use_team(battle_format):
        raise ValueError("Offline self-play currently requires a fixed team format")
    team_text = load_team(args.team)
    opponent_kind = args.opponent_for_iteration
    client = OfflineShowdownClient(args.simulator_url, timeout=args.simulator_timeout)
    model = _build_model(args)
    searcher = OfflineAlphaZeroSearch(
        model=model,
        client=client,
        simulations=args.mcts_simulations,
        depth=args.mcts_depth,
        max_candidates=args.max_candidates,
        cpuct=args.cpuct,
        temperature=args.temperature,
        device=args.device,
        simulator_max_choices=args.simulator_max_choices,
        simulator_opponent_policy=args.simulator_opponent_policy,
        simulator_robust_worst_weight=args.simulator_robust_worst_weight,
        require_simulator=args.require_simulator,
    )

    print(
        f"Collecting {args.self_play_games} offline battle(s): "
        f"alphazero_mcts vs {opponent_kind} on {battle_format}"
    )
    written = 0
    real_simulated = 0
    fallback_simulated = 0
    sim_errors = 0
    skipped_branches = 0
    diagnostics_written = 0
    start = time.time()

    for game_index in range(args.self_play_games):
        rng = random.Random((iteration * 1000003) + game_index + int(args.offline_seed))
        seed = [
            rng.randrange(1, 0xFFFFFFFF),
            rng.randrange(1, 0xFFFFFFFF),
            rng.randrange(1, 0xFFFFFFFF),
            rng.randrange(1, 0xFFFFFFFF),
        ]
        snapshot = client.start(
            battle_format=battle_format,
            team_text=team_text,
            seed=seed,
            team_choice_p1=_team_order(rng, team_text, args.offline_team_preview),
            team_choice_p2=_team_order(rng, team_text, args.offline_team_preview),
            max_choices=args.max_candidates,
        )
        battle_tag = f"offline-{iteration}-{game_index + 1}"
        pending_rows: list[dict[str, Any]] = []

        for _ in range(max(1, int(args.offline_max_turns))):
            if snapshot.get("ended"):
                break
            legal = snapshot.get("legal") or {}
            choices: dict[str, str] = {}

            p1_legal = list(legal.get("p1") or [])
            if p1_legal:
                result = searcher.search(snapshot, "p1")
                if result is not None:
                    choices["p1"] = result.choice
                    row = dict(result.row)
                    row["battle_tag"] = battle_tag
                    pending_rows.append(row)

            p2_legal = list(legal.get("p2") or [])
            if p2_legal:
                if opponent_kind == "self":
                    result = searcher.search(snapshot, "p2")
                    if result is not None:
                        choices["p2"] = result.choice
                        row = dict(result.row)
                        row["battle_tag"] = battle_tag
                        pending_rows.append(row)
                else:
                    choices["p2"] = _baseline_choice(opponent_kind, p2_legal, rng)

            if not choices:
                break
            snapshot = client.choose(
                state=snapshot["state"],
                choices=choices,
                max_choices=args.max_candidates,
            )

        for row in pending_rows:
            row.update(
                {
                    "schema_version": "az-rollout-offline-v1",
                    "iteration": iteration,
                    "opponent": opponent_kind,
                    "outcome": _outcome_for_side(snapshot, row.get("side", "p1")),
                    "format": battle_format,
                    "mcts_depth": args.mcts_depth,
                    "mcts_simulations": args.mcts_simulations,
                    "simulator_opponent_policy": args.simulator_opponent_policy,
                    "simulator_robust_worst_weight": args.simulator_robust_worst_weight,
                    "offline_seed": seed,
                    "offline_winner_side": snapshot.get("winner_side", ""),
                }
            )
            rollout_row = dict(row)
            rollout_row.pop("simulator_error_details", None)
            rollout_row.pop("simulator_error_stage_counts", None)
            append_jsonl(args.rollout_path, rollout_row)
            if row.get("simulator_used"):
                real_simulated += 1
            elif args.mcts_depth >= 2:
                fallback_simulated += 1
            sim_errors += int(row.get("simulator_errors") or 0)
            skipped_branches += int(row.get("simulator_skipped_branches") or 0)
            if diagnostics_path and int(row.get("simulator_errors") or 0) > 0:
                append_jsonl(
                    diagnostics_path,
                    compact_diagnostic(
                        row,
                        iteration=iteration,
                        opponent=opponent_kind,
                        error_limit=args.simulator_diagnostics_error_limit,
                    ),
                )
                diagnostics_written += 1
            written += 1

        if (game_index + 1) % max(1, int(args.offline_progress_games)) == 0:
            print(
                f"  Offline rollout progress: {game_index + 1}/{args.self_play_games} "
                f"battle(s), {written} decisions",
                flush=True,
            )

    elapsed = time.time() - start
    print(f"Collected offline rollout decisions: {written} in {elapsed:.1f}s")
    if args.mcts_depth >= 2:
        print(
            f"Offline simulator stats: real={real_simulated} fallback={fallback_simulated} "
            f"errors={sim_errors} skipped_branches={skipped_branches}"
        )
        if diagnostics_written:
            print(f"Simulator diagnostics: wrote {diagnostics_written} rows to {diagnostics_path}")
    print(f"Appended rollouts to {args.rollout_path}")
    return written
