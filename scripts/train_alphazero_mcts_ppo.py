"""AlphaZero-style loop: MCTS rollouts on Showdown + PPO update.

Typical flow:
  1. pretrain from data/replays/..._double_decisions.jsonl
  2. collect local battles with AlphaZero MCTS
  3. update the same policy/value network with PPO + MCTS visit targets
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import sys
import time
import uuid
from collections import Counter, deque
from pathlib import Path
from typing import Any

import torch
from poke_env import AccountConfiguration, MaxBasePowerPlayer, RandomPlayer
from poke_env.player.player import handle_threaded_coroutines

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from login import build_server_config, load_team, should_use_team
from src.alphazero.features import ACTION_FEATURE_SIZE, STATE_FEATURE_SIZE
from src.alphazero.network import AlphaZeroPolicyValueNet, load_checkpoint, save_checkpoint
from src.alphazero.offline_selfplay import collect_offline_rollouts
from src.alphazero.player import AlphaZeroMCTSPlayer
from src.alphazero.showdown_simulator import ShowdownSimulationTracker, attach_simulation_tracking
from src.format_resolver import resolve_format
from src.metrics.common import add_report_args, write_alphazero_report


def anonymous_account(prefix: str) -> AccountConfiguration:
    return AccountConfiguration(f"{prefix}{uuid.uuid4().hex[:8]}", None)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def memory_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "VmPeak:")):
                key, value = line.split(":", 1)
                snapshot[key.lower()] = value.strip()
    if torch.cuda.is_available():
        snapshot["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
        snapshot["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 1)
    return snapshot


def format_memory(snapshot: dict[str, Any] | None = None) -> str:
    snapshot = snapshot or memory_snapshot()
    if not snapshot:
        return "mem=n/a"
    preferred = ["vmrss", "vmhwm", "vmsize", "vmpeak", "cuda_allocated_mb", "cuda_reserved_mb"]
    parts = [f"{key}={snapshot[key]}" for key in preferred if key in snapshot]
    return "mem=" + " ".join(parts)


def append_training_event(args, event: str, **fields: Any) -> None:
    path = getattr(args, "training_log_path", None)
    if not path:
        return
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
        "memory": memory_snapshot(),
    }
    row.update(fields)
    append_jsonl(path, row)


def diagnostics_path_for(args) -> Path | None:
    path = args.simulator_diagnostics_path
    if path is not None:
        return path
    if not args.rollout_path:
        return None
    name = f"{args.rollout_path.stem}.simulator_diagnostics.jsonl"
    return args.rollout_path.with_name(name)


def compact_simulator_diagnostic(
    row: dict[str, Any],
    *,
    iteration: int,
    opponent: str,
    error_limit: int,
) -> dict[str, Any]:
    details = row.get("simulator_error_details") or []
    raw_stage_counts = row.get("simulator_error_stage_counts") or {}
    if isinstance(raw_stage_counts, dict):
        stage_counts = Counter({str(key): int(value) for key, value in raw_stage_counts.items()})
    else:
        stage_counts = Counter(
            str(item.get("stage") or "unknown")
            for item in details
            if isinstance(item, dict)
        )
    return {
        "schema_version": "az-simdiag-v2",
        "iteration": iteration,
        "opponent": opponent,
        "battle_tag": row.get("battle_tag", ""),
        "turn": row.get("turn", 0),
        "candidate_count": row.get("candidate_count", 0),
        "selected_index": row.get("selected_index", 0),
        "selected_message": row.get("selected_message", ""),
        "simulator_used": bool(row.get("simulator_used")),
        "simulator_repairs": int(row.get("simulator_repairs") or 0),
        "simulator_errors": int(row.get("simulator_errors") or 0),
        "simulator_skipped_branches": int(row.get("simulator_skipped_branches") or 0),
        "stage_counts": dict(stage_counts),
        "error_samples": details[:error_limit],
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_jsonl_tail_offsets(path: Path, max_rows: int) -> list[int]:
    if not path.exists():
        return []
    if max_rows and max_rows > 0:
        offsets: deque[int] | list[int] = deque(maxlen=max_rows)
    else:
        offsets = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return list(offsets)


def iter_jsonl_offsets(path: Path, offsets: list[int]):
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            line = handle.readline()
            if line.strip():
                yield json.loads(line)


async def close_player(player) -> None:
    ps_client = getattr(player, "ps_client", None)
    if ps_client is not None and hasattr(ps_client, "stop_listening"):
        try:
            await handle_threaded_coroutines(ps_client.stop_listening(), ps_client.loop)
        except AttributeError:
            pass


def make_baseline_player(kind: str, *, battle_format: str, team: str | None, server_configuration):
    kwargs = dict(
        account_configuration=anonymous_account(kind),
        battle_format=battle_format,
        team=team,
        server_configuration=server_configuration,
        max_concurrent_battles=1,
    )
    if kind == "random":
        return RandomPlayer(**kwargs)
    if kind == "greedy":
        return MaxBasePowerPlayer(**kwargs)
    raise ValueError(f"Unknown opponent: {kind}")


def make_alphazero_player(
    args,
    *,
    battle_format: str,
    team: str | None,
    server_configuration,
    record: bool,
    simulation_tracker: ShowdownSimulationTracker | None = None,
):
    checkpoint = args.output_dir / "best.pt"
    if not checkpoint.exists():
        checkpoint = args.init_checkpoint if args.init_checkpoint else None
    return AlphaZeroMCTSPlayer(
        account_configuration=anonymous_account("az"),
        battle_format=battle_format,
        team=team,
        server_configuration=server_configuration,
        max_concurrent_battles=1,
        checkpoint_path=checkpoint,
        simulations=args.mcts_simulations,
        search_depth=args.mcts_depth,
        max_candidates=args.max_candidates,
        cpuct=args.cpuct,
        temperature=args.temperature,
        heuristic_weight=args.heuristic_weight,
        depth2_weight=args.depth2_weight,
        showdown_simulator_url=args.simulator_url,
        live_state_url=args.live_state_url,
        simulation_tracker=simulation_tracker,
        simulator_timeout=args.simulator_timeout,
        simulator_max_choices=args.simulator_max_choices,
        simulator_opponent_policy=args.simulator_opponent_policy,
        simulator_robust_worst_weight=args.simulator_robust_worst_weight,
        require_showdown_simulator=args.require_simulator,
        device=args.device,
        record_decisions=record,
    )


def opponent_for_iteration(args, iteration: int) -> str:
    if args.opponent_cycle:
        cycle = [item.strip() for item in args.opponent_cycle.split(",") if item.strip()]
        if cycle:
            return cycle[(iteration - 1) % len(cycle)]
    return args.opponent


async def collect_rollouts(args, iteration: int) -> int:
    battle_format = resolve_format(args.format)
    server_cfg = build_server_config(args.server)
    team = load_team(args.team) if should_use_team(battle_format) else None
    opponent_kind = opponent_for_iteration(args, iteration)
    simulation_tracker = None
    if args.mcts_depth >= 2 and team is not None and not args.live_state_url:
        simulation_tracker = ShowdownSimulationTracker(
            battle_format=battle_format,
            team_text=team,
        )

    player = make_alphazero_player(
        args,
        battle_format=battle_format,
        team=team,
        server_configuration=server_cfg,
        record=True,
        simulation_tracker=simulation_tracker,
    )
    if opponent_kind == "self":
        opponent = make_alphazero_player(
            args,
            battle_format=battle_format,
            team=team,
            server_configuration=server_cfg,
            record=False,
            simulation_tracker=simulation_tracker,
        )
    else:
        opponent = make_baseline_player(
            opponent_kind,
            battle_format=battle_format,
            team=team,
            server_configuration=server_cfg,
        )
    if simulation_tracker is not None:
        attach_simulation_tracking(player, simulation_tracker)
        attach_simulation_tracking(opponent, simulation_tracker)

    print(
        f"Collecting {args.self_play_games} battle(s): alphazero_mcts vs {opponent_kind} "
        f"on {battle_format}"
    )
    try:
        runner = player.battle_against(opponent, n_battles=args.self_play_games)
        if args.battle_timeout and args.battle_timeout > 0:
            await asyncio.wait_for(runner, timeout=args.battle_timeout)
        else:
            await runner
    finally:
        await close_player(player)
        await close_player(opponent)

    written = 0
    real_simulated = 0
    fallback_simulated = 0
    repaired = 0
    sim_errors = 0
    skipped_branches = 0
    stage_counts: Counter[str] = Counter()
    diagnostic_path = diagnostics_path_for(args)
    diagnostics_written = 0
    for row in player.decision_log:
        battle = player.battles.get(row["battle_tag"])
        if battle is None or not battle.finished:
            outcome = 0.0
        elif battle.won:
            outcome = 1.0
        elif getattr(battle, "lost", False):
            outcome = -1.0
        else:
            outcome = 0.0
        row.update(
            {
                "schema_version": "az-rollout-v2",
                "iteration": iteration,
                "opponent": opponent_kind,
                "outcome": outcome,
                "format": battle_format,
                "mcts_depth": args.mcts_depth,
                "mcts_simulations": args.mcts_simulations,
                "simulator_opponent_policy": args.simulator_opponent_policy,
                "simulator_robust_worst_weight": args.simulator_robust_worst_weight,
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
        if int(row.get("simulator_repairs") or 0) > 0:
            repaired += 1
        sim_errors += int(row.get("simulator_errors") or 0)
        skipped_branches += int(row.get("simulator_skipped_branches") or 0)
        raw_stage_counts = row.get("simulator_error_stage_counts") or {}
        if isinstance(raw_stage_counts, dict):
            for stage, count in raw_stage_counts.items():
                stage_counts[str(stage)] += int(count)
        else:
            for item in row.get("simulator_error_details") or []:
                if isinstance(item, dict):
                    stage_counts[str(item.get("stage") or "unknown")] += 1
        if diagnostic_path and (
            int(row.get("simulator_errors") or 0) > 0
            or int(row.get("simulator_repairs") or 0) > 0
        ):
            append_jsonl(
                diagnostic_path,
                compact_simulator_diagnostic(
                    row,
                    iteration=iteration,
                    opponent=opponent_kind,
                    error_limit=args.simulator_diagnostics_error_limit,
                ),
            )
            diagnostics_written += 1
        written += 1
    print(f"Collected rollout decisions: {written}")
    if args.mcts_depth >= 2:
        print(
            f"Simulator stats: real={real_simulated} fallback={fallback_simulated} "
            f"repaired={repaired} errors={sim_errors} skipped_branches={skipped_branches}"
        )
        if stage_counts:
            top_stages = ", ".join(
                f"{stage}={count}" for stage, count in stage_counts.most_common(5)
            )
            print(f"Simulator error stages: {top_stages}")
        if diagnostics_written:
            print(f"Simulator diagnostics: wrote {diagnostics_written} rows to {diagnostic_path}")
    print(f"Appended rollouts to {args.rollout_path}")
    return written


def collate_rollouts(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_candidates = max(len(row["action_features"]) for row in batch)
    batch_size = len(batch)
    states = torch.zeros((batch_size, STATE_FEATURE_SIZE), dtype=torch.float32)
    actions = torch.zeros((batch_size, max_candidates, ACTION_FEATURE_SIZE), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    selected = torch.zeros((batch_size,), dtype=torch.long)
    visit_probs = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    old_logprob = torch.zeros((batch_size,), dtype=torch.float32)
    old_value = torch.zeros((batch_size,), dtype=torch.float32)
    outcomes = torch.zeros((batch_size,), dtype=torch.float32)
    value_targets = torch.zeros((batch_size,), dtype=torch.float32)
    sample_weights = torch.ones((batch_size,), dtype=torch.float32)

    for i, row in enumerate(batch):
        candidates = row["action_features"]
        count = len(candidates)
        states[i] = torch.tensor(row["state_features"], dtype=torch.float32)
        actions[i, :count] = torch.tensor(candidates, dtype=torch.float32)
        mask[i, :count] = True
        selected[i] = min(int(row["selected_index"]), count - 1)
        visits = torch.tensor(row.get("visit_probs") or [1.0 / count] * count, dtype=torch.float32)
        visit_probs[i, :count] = visits / visits.sum().clamp_min(1e-8)
        old_logprob[i] = float(row.get("old_logprob", 0.0))
        old_value[i] = float(row.get("old_value", 0.0))
        outcomes[i] = float(row.get("outcome", 0.0))
        value_targets[i] = float(row.get("value_target", row.get("outcome", 0.0)))
        sample_weights[i] = float(row.get("sample_weight", 1.0))

    return {
        "states": states,
        "actions": actions,
        "mask": mask,
        "selected": selected,
        "visit_probs": visit_probs,
        "old_logprob": old_logprob,
        "old_value": old_value,
        "outcomes": outcomes,
        "value_targets": value_targets,
        "sample_weights": sample_weights,
    }


def make_batches(rows: list[dict[str, Any]], batch_size: int, shuffle: bool) -> list[list[dict[str, Any]]]:
    if shuffle:
        generator = torch.Generator().manual_seed(int(time.time()))
        order = torch.randperm(len(rows), generator=generator).tolist()
        rows = [rows[i] for i in order]
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(float(value), high))


def is_trainable_rollout_row(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("action_features"))
        and bool(row.get("state_features"))
        and len(row["state_features"]) == STATE_FEATURE_SIZE
        and all(len(features) == ACTION_FEATURE_SIZE for features in row["action_features"])
    )


def empty_rollout_stats() -> dict[str, float]:
    return {
        "forced": 0.0,
        "repaired": 0.0,
        "real_simulated": 0.0,
        "fallback_simulated": 0.0,
        "sim_error_rows": 0.0,
        "effective_weight": 0.0,
        "filtered_fallback": 0.0,
    }


def add_rollout_stats(total: dict[str, float], update: dict[str, float]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + float(value)


def prepare_rollout_row(args, row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, float]]:
    row = dict(row)
    stats = empty_rollout_stats()
    simulator_used = bool(row.get("simulator_used"))
    if args.mcts_depth >= 2:
        if simulator_used:
            stats["real_simulated"] += 1
        else:
            stats["fallback_simulated"] += 1
            if args.require_simulator:
                stats["filtered_fallback"] += 1
                return None, stats

    candidate_count = int(row.get("candidate_count") or len(row.get("action_features") or []))
    is_forced = bool(row.get("forced_decision")) or candidate_count <= 1
    repairs = int(row.get("simulator_repairs") or 0)
    sim_errors = int(row.get("simulator_errors") or 0)

    weight = 1.0
    if is_forced:
        stats["forced"] += 1
        weight *= args.forced_weight
    if repairs > 0:
        stats["repaired"] += 1
        weight *= args.repair_weight
    if sim_errors > 0:
        stats["sim_error_rows"] += 1
        weight *= args.simulator_error_weight

    outcome = _clamp(float(row.get("outcome", 0.0)), -1.0, 1.0)
    search_value = _clamp(float(row.get("old_value", 0.0)), -1.0, 1.0)
    mix = _clamp(args.value_search_weight, 0.0, 1.0)
    row["sample_weight"] = max(0.0, float(weight))
    row["value_target"] = _clamp((1.0 - mix) * outcome + mix * search_value, -1.0, 1.0)
    stats["effective_weight"] += row["sample_weight"]
    return row, stats


def prepare_rollout_rows(args, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    prepared = []
    stats = empty_rollout_stats()
    for row in rows:
        prepared_row, row_stats = prepare_rollout_row(args, row)
        add_rollout_stats(stats, row_stats)
        if prepared_row is not None:
            prepared.append(prepared_row)
    return prepared, stats


def collect_training_offsets(args) -> tuple[list[int], dict[str, float]]:
    window_offsets = collect_jsonl_tail_offsets(args.rollout_path, args.train_window)
    stats = empty_rollout_stats()
    stats["window_rows"] = float(len(window_offsets))
    stats["invalid_rows"] = 0.0
    valid_offsets: list[int] = []
    for offset, row in zip(window_offsets, iter_jsonl_offsets(args.rollout_path, window_offsets)):
        if not is_trainable_rollout_row(row):
            stats["invalid_rows"] += 1
            continue
        prepared_row, row_stats = prepare_rollout_row(args, row)
        add_rollout_stats(stats, row_stats)
        if prepared_row is not None:
            valid_offsets.append(offset)
    return valid_offsets, stats


def iter_prepared_batches(args, offsets: list[int], batch_size: int, shuffle: bool):
    if shuffle:
        generator = torch.Generator().manual_seed(int(time.time()))
        order = torch.randperm(len(offsets), generator=generator).tolist()
    else:
        order = list(range(len(offsets)))

    batch: list[dict[str, Any]] = []
    with args.rollout_path.open("rb") as handle:
        for index in order:
            handle.seek(offsets[index])
            line = handle.readline()
            if not line.strip():
                continue
            row = json.loads(line)
            if not is_trainable_rollout_row(row):
                continue
            prepared_row, _ = prepare_rollout_row(args, row)
            if prepared_row is None:
                continue
            batch.append(prepared_row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def train_from_rollouts(args, *, iteration: int) -> None:
    offsets, row_stats = collect_training_offsets(args)
    if not offsets:
        print("No rollout rows available for PPO update.")
        append_training_event(
            args,
            "ppo_skipped",
            iteration=iteration,
            reason="no_rows",
            rollout_path=str(args.rollout_path),
            window_rows=int(row_stats.get("window_rows", 0)),
            invalid_rows=int(row_stats.get("invalid_rows", 0)),
            filtered_fallback=int(row_stats.get("filtered_fallback", 0)),
        )
        return

    device = torch.device(args.device)
    checkpoint = args.output_dir / "best.pt"
    if checkpoint.exists():
        model = load_checkpoint(checkpoint, device=device)
        print(f"Loaded checkpoint: {checkpoint}")
    elif args.init_checkpoint and args.init_checkpoint.exists():
        model = load_checkpoint(args.init_checkpoint, device=device)
        print(f"Loaded init checkpoint: {args.init_checkpoint}")
    else:
        model = AlphaZeroPolicyValueNet().to(device)
        print("Initialized new AlphaZero model.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()
    print(
        f"Training samples: {len(offsets)} "
        f"window_rows={int(row_stats['window_rows'])} "
        f"effective_weight={row_stats['effective_weight']:.1f} "
        f"forced={int(row_stats['forced'])} repaired={int(row_stats['repaired'])} "
        f"real_sim={int(row_stats['real_simulated'])} "
        f"fallback_sim={int(row_stats['fallback_simulated'])} "
        f"sim_error_rows={int(row_stats['sim_error_rows'])} "
        f"invalid_rows={int(row_stats['invalid_rows'])} "
        f"filtered_fallback={int(row_stats['filtered_fallback'])} "
        f"{format_memory()}"
    )
    append_training_event(
        args,
        "ppo_start",
        iteration=iteration,
        samples=len(offsets),
        window_rows=int(row_stats["window_rows"]),
        effective_weight=round(row_stats["effective_weight"], 3),
        forced=int(row_stats["forced"]),
        repaired=int(row_stats["repaired"]),
        real_sim=int(row_stats["real_simulated"]),
        fallback_sim=int(row_stats["fallback_simulated"]),
        sim_error_rows=int(row_stats["sim_error_rows"]),
        invalid_rows=int(row_stats["invalid_rows"]),
        filtered_fallback=int(row_stats["filtered_fallback"]),
    )

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        totals = {"loss": 0.0, "ppo": 0.0, "value": 0.0, "mcts_ce": 0.0, "entropy": 0.0}
        seen = 0
        for batch_rows in iter_prepared_batches(args, offsets, args.batch_size, shuffle=True):
            batch = collate_rollouts(batch_rows)
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            mask = batch["mask"].to(device)
            selected = batch["selected"].to(device)
            visit_probs = batch["visit_probs"].to(device)
            old_logprob = batch["old_logprob"].to(device)
            old_value = batch["old_value"].to(device)
            value_targets = batch["value_targets"].to(device)
            sample_weights = batch["sample_weights"].to(device)

            logits, values = model(states, actions, mask)
            log_probs = torch.log_softmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            new_logprob = log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)
            advantages = (value_targets - old_value).detach()
            ratio = torch.exp(new_logprob - old_logprob)
            clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
            ppo_loss = weighted_mean(
                -torch.minimum(ratio * advantages, clipped * advantages),
                sample_weights,
            )
            value_loss = weighted_mean((values - value_targets).pow(2), sample_weights)
            mcts_ce = weighted_mean(-(visit_probs * log_probs).sum(dim=1), sample_weights)
            entropy = weighted_mean(-(probs * log_probs).sum(dim=1), sample_weights)
            loss = (
                ppo_loss
                + args.value_weight * value_loss
                + args.mcts_weight * mcts_ce
                - args.entropy_weight * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            count = len(batch_rows)
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["ppo"] += float(ppo_loss.detach().cpu()) * count
            totals["value"] += float(value_loss.detach().cpu()) * count
            totals["mcts_ce"] += float(mcts_ce.detach().cpu()) * count
            totals["entropy"] += float(entropy.detach().cpu()) * count
            seen += count
            del batch, states, actions, mask, selected, visit_probs, old_logprob
            del old_value, value_targets, sample_weights, logits, values
            del log_probs, probs, new_logprob, advantages, ratio, clipped
            del ppo_loss, value_loss, mcts_ce, entropy, loss

        elapsed = time.time() - start
        if seen <= 0:
            print("No rollout rows were yielded during PPO epoch.")
            append_training_event(
                args,
                "ppo_epoch_skipped",
                iteration=iteration,
                epoch=epoch,
                reason="no_batches",
            )
            return
        epoch_metrics = {
            "loss": totals["loss"] / seen,
            "ppo": totals["ppo"] / seen,
            "value": totals["value"] / seen,
            "mcts_ce": totals["mcts_ce"] / seen,
            "entropy": totals["entropy"] / seen,
        }
        print(
            f"ppo_epoch={epoch:03d} "
            f"loss={epoch_metrics['loss']:.4f} "
            f"ppo={epoch_metrics['ppo']:.4f} "
            f"value={epoch_metrics['value']:.4f} "
            f"mcts_ce={epoch_metrics['mcts_ce']:.4f} "
            f"entropy={epoch_metrics['entropy']:.4f} "
            f"time={elapsed:.1f}s "
            f"{format_memory()}",
            flush=True,
        )
        append_training_event(
            args,
            "ppo_epoch",
            iteration=iteration,
            epoch=epoch,
            samples=seen,
            elapsed_sec=round(elapsed, 3),
            **{key: round(value, 6) for key, value in epoch_metrics.items()},
        )
        gc.collect()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output_dir / "best.pt", model, extra={"trained_rows": len(offsets)})
    print(f"Saved AlphaZero MCTS+PPO checkpoint to {args.output_dir}")
    append_training_event(
        args,
        "checkpoint_saved",
        iteration=iteration,
        output_dir=str(args.output_dir),
        trained_rows=len(offsets),
    )
    del model, optimizer
    gc.collect()


async def main_async(args) -> None:
    for iteration in range(1, args.iterations + 1):
        opponent_kind = opponent_for_iteration(args, iteration)
        args.opponent_for_iteration = opponent_kind
        print(f"\n=== AlphaZero iteration {iteration}/{args.iterations} ===")
        print(f"Curriculum opponent: {opponent_kind}")
        append_training_event(
            args,
            "iteration_start",
            iteration=iteration,
            total_iterations=args.iterations,
            opponent=opponent_kind,
        )
        if args.rollout_source == "offline":
            collected = collect_offline_rollouts(
                args,
                iteration=iteration,
                append_jsonl=append_jsonl,
                diagnostics_path=diagnostics_path_for(args),
                compact_diagnostic=compact_simulator_diagnostic,
            )
        else:
            collected = await collect_rollouts(args, iteration)
        append_training_event(
            args,
            "rollout_collected",
            iteration=iteration,
            opponent=opponent_kind,
            decisions=collected,
            rollout_path=str(args.rollout_path),
        )
        train_from_rollouts(args, iteration=iteration)
        append_training_event(
            args,
            "iteration_finished",
            iteration=iteration,
            opponent=opponent_kind,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AlphaZero-style MCTS + PPO")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--self-play-games", type=int, default=10)
    parser.add_argument(
        "--rollout-source",
        choices=["websocket", "offline"],
        default="websocket",
        help="Use websocket battles or exact offline Showdown simulator battles for rollout collection.",
    )
    parser.add_argument("--opponent", choices=["random", "greedy", "self"], default="random")
    parser.add_argument("--opponent-cycle", type=str, default="")
    parser.add_argument("--server", type=str, default="showdown:8000")
    parser.add_argument("--format", type=str, default="gen9vgc2026regi")
    parser.add_argument("--team", type=str, default="team.txt")
    parser.add_argument("--battle-timeout", type=float, default=0.0)
    parser.add_argument("--mcts-simulations", type=int, default=64)
    parser.add_argument("--mcts-depth", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=96)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--heuristic-weight", type=float, default=0.75)
    parser.add_argument("--depth2-weight", type=float, default=0.65)
    parser.add_argument("--simulator-url", type=str, default=os.environ.get("SHOWDOWN_SIMULATOR_URL", ""))
    parser.add_argument("--live-state-url", type=str, default=os.environ.get("SHOWDOWN_LIVE_STATE_URL", ""))
    parser.add_argument("--simulator-timeout", type=float, default=10.0)
    parser.add_argument("--simulator-max-choices", type=int, default=12)
    parser.add_argument(
        "--simulator-eval-candidates",
        type=int,
        default=0,
        help=(
            "Cantidad maxima de candidatos de la politica a evaluar con el simulador real. "
            "0 evalua todos; para D4 conviene usar un valor chico para evitar timeouts."
        ),
    )
    parser.add_argument(
        "--simulator-opponent-policy",
        choices=["minimax", "mean", "robust"],
        default="robust",
    )
    parser.add_argument("--simulator-robust-worst-weight", type=float, default=0.35)
    parser.add_argument(
        "--require-simulator",
        action="store_true",
        help="Fail instead of using heuristic fallback when --mcts-depth >= 2 and the real simulator is unavailable.",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--mcts-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--value-search-weight", type=float, default=0.35)
    parser.add_argument("--forced-weight", type=float, default=0.2)
    parser.add_argument("--repair-weight", type=float, default=0.6)
    parser.add_argument("--simulator-error-weight", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/alphazero_mcts_ppo"))
    parser.add_argument("--rollout-path", type=Path, default=Path("data/alphazero/rollouts.jsonl"))
    parser.add_argument("--train-window", type=int, default=5000)
    parser.add_argument(
        "--training-log-path",
        type=Path,
        default=Path("logs/alphazero_train_events.jsonl"),
        help="JSONL log for iteration, PPO, checkpoint and memory events.",
    )
    parser.add_argument("--offline-max-turns", type=int, default=60)
    parser.add_argument("--offline-seed", type=int, default=7)
    parser.add_argument(
        "--offline-team-preview",
        choices=["default", "random"],
        default="random",
        help="Team preview order policy for exact offline self-play.",
    )
    parser.add_argument("--offline-progress-games", type=int, default=5)
    parser.add_argument(
        "--simulator-diagnostics-path",
        type=Path,
        default=None,
        help="JSONL path for per-decision simulator diagnostics. Defaults next to rollout path.",
    )
    parser.add_argument("--simulator-diagnostics-error-limit", type=int, default=5)
    add_report_args(parser)
    return parser.parse_args()


def maybe_write_metrics_report(args, *, status: str) -> None:
    if args.no_metrics_report:
        return
    try:
        out = write_alphazero_report(
            training_log_path=args.training_log_path,
            rollout_path=args.rollout_path,
            output_root=args.metrics_report_dir,
            title=f"AlphaZero MCTS+PPO ({status})",
        )
    except Exception as exc:
        print(f"  [metrics] no se pudo generar reporte: {exc}")
        return
    print(f"\n  ✓ Reporte comparable generado en: {out}")
    print(f"    · {out / 'report.html'}")
    print(f"    · {out / 'report.json'}")
    print(f"    · {out / 'common_metrics.json'}")
    print(f"    · {out / 'league_stats.json'}")
    print(f"    · {out / 'plots'}")


def main() -> int:
    args = parse_args()
    append_training_event(
        args,
        "run_start",
        rollout_source=args.rollout_source,
        iterations=args.iterations,
        self_play_games=args.self_play_games,
        opponent=args.opponent,
        opponent_cycle=args.opponent_cycle,
        format=args.format,
        team=str(args.team),
        mcts_simulations=args.mcts_simulations,
        mcts_depth=args.mcts_depth,
        max_candidates=args.max_candidates,
        simulator_max_choices=args.simulator_max_choices,
        simulator_eval_candidates=args.simulator_eval_candidates,
        simulator_opponent_policy=args.simulator_opponent_policy,
        simulator_robust_worst_weight=args.simulator_robust_worst_weight,
        require_simulator=args.require_simulator,
        lr=args.lr,
        value_weight=args.value_weight,
        mcts_weight=args.mcts_weight,
        entropy_weight=args.entropy_weight,
        value_search_weight=args.value_search_weight,
        train_window=args.train_window,
        batch_size=args.batch_size,
        epochs=args.epochs,
        init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint else "",
        rollout_path=str(args.rollout_path),
        output_dir=str(args.output_dir),
    )
    try:
        asyncio.run(main_async(args))
    except BaseException as exc:
        append_training_event(
            args,
            "run_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        maybe_write_metrics_report(args, status="failed")
        raise
    append_training_event(args, "run_finished")
    maybe_write_metrics_report(args, status="finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
