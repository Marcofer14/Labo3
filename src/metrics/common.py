"""Shared metrics/reporting for PPO and AlphaZero-style runs.

The PPO branch already emits reports with a useful shape. This module maps
AlphaZero JSONL logs into the same high-level fields and also exposes a
``common_metrics`` block that both families can use for apples-to-apples
comparison.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num) / float(den) if den else default


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


_REPORT_CSS = """
body { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }
h1, h2, h3 { color: #222; }
table { border-collapse: collapse; margin: 1em 0; width: 100%; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
th { background: #f5f5f5; }
.stage { background: #eef; padding: 1em; border-radius: 6px; margin: 1em 0; }
.metric { display: inline-block; margin: 0.5em 1em; }
.metric .label { font-weight: bold; }
img { max-width: 100%; margin: 1em 0; border: 1px solid #ddd; }
code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }
"""


def _escape(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.4g}"
    return html.escape(str(value))


def _metric_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "n/a"
        return f"{value:.4g}"
    return html.escape(str(value))


def _html_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p>Sin datos.</p>"
    parts = ["<table><tr>"]
    parts.extend(f"<th>{html.escape(label)}</th>" for _, label in columns)
    parts.append("</tr>")
    for row in rows:
        parts.append("<tr>")
        for key, _ in columns:
            parts.append(f"<td>{_escape(row.get(key))}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    return "\n".join(parts)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield {"event": "_invalid_json", "line": line_no}
                continue
            if isinstance(row, dict):
                yield row


def _event_index(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "epochs": [],
            "started_at": None,
            "ended_at": None,
            "opponent": "",
            "events": [],
        }
    )
    for event in events:
        iteration = event.get("iteration")
        if iteration is None:
            continue
        try:
            iteration = int(iteration)
        except (TypeError, ValueError):
            continue
        item = indexed[iteration]
        item["events"].append(event.get("event"))
        event_name = event.get("event")
        if event_name == "iteration_start":
            item["started_at"] = event.get("ts")
            item["opponent"] = event.get("opponent", item["opponent"])
            item["total_iterations"] = event.get("total_iterations")
        elif event_name == "iteration_finished":
            item["ended_at"] = event.get("ts")
        elif event_name == "rollout_collected":
            item["rollout_decisions"] = int(event.get("decisions") or 0)
            item["rollout_path"] = event.get("rollout_path")
            item["opponent"] = event.get("opponent", item["opponent"])
        elif event_name == "ppo_start":
            item["ppo_start"] = event
        elif event_name == "ppo_epoch":
            item["epochs"].append(event)
        elif event_name == "checkpoint_saved":
            item["checkpoint"] = event
    return indexed


def _rollout_index(rollout_path: Path) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "decisions": 0,
            "battle_tags": set(),
            "battle_turns": defaultdict(int),
            "battle_winners": {},
            "battle_p1_outcomes": {},
            "opponents": Counter(),
            "candidate_count_sum": 0.0,
            "candidate_count_n": 0,
            "simulator": Counter(),
            "mcts_depth": None,
            "mcts_simulations": None,
        }
    )
    for row in iter_jsonl(rollout_path):
        if "iteration" not in row:
            continue
        try:
            iteration = int(row.get("iteration"))
        except (TypeError, ValueError):
            continue
        item = indexed[iteration]
        item["decisions"] += 1
        opponent = str(row.get("opponent") or "")
        if opponent:
            item["opponents"][opponent] += 1
        tag = str(row.get("battle_tag") or f"decision-{item['decisions']}")
        item["battle_tags"].add(tag)
        turn = int(row.get("turn") or 0)
        item["battle_turns"][tag] = max(item["battle_turns"][tag], turn)
        winner_side = row.get("offline_winner_side") or row.get("winner_side")
        if winner_side in {"p1", "p2"}:
            item["battle_winners"][tag] = winner_side
        if row.get("side", "p1") == "p1":
            try:
                item["battle_p1_outcomes"][tag] = float(row.get("outcome") or 0.0)
            except (TypeError, ValueError):
                pass
        candidate_count = row.get("candidate_count")
        if candidate_count is not None:
            item["candidate_count_sum"] += float(candidate_count)
            item["candidate_count_n"] += 1
        if row.get("simulator_used"):
            item["simulator"]["real"] += 1
        elif int(row.get("mcts_depth") or 0) >= 2:
            item["simulator"]["fallback"] += 1
        item["simulator"]["repairs"] += int(row.get("simulator_repairs") or 0)
        item["simulator"]["errors"] += int(row.get("simulator_errors") or 0)
        item["simulator"]["skipped_branches"] += int(row.get("simulator_skipped_branches") or 0)
        item["mcts_depth"] = row.get("mcts_depth", item["mcts_depth"])
        item["mcts_simulations"] = row.get("mcts_simulations", item["mcts_simulations"])
    return indexed


def _battle_results(item: dict[str, Any]) -> tuple[int, int, int]:
    wins = losses = draws = 0
    for tag in item["battle_tags"]:
        winner = item["battle_winners"].get(tag)
        if winner == "p1":
            wins += 1
        elif winner == "p2":
            losses += 1
        else:
            outcome = float(item["battle_p1_outcomes"].get(tag, 0.0))
            if outcome > 0:
                wins += 1
            elif outcome < 0:
                losses += 1
            else:
                draws += 1
    return wins, losses, draws


def _duration_seconds(start_ts: str | None, end_ts: str | None) -> float | None:
    start = _parse_ts(start_ts)
    end = _parse_ts(end_ts)
    if start and end:
        return max(0.0, (end - start).total_seconds())
    return None


def build_alphazero_report(
    *,
    training_log_path: Path,
    rollout_path: Path,
    title: str = "AlphaZero MCTS+PPO",
) -> dict[str, Any]:
    events = list(iter_jsonl(training_log_path))
    run_start = next((event for event in events if event.get("event") == "run_start"), {})
    run_failed = next((event for event in reversed(events) if event.get("event") == "run_failed"), None)
    run_finished = next((event for event in reversed(events) if event.get("event") == "run_finished"), None)
    by_event = _event_index(events)
    by_rollout = _rollout_index(rollout_path)

    stages: list[dict[str, Any]] = []
    cumulative_decisions = 0
    totals = Counter()
    reward_components = Counter()
    all_lengths: list[float] = []
    all_rewards: list[float] = []
    all_epochs: list[dict[str, Any]] = []
    iterations = sorted(set(by_event) | set(by_rollout))
    for iteration in iterations:
        event_item = by_event.get(iteration, {})
        rollout_item = by_rollout.get(iteration, {})
        decisions = int(rollout_item.get("decisions") or event_item.get("rollout_decisions") or 0)
        wins, losses, draws = _battle_results(rollout_item) if rollout_item else (0, 0, 0)
        games = wins + losses + draws
        rewards = [1.0] * wins + [-1.0] * losses + [0.0] * draws
        lengths = list((rollout_item.get("battle_turns") or {}).values())
        epochs = list(event_item.get("epochs") or [])
        all_epochs.extend(epochs)
        final_epoch = epochs[-1] if epochs else {}
        start = cumulative_decisions
        cumulative_decisions += decisions
        totals.update({"games": games, "wins": wins, "losses": losses, "draws": draws})
        totals["decisions"] += decisions
        totals["updates"] += len(epochs)
        reward_components["a.win"] += wins * 15.0
        reward_components["a.loss"] += losses * -15.0
        if draws:
            reward_components["a.draw"] += 0.0
        all_lengths.extend(float(value) for value in lengths)
        all_rewards.extend(rewards)
        duration_sec = _duration_seconds(event_item.get("started_at"), event_item.get("ended_at"))
        rollout_opponents = rollout_item.get("opponents") or Counter()
        rollout_opponent = rollout_opponents.most_common(1)[0][0] if rollout_opponents else ""
        opponent = event_item.get("opponent") or rollout_opponent
        stage = {
            "stage": iteration,
            "started": start,
            "ended": cumulative_decisions,
            "duration": decisions,
            "transition_reason": "iteration_finished"
            if "iteration_finished" in event_item.get("events", [])
            else "incomplete",
            "n_episodes": games,
            "win_rate": _safe_div(wins, games),
            "avg_reward": _mean(rewards),
            "avg_episode_length": _mean(float(value) for value in lengths),
            "final_value_loss": final_epoch.get("value"),
            "final_policy_loss": final_epoch.get("ppo"),
            "final_loss": final_epoch.get("loss"),
            "reward_breakdown": {
                "a.win": wins * 15.0,
                "a.loss": losses * -15.0,
            },
            "n_updates": len(epochs),
            "opponent": opponent,
            "decisions": decisions,
            "wall_time_sec": duration_sec,
            "avg_candidate_count": _safe_div(
                rollout_item.get("candidate_count_sum", 0.0),
                rollout_item.get("candidate_count_n", 0),
            ),
            "simulator": dict(rollout_item.get("simulator") or {}),
        }
        stages.append(stage)

    final_epoch = all_epochs[-1] if all_epochs else {}
    common_metrics = {
        "schema_version": "common-v1",
        "model_family": "alphazero_mcts_ppo",
        "algorithm": "alphazero_mcts_ppo",
        "total_games": int(totals["games"]),
        "wins": int(totals["wins"]),
        "losses": int(totals["losses"]),
        "draws": int(totals["draws"]),
        "win_rate": _safe_div(totals["wins"], totals["games"]),
        "avg_episode_length": _mean(all_lengths),
        "avg_reward": _mean(all_rewards),
        "final_loss": final_epoch.get("loss"),
        "final_policy_loss": final_epoch.get("ppo"),
        "final_value_loss": final_epoch.get("value"),
        "final_entropy": final_epoch.get("entropy"),
        "total_updates": int(totals["updates"]),
        "total_decisions": int(totals["decisions"]),
        "total_iterations": int(run_start.get("iterations") or len(iterations)),
        "completed_iterations": sum(1 for stage in stages if stage["transition_reason"] == "iteration_finished"),
        "last_iteration": max(iterations) if iterations else 0,
        "training_log_path": str(training_log_path),
        "rollout_path": str(rollout_path),
    }
    training_config = {
        key: value
        for key, value in run_start.items()
        if key not in {"event", "ts", "memory"}
    }
    extra_meta = {
        "title": title,
        "status": "failed" if run_failed else "finished" if run_finished else "partial",
        "failure": run_failed,
        "loss_names": {
            "final_policy_loss": "AlphaZero PPO surrogate loss",
            "final_value_loss": "value MSE",
            "final_loss": "combined PPO + value + MCTS CE - entropy loss",
        },
    }
    return {
        "timestamp": _now_stamp(),
        "stages": stages,
        "training_config": training_config,
        "extra_meta": extra_meta,
        "common_metrics": common_metrics,
        "league": [],
        "reward_breakdown": dict(reward_components),
        "loss_history": [
            {
                "iteration": int(event.get("iteration") or 0),
                "epoch": int(event.get("epoch") or 0),
                "loss": event.get("loss"),
                "policy_loss": event.get("ppo"),
                "value_loss": event.get("value"),
                "mcts_ce": event.get("mcts_ce"),
                "entropy": event.get("entropy"),
            }
            for event in all_epochs
        ],
    }


def build_common_from_ppo_report(report: dict[str, Any], *, source: str = "") -> dict[str, Any]:
    stages = list(report.get("stages") or [])
    total_games = sum(int(stage.get("n_episodes") or 0) for stage in stages)
    wins = sum(round(float(stage.get("win_rate") or 0.0) * int(stage.get("n_episodes") or 0)) for stage in stages)
    losses = max(0, total_games - wins)
    avg_len = _safe_div(
        sum(float(stage.get("avg_episode_length") or 0.0) * int(stage.get("n_episodes") or 0) for stage in stages),
        total_games,
    )
    avg_reward = _safe_div(
        sum(float(stage.get("avg_reward") or 0.0) * int(stage.get("n_episodes") or 0) for stage in stages),
        total_games,
    )
    final_stage = stages[-1] if stages else {}
    config = report.get("training_config") or {}
    return {
        "schema_version": "common-v1",
        "model_family": "ppo",
        "algorithm": config.get("algorithm") or (report.get("extra_meta") or {}).get("algorithm") or "ppo",
        "total_games": total_games,
        "wins": wins,
        "losses": losses,
        "draws": 0,
        "win_rate": _safe_div(wins, total_games),
        "avg_episode_length": avg_len,
        "avg_reward": avg_reward,
        "final_loss": final_stage.get("final_loss"),
        "final_policy_loss": final_stage.get("final_policy_loss"),
        "final_value_loss": final_stage.get("final_value_loss"),
        "final_entropy": None,
        "total_updates": sum(int(stage.get("n_updates") or 0) for stage in stages),
        "total_decisions": sum(int(stage.get("duration") or 0) for stage in stages),
        "total_iterations": len(stages),
        "completed_iterations": len(stages),
        "last_iteration": stages[-1].get("stage") if stages else 0,
        "training_log_path": "",
        "rollout_path": "",
        "source_report": source,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=_json_default), encoding="utf-8")


def _maybe_plot(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    plot_paths: dict[str, str] = {}
    stages = report.get("stages") or []
    if stages:
        labels = [str(stage.get("stage")) for stage in stages]
        win_rates = [float(stage.get("win_rate") or 0.0) for stage in stages]
        rewards = [float(stage.get("avg_reward") or 0.0) for stage in stages]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(labels, win_rates, color="#4a90e2")
        ax.set_ylim(0, 1)
        ax.set_xlabel("stage")
        ax.set_ylabel("win rate")
        ax.set_title("Win rate por stage")
        fig.tight_layout()
        path = plots_dir / "winrate_per_stage.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths["winrate"] = str(path.relative_to(output_dir))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(range(len(rewards)), rewards, marker="o", linewidth=1.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("stage")
        ax.set_ylabel("reward")
        ax.set_title("Reward promedio por stage")
        fig.tight_layout()
        path = plots_dir / "episode_rewards.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths["episodes"] = str(path.relative_to(output_dir))

    history = report.get("loss_history") or []
    if history:
        metric_specs = [
            ("value_loss", "value_loss"),
            ("policy_loss", "policy_loss"),
            ("mcts_ce", "mcts_ce"),
            ("entropy", "entropy"),
        ]
        by_iteration: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in history:
            by_iteration[int(row.get("iteration") or 0)].append(row)

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, (key, label) in zip(axes.ravel(), metric_specs):
            for iteration, rows in sorted(by_iteration.items()):
                values: list[float] = []
                for row in rows:
                    raw = row.get(key)
                    try:
                        values.append(float(raw))
                    except (TypeError, ValueError):
                        values.append(float("nan"))
            if any(not math.isnan(value) for value in values):
                ax.plot(range(len(values)), values, label=f"stage {iteration}")
            ax.set_title(label)
            ax.set_xlabel("epoch")
            if ax.get_lines():
                ax.legend(fontsize=7)
        fig.tight_layout()
        path = plots_dir / "loss_curves.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths["loss"] = str(path.relative_to(output_dir))

    stage_breakdowns = [
        (str(stage.get("stage")), stage.get("reward_breakdown") or {})
        for stage in stages
        if stage.get("reward_breakdown")
    ]
    breakdown = report.get("reward_breakdown") or {}
    if stage_breakdowns:
        keys = sorted({key for _, values in stage_breakdowns for key in values})
        x = list(range(len(stage_breakdowns)))
        width = 0.8 / max(1, len(keys))
        fig, ax = plt.subplots(figsize=(10, 5))
        for offset, key in enumerate(keys):
            values = [float(values.get(key) or 0.0) for _, values in stage_breakdowns]
            positions = [pos + offset * width for pos in x]
            ax.bar(positions, values, width=width, label=key)
        ax.set_xticks([pos + width * (len(keys) - 1) / 2 for pos in x])
        ax.set_xticklabels([stage for stage, _ in stage_breakdowns])
        ax.set_title("Reward breakdown")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = plots_dir / "reward_breakdown.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths["breakdown"] = str(path.relative_to(output_dir))
    elif breakdown:
        labels = list(breakdown)
        values = [float(breakdown[key]) for key in labels]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(labels, values)
        ax.set_title("Reward breakdown acumulado")
        fig.tight_layout()
        path = plots_dir / "reward_breakdown.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths["breakdown"] = str(path.relative_to(output_dir))
    return plot_paths


def _write_html(report: dict[str, Any], output_dir: Path, plot_paths: dict[str, str]) -> None:
    common = report.get("common_metrics") or {}
    stages = list(report.get("stages") or [])
    config = report.get("training_config") or {}
    league = report.get("league") or []
    timestamp = report.get("timestamp") or _now_stamp()
    html_parts = [
        "<!doctype html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        f"<title>VGC Bot Report - {_escape(timestamp)}</title>",
        f"<style>{_REPORT_CSS}</style>",
        "</head><body>",
        "<h1>VGC Bot &mdash; Reporte de entrenamiento</h1>",
        f"<p>Generado: <code>{_escape(timestamp)}</code></p>",
        "<h2>Resumen global</h2>",
        f"<div class='metric'><span class='label'>Stages completados:</span> {len(stages)}</div>",
        f"<div class='metric'><span class='label'>Timesteps/decisiones totales:</span> {_metric_value(common.get('total_decisions'))}</div>",
        f"<div class='metric'><span class='label'>Episodios totales:</span> {_metric_value(common.get('total_games'))}</div>",
        f"<div class='metric'><span class='label'>Updates PPO:</span> {_metric_value(common.get('total_updates'))}</div>",
        f"<div class='metric'><span class='label'>Win rate global:</span> {_metric_value(common.get('win_rate'))}</div>",
        f"<div class='metric'><span class='label'>Reward promedio:</span> {_metric_value(common.get('avg_reward'))}</div>",
        "<h2>Metricas comunes normalizadas</h2>",
        _html_table(
            [common],
            [
                ("model_family", "model_family"),
                ("algorithm", "algorithm"),
                ("wins", "wins"),
                ("losses", "losses"),
                ("draws", "draws"),
                ("avg_episode_length", "avg_episode_length"),
                ("final_loss", "final_loss"),
                ("final_policy_loss", "final_policy_loss"),
                ("final_value_loss", "final_value_loss"),
                ("final_entropy", "final_entropy"),
            ],
        ),
        "<h2>Detalle por stage</h2>",
    ]

    for stage in stages:
        html_parts.extend(
            [
                f"<div class='stage'><h3>Stage {_escape(stage.get('stage'))}</h3>",
                f"<p><strong>Timesteps/decisiones:</strong> {_metric_value(stage.get('duration'))} "
                f"({_metric_value(stage.get('started'))} &rarr; {_metric_value(stage.get('ended'))})</p>",
                f"<p><strong>Razon cierre:</strong> {_escape(stage.get('transition_reason'))}</p>",
                f"<p><strong>Oponente/curriculum:</strong> {_escape(stage.get('opponent'))}</p>",
                f"<div class='metric'><span class='label'>Episodios:</span> {_metric_value(stage.get('n_episodes'))}</div>",
                f"<div class='metric'><span class='label'>Win rate:</span> {_metric_value(stage.get('win_rate'))}</div>",
                f"<div class='metric'><span class='label'>Reward avg:</span> {_metric_value(stage.get('avg_reward'))}</div>",
                f"<div class='metric'><span class='label'>Episodio avg len:</span> {_metric_value(stage.get('avg_episode_length'))}</div>",
                f"<div class='metric'><span class='label'>Updates PPO:</span> {_metric_value(stage.get('n_updates'))}</div>",
                f"<div class='metric'><span class='label'>value_loss final:</span> {_metric_value(stage.get('final_value_loss'))}</div>",
                f"<div class='metric'><span class='label'>policy_loss final:</span> {_metric_value(stage.get('final_policy_loss'))}</div>",
                f"<div class='metric'><span class='label'>loss final:</span> {_metric_value(stage.get('final_loss'))}</div>",
            ]
        )
        breakdown = stage.get("reward_breakdown") or {}
        if breakdown:
            rows = [{"component": key, "total": value} for key, value in sorted(breakdown.items())]
            html_parts.append("<h4>Reward breakdown</h4>")
            html_parts.append(_html_table(rows, [("component", "component"), ("total", "total")]))
        simulator = stage.get("simulator") or {}
        if simulator:
            rows = [{"metric": key, "value": value} for key, value in sorted(simulator.items())]
            html_parts.append("<h4>Simulador</h4>")
            html_parts.append(_html_table(rows, [("metric", "metric"), ("value", "value")]))
        html_parts.append("</div>")

    html_parts.append("<h2>Curvas y plots</h2>")
    if plot_paths:
        plot_titles = {
            "loss": "Loss / entrenamiento",
            "winrate": "Win rate por stage",
            "breakdown": "Reward breakdown",
            "episodes": "Reward promedio",
        }
        for key in ("loss", "winrate", "breakdown", "episodes"):
            rel = plot_paths.get(key)
            if not rel:
                continue
            html_parts.append(f"<h3>{plot_titles.get(key, key)}</h3>")
            html_parts.append(f"<img src='{_escape(rel)}'>")
    else:
        html_parts.append("<p>No se pudieron generar graficos (matplotlib no disponible o sin datos).</p>")

    html_parts.append("<h2>Self-play league</h2>")
    if league:
        html_parts.append(_html_table(list(league), [(key, key) for key in sorted(league[0])]))
    else:
        html_parts.append("<p>No hubo entradas de league para este modelo/reporte.</p>")

    html_parts.extend(
        [
            "<h2>Configuracion del entrenamiento</h2>",
            _html_table([{"key": key, "value": value} for key, value in sorted(config.items())], [("key", "key"), ("value", "value")]),
            "</body></html>",
        ]
    )
    (output_dir / "report.html").write_text("\n".join(html_parts), encoding="utf-8")


def write_alphazero_report(
    *,
    training_log_path: Path,
    rollout_path: Path,
    output_root: Path = Path("reports"),
    output_dir: Path | None = None,
    title: str = "AlphaZero MCTS+PPO",
) -> Path:
    report = build_alphazero_report(
        training_log_path=training_log_path,
        rollout_path=rollout_path,
        title=title,
    )
    target = output_dir or output_root / f"alphazero_{_now_stamp()}"
    target.mkdir(parents=True, exist_ok=True)
    plot_paths = _maybe_plot(report, target)
    report["plots"] = plot_paths
    report.setdefault("league", [])
    _write_json(target / "report.json", report)
    _write_json(target / "common_metrics.json", report["common_metrics"])
    _write_json(target / "league_stats.json", {"league": report.get("league") or []})
    _write_html(report, target, plot_paths)
    return target


def write_report_payload(
    report: dict[str, Any],
    *,
    output_root: Path = Path("reports"),
    output_dir: Path | None = None,
    prefix: str = "model",
) -> Path:
    target = output_dir or output_root / f"{prefix}_{_now_stamp()}"
    target.mkdir(parents=True, exist_ok=True)
    report.setdefault("timestamp", _now_stamp())
    report.setdefault("league", [])
    plot_paths = _maybe_plot(report, target)
    report["plots"] = plot_paths
    _write_json(target / "report.json", report)
    if "common_metrics" in report:
        _write_json(target / "common_metrics.json", report["common_metrics"])
    _write_json(target / "league_stats.json", {"league": report.get("league") or []})
    _write_html(report, target, plot_paths)
    return target


def write_ppo_common_report(*, report_json: Path, output_dir: Path) -> Path:
    report = json.loads(report_json.read_text(encoding="utf-8"))
    report["common_metrics"] = build_common_from_ppo_report(report, source=str(report_json))
    report.setdefault("league", [])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "report.json", report)
    _write_json(output_dir / "common_metrics.json", report["common_metrics"])
    _write_json(output_dir / "league_stats.json", {"league": report.get("league") or []})
    _write_html(report, output_dir, {})
    return output_dir


def add_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metrics-report-dir",
        type=Path,
        default=Path("reports"),
        help="Directorio raiz para reportes comparables (default: reports).",
    )
    parser.add_argument(
        "--no-metrics-report",
        action="store_true",
        help="No generar report.json/common_metrics.json al finalizar.",
    )
