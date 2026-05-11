"""Round-robin tournament between AlphaZero MCTS+PPO, CFR, and PPO/RecurrentPPO."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from html import escape
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from login import (
    DEFAULT_TEAM_PATH,
    build_server_config,
    load_team,
    make_anonymous_account,
    make_policy_player,
    should_use_team,
)
from play import (
    alphazero_policy_kwargs,
    cfr_policy_kwargs,
    cleanup_startup_battles,
    close_player,
    enable_turn_logging,
    forfeit_unfinished_battles,
    ppo_policy_kwargs,
    wait_until_battles_closed,
)
from src.alphazero.showdown_simulator import ShowdownSimulationTracker, attach_simulation_tracking
from src.format_resolver import resolve_format


TOURNAMENT_MODELS = ("alphazero_mcts", "cfr", "ppo_recurrent")


def first_existing(candidates: list[str]) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return candidates[0]


def compact_name(model: str) -> str:
    return {
        "alphazero_mcts": "AlphaZero",
        "cfr": "CFR",
        "ppo_recurrent": "PPO recurrente",
    }.get(model, model)


def make_account_prefix(model: str) -> str:
    return {
        "alphazero_mcts": "AZ",
        "cfr": "CFR",
        "ppo_recurrent": "PPO",
    }.get(model, "BOT")


def make_policy_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        team=args.team,
        alphazero_checkpoint=args.alphazero_checkpoint,
        alphazero_simulations=args.alphazero_simulations,
        alphazero_depth=args.alphazero_depth,
        alphazero_max_candidates=args.alphazero_max_candidates,
        alphazero_cpuct=args.alphazero_cpuct,
        alphazero_temperature=args.alphazero_temperature,
        alphazero_heuristic_weight=args.alphazero_heuristic_weight,
        alphazero_depth2_weight=args.alphazero_depth2_weight,
        alphazero_simulator_url=args.alphazero_simulator_url,
        alphazero_live_state_url=args.alphazero_live_state_url,
        alphazero_simulator_timeout=args.alphazero_simulator_timeout,
        alphazero_simulator_max_choices=args.alphazero_simulator_max_choices,
        alphazero_simulator_opponent_policy=args.alphazero_simulator_opponent_policy,
        alphazero_simulator_robust_worst_weight=args.alphazero_simulator_robust_worst_weight,
        alphazero_require_simulator=args.alphazero_require_simulator,
        alphazero_device=args.alphazero_device,
        cfr_checkpoint=args.cfr_checkpoint,
        cfr_max_candidates=args.cfr_max_candidates,
        cfr_temperature=args.cfr_temperature,
        cfr_fallback=args.cfr_fallback,
        cfr_neural_checkpoint=args.cfr_neural_checkpoint,
        cfr_neural_weight=args.cfr_neural_weight,
        cfr_min_average_visits=args.cfr_min_average_visits,
        cfr_neural_device=args.cfr_neural_device,
        ppo_checkpoint=args.ppo_checkpoint,
        ppo_device=args.ppo_device,
        ppo_deterministic=args.ppo_deterministic,
        ppo_strict_actions=args.ppo_strict_actions,
    )


def kwargs_for_model(
    model: str,
    policy_args: SimpleNamespace,
    tracker: ShowdownSimulationTracker | None,
) -> dict[str, Any]:
    if model == "alphazero_mcts":
        return alphazero_policy_kwargs(policy_args, tracker)
    if model == "cfr":
        return cfr_policy_kwargs(policy_args)
    if model == "ppo_recurrent":
        return ppo_policy_kwargs(policy_args)
    raise ValueError(f"Modelo no permitido en torneo: {model}")


def make_player(
    model: str,
    *,
    battle_format: str,
    server_configuration,
    team: str | None,
    policy_kwargs: dict[str, Any],
):
    prefix = make_account_prefix(model)
    username_prefix = f"T{prefix}{uuid.uuid4().hex[:5]}"
    return make_policy_player(
        model,
        battle_format=battle_format,
        team=team,
        server_configuration=server_configuration,
        account_configuration=make_anonymous_account(username_prefix),
        policy_kwargs=policy_kwargs,
    )


def result_from_battle(
    battle,
    *,
    p1_model: str,
    p2_model: str,
    timeout_tags: set[str],
    segment_id: str,
) -> dict[str, Any]:
    tag = getattr(battle, "battle_tag", "")
    finished = bool(getattr(battle, "finished", False))
    timed_out = tag in timeout_tags
    winner_model = None
    loser_model = None
    result = "timeout" if timed_out else "unfinished"

    if finished and not timed_out:
        if bool(getattr(battle, "won", False)):
            winner_model = p1_model
            loser_model = p2_model
            result = "p1_win"
        elif bool(getattr(battle, "lost", False)):
            winner_model = p2_model
            loser_model = p1_model
            result = "p2_win"
        else:
            result = "draw"

    return {
        "segment_id": segment_id,
        "battle_tag": tag,
        "p1_model": p1_model,
        "p2_model": p2_model,
        "p1_label": compact_name(p1_model),
        "p2_label": compact_name(p2_model),
        "winner_model": winner_model,
        "winner_label": compact_name(winner_model) if winner_model else "",
        "loser_model": loser_model,
        "loser_label": compact_name(loser_model) if loser_model else "",
        "result": result,
        "finished": finished and not timed_out,
        "timed_out": timed_out,
        "turns": int(getattr(battle, "turn", 0) or 0),
    }


async def run_segment(
    *,
    args: argparse.Namespace,
    policy_args: SimpleNamespace,
    p1_model: str,
    p2_model: str,
    n_games: int,
    segment_index: int,
) -> list[dict[str, Any]]:
    battle_format = resolve_format(args.format)
    server_configuration = build_server_config(args.server)
    team = load_team(args.team) if should_use_team(battle_format) else None
    segment_id = f"{segment_index:02d}-{p1_model}-vs-{p2_model}"

    tracker = None
    if (
        "alphazero_mcts" in {p1_model, p2_model}
        and args.alphazero_depth >= 2
        and should_use_team(battle_format)
        and not args.alphazero_live_state_url
    ):
        tracker = ShowdownSimulationTracker(
            battle_format=battle_format,
            team_text=load_team(args.team),
        )

    p1 = make_player(
        p1_model,
        battle_format=battle_format,
        server_configuration=server_configuration,
        team=team,
        policy_kwargs=kwargs_for_model(p1_model, policy_args, tracker),
    )
    p2 = make_player(
        p2_model,
        battle_format=battle_format,
        server_configuration=server_configuration,
        team=team,
        policy_kwargs=kwargs_for_model(p2_model, policy_args, tracker),
    )

    if tracker is not None:
        attach_simulation_tracking(p1, tracker)
        attach_simulation_tracking(p2, tracker)
    if args.verbose_turns:
        enable_turn_logging(p1, f"{compact_name(p1_model)} P1")
        enable_turn_logging(p2, f"{compact_name(p2_model)} P2")

    print(
        f"\n[{segment_id}] {compact_name(p1_model)} vs {compact_name(p2_model)} "
        f"({n_games} partidas)",
        flush=True,
    )

    timeout_tags: set[str] = set()
    try:
        await cleanup_startup_battles(
            [p1, p2],
            wait_seconds=args.startup_cleanup_wait,
            login_timeout=args.login_timeout,
        )
        runner = p1.battle_against(p2, n_battles=n_games)
        if args.battle_timeout:
            try:
                await asyncio.wait_for(runner, timeout=args.battle_timeout)
            except asyncio.TimeoutError:
                timeout_tags = {
                    tag
                    for tag, battle in p1.battles.items()
                    if not getattr(battle, "finished", False)
                }
                print(
                    f"  Timeout: {len(timeout_tags)} batalla(s) sin cerrar en {segment_id}.",
                    flush=True,
                )
                await forfeit_unfinished_battles(p1, "Timeout torneo")
                await forfeit_unfinished_battles(p2, "Timeout torneo")
                await wait_until_battles_closed([p1, p2], timeout=10.0)
        else:
            await runner
    finally:
        await close_player(p1)
        await close_player(p2)

    rows = [
        result_from_battle(
            battle,
            p1_model=p1_model,
            p2_model=p2_model,
            timeout_tags=timeout_tags,
            segment_id=segment_id,
        )
        for battle in p1.battles.values()
    ]
    print(
        f"  Segmento cerrado: {sum(1 for row in rows if row['finished'])}/{n_games} "
        "partidas terminadas.",
        flush=True,
    )
    return rows


def aggregate_results(games: list[dict[str, Any]]) -> dict[str, Any]:
    standings: dict[str, Counter] = {model: Counter() for model in TOURNAMENT_MODELS}
    turns_by_model: dict[str, list[int]] = defaultdict(list)
    side_stats: dict[str, Counter] = {model: Counter() for model in TOURNAMENT_MODELS}
    matchup_stats: dict[str, Any] = {}

    for row in games:
        p1 = row["p1_model"]
        p2 = row["p2_model"]
        pair = " vs ".join(sorted([p1, p2]))
        matchup = matchup_stats.setdefault(
            pair,
            {
                "models": sorted([p1, p2]),
                "played": 0,
                "finished": 0,
                "timeouts": 0,
                "draws": 0,
                "wins": {p1: 0, p2: 0},
                "avg_turns": 0.0,
                "turns": [],
            },
        )

        matchup["played"] += 1
        if row["timed_out"]:
            matchup["timeouts"] += 1
        if row["result"] == "draw":
            matchup["draws"] += 1
        if row["finished"]:
            matchup["finished"] += 1
            matchup["turns"].append(row["turns"])
            for model in [p1, p2]:
                standings[model]["played"] += 1
                turns_by_model[model].append(row["turns"])
            winner = row.get("winner_model")
            loser = row.get("loser_model")
            if winner:
                standings[winner]["wins"] += 1
                standings[loser]["losses"] += 1
                matchup["wins"][winner] = matchup["wins"].get(winner, 0) + 1
                side_stats[winner]["p1_wins" if winner == p1 else "p2_wins"] += 1
                side_stats[loser]["p1_losses" if loser == p1 else "p2_losses"] += 1
            else:
                standings[p1]["draws"] += 1
                standings[p2]["draws"] += 1
        else:
            standings[p1]["unfinished"] += 1
            standings[p2]["unfinished"] += 1

    standing_rows = []
    for model in TOURNAMENT_MODELS:
        played = int(standings[model]["played"])
        wins = int(standings[model]["wins"])
        losses = int(standings[model]["losses"])
        draws = int(standings[model]["draws"])
        unfinished = int(standings[model]["unfinished"])
        avg_turns = (
            sum(turns_by_model[model]) / len(turns_by_model[model])
            if turns_by_model[model]
            else 0.0
        )
        standing_rows.append(
            {
                "model": model,
                "label": compact_name(model),
                "played": played,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "unfinished": unfinished,
                "winrate": wins / played if played else 0.0,
                "avg_turns": avg_turns,
                "side_stats": dict(side_stats[model]),
            }
        )

    standing_rows.sort(key=lambda row: (-row["wins"], row["losses"], row["avg_turns"]))
    for rank, row in enumerate(standing_rows, start=1):
        row["rank"] = rank

    for matchup in matchup_stats.values():
        turns = matchup.pop("turns", [])
        matchup["avg_turns"] = sum(turns) / len(turns) if turns else 0.0

    return {
        "standings": standing_rows,
        "matchups": matchup_stats,
        "totals": {
            "recorded_games": len(games),
            "finished_games": sum(1 for row in games if row["finished"]),
            "timeouts": sum(1 for row in games if row["timed_out"]),
            "draws": sum(1 for row in games if row["result"] == "draw"),
        },
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{escape(str(cell))}</td>" for cell in row)
            + "</tr>"
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_report(
    *,
    args: argparse.Namespace,
    games: list[dict[str, Any]],
    aggregate: dict[str, Any],
    started_at: str,
    elapsed_seconds: float,
) -> Path:
    report_dir = args.output_dir or Path(args.output_root) / f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "plots").mkdir(exist_ok=True)

    config = {
        "models": list(TOURNAMENT_MODELS),
        "games_per_pair": args.games_per_pair,
        "server": args.server,
        "format": resolve_format(args.format),
        "team": args.team,
        "alphazero_checkpoint": args.alphazero_checkpoint,
        "alphazero_simulations": args.alphazero_simulations,
        "alphazero_depth": args.alphazero_depth,
        "cfr_checkpoint": args.cfr_checkpoint,
        "cfr_neural_checkpoint": args.cfr_neural_checkpoint,
        "ppo_checkpoint": args.ppo_checkpoint,
    }
    payload = {
        "schema_version": "model-tournament-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at,
        "elapsed_seconds": elapsed_seconds,
        "config": config,
        "standings": aggregate["standings"],
        "matchups": aggregate["matchups"],
        "totals": aggregate["totals"],
        "games": games,
    }

    common_metrics = {
        "schema_version": "common-tournament-v1",
        "model_family": "tournament",
        "algorithm": "round_robin_three_models",
        "models": list(TOURNAMENT_MODELS),
        "total_games": aggregate["totals"]["recorded_games"],
        "finished_games": aggregate["totals"]["finished_games"],
        "timeouts": aggregate["totals"]["timeouts"],
        "draws": aggregate["totals"]["draws"],
        "leader": aggregate["standings"][0] if aggregate["standings"] else None,
        "standings": aggregate["standings"],
    }
    league_stats = {
        "standings": aggregate["standings"],
        "matchups": aggregate["matchups"],
        "games": games,
    }

    (report_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (report_dir / "common_metrics.json").write_text(
        json.dumps(common_metrics, indent=2),
        encoding="utf-8",
    )
    (report_dir / "league_stats.json").write_text(
        json.dumps(league_stats, indent=2),
        encoding="utf-8",
    )
    with (report_dir / "battles.jsonl").open("w", encoding="utf-8") as handle:
        for row in games:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    standings_table = table(
        ["Rank", "Modelo", "Jugadas", "Victorias", "Derrotas", "Empates", "Timeouts", "Win rate", "Turnos prom."],
        [
            [
                row["rank"],
                row["label"],
                row["played"],
                row["wins"],
                row["losses"],
                row["draws"],
                row["unfinished"],
                format_pct(row["winrate"]),
                f"{row['avg_turns']:.1f}",
            ]
            for row in aggregate["standings"]
        ],
    )
    matchup_table = table(
        ["Cruce", "Terminadas", "Timeouts", "Empates", "Victorias", "Turnos prom."],
        [
            [
                " vs ".join(compact_name(model) for model in stats["models"]),
                stats["finished"],
                stats["timeouts"],
                stats["draws"],
                ", ".join(
                    f"{compact_name(model)}={wins}"
                    for model, wins in sorted(stats["wins"].items())
                ),
                f"{stats['avg_turns']:.1f}",
            ]
            for stats in aggregate["matchups"].values()
        ],
    )
    games_table = table(
        ["Batalla", "P1", "P2", "Ganador", "Resultado", "Turnos"],
        [
            [
                row["battle_tag"],
                row["p1_label"],
                row["p2_label"],
                row["winner_label"] or "-",
                row["result"],
                row["turns"],
            ]
            for row in games
        ],
    )

    css = """
body { font-family: Arial, sans-serif; margin: 32px; color: #17202a; }
h1, h2 { margin-bottom: 0.35em; }
.meta { color: #566573; margin-bottom: 1.5em; }
.cards { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
.card { border: 1px solid #d5d8dc; border-radius: 6px; padding: 12px 14px; min-width: 160px; }
.metric { font-size: 1.45em; font-weight: 700; }
.label { color: #566573; font-size: 0.9em; }
table { border-collapse: collapse; width: 100%; margin: 18px 0 28px; }
th, td { border: 1px solid #d5d8dc; padding: 8px 10px; text-align: left; }
th { background: #f4f6f7; }
code { background: #f4f6f7; padding: 2px 5px; border-radius: 4px; }
"""
    leader = aggregate["standings"][0] if aggregate["standings"] else None
    leader_label = leader["label"] if leader else "-"
    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Torneo de modelos VGC</title>
  <style>{css}</style>
</head>
<body>
  <h1>Torneo de modelos VGC</h1>
  <p class="meta">AlphaZero MCTS+PPO vs CFR vs PPO recurrente. Generado en {escape(payload["created_at"])}.</p>
  <div class="cards">
    <div class="card"><div class="label">Ganador parcial</div><div class="metric">{escape(leader_label)}</div></div>
    <div class="card"><div class="label">Partidas registradas</div><div class="metric">{aggregate["totals"]["recorded_games"]}</div></div>
    <div class="card"><div class="label">Terminadas</div><div class="metric">{aggregate["totals"]["finished_games"]}</div></div>
    <div class="card"><div class="label">Timeouts</div><div class="metric">{aggregate["totals"]["timeouts"]}</div></div>
    <div class="card"><div class="label">Empates</div><div class="metric">{aggregate["totals"]["draws"]}</div></div>
  </div>
  <h2>Configuracion</h2>
  <p><code>{escape(json.dumps(config, ensure_ascii=False))}</code></p>
  <h2>Ranking</h2>
  {standings_table}
  <h2>Cruces</h2>
  {matchup_table}
  <h2>Partidas</h2>
  {games_table}
</body>
</html>
"""
    (report_dir / "report.html").write_text(html, encoding="utf-8")
    return report_dir


async def run_tournament(args: argparse.Namespace) -> Path:
    battle_format = resolve_format(args.format)
    print("=" * 68)
    print("  Torneo VGC - AlphaZero vs CFR vs PPO recurrente")
    print("=" * 68)
    print(f"  Servidor: {build_server_config(args.server).websocket_url}")
    print(f"  Formato:  {battle_format}")
    print(f"  Partidas por cruce: {args.games_per_pair}")
    print(f"  Modelos: {', '.join(compact_name(model) for model in TOURNAMENT_MODELS)}")
    print("=" * 68)

    policy_args = make_policy_args(args)
    games: list[dict[str, Any]] = []
    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.time()
    segment_index = 1

    for model_a, model_b in combinations(TOURNAMENT_MODELS, 2):
        first_side_games = (args.games_per_pair + 1) // 2
        second_side_games = args.games_per_pair // 2
        if first_side_games:
            games.extend(
                await run_segment(
                    args=args,
                    policy_args=policy_args,
                    p1_model=model_a,
                    p2_model=model_b,
                    n_games=first_side_games,
                    segment_index=segment_index,
                )
            )
            segment_index += 1
        if second_side_games:
            games.extend(
                await run_segment(
                    args=args,
                    policy_args=policy_args,
                    p1_model=model_b,
                    p2_model=model_a,
                    n_games=second_side_games,
                    segment_index=segment_index,
                )
            )
            segment_index += 1

    aggregate = aggregate_results(games)
    report_dir = write_report(
        args=args,
        games=games,
        aggregate=aggregate,
        started_at=started_at,
        elapsed_seconds=time.time() - start,
    )
    print("\n" + "=" * 68)
    print("  RESULTADOS DEL TORNEO")
    print("=" * 68)
    for row in aggregate["standings"]:
        print(
            f"  #{row['rank']} {row['label']}: "
            f"{row['wins']}-{row['losses']}-{row['draws']} "
            f"({format_pct(row['winrate'])})"
        )
    print(f"\n  Reporte generado en: {report_dir}")
    print(f"    · {report_dir / 'report.html'}")
    print(f"    · {report_dir / 'report.json'}")
    print(f"    · {report_dir / 'common_metrics.json'}")
    print(f"    · {report_dir / 'league_stats.json'}")
    return report_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Torneo round-robin solo entre alphazero_mcts, cfr y ppo_recurrent."
    )
    parser.add_argument("--games-per-pair", type=int, default=10)
    parser.add_argument("--server", type=str, default="showdown:8000")
    parser.add_argument("--format", type=str, default="gen9vgc2026regi")
    parser.add_argument("--team", type=str, default=str(DEFAULT_TEAM_PATH))
    parser.add_argument("--output-root", type=Path, default=Path("reports"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--battle-timeout", type=float, default=None)
    parser.add_argument("--startup-cleanup-wait", type=float, default=2.0)
    parser.add_argument("--login-timeout", type=float, default=30.0)
    parser.add_argument("--verbose-turns", action="store_true")

    parser.add_argument(
        "--alphazero-checkpoint",
        type=str,
        default=first_existing(
            [
                "checkpoints/alphazero_mcts_ppo_offline_d4_zero_20260509/best.pt",
                "checkpoints/alphazero_mcts_ppo_offline_d2/best.pt",
                "checkpoints/alphazero_mcts_ppo_d2_required_v6/best.pt",
            ]
        ),
    )
    parser.add_argument("--alphazero-device", type=str, default="cpu")
    parser.add_argument("--alphazero-simulations", type=int, default=128)
    parser.add_argument("--alphazero-depth", type=int, default=4)
    parser.add_argument("--alphazero-max-candidates", type=int, default=96)
    parser.add_argument("--alphazero-cpuct", type=float, default=1.5)
    parser.add_argument("--alphazero-temperature", type=float, default=0.0)
    parser.add_argument("--alphazero-heuristic-weight", type=float, default=0.75)
    parser.add_argument("--alphazero-depth2-weight", type=float, default=0.8)
    parser.add_argument("--alphazero-simulator-url", type=str, default="http://showdown-sim:9001")
    parser.add_argument("--alphazero-live-state-url", type=str, default="http://showdown:9002")
    parser.add_argument("--alphazero-simulator-timeout", type=float, default=180.0)
    parser.add_argument("--alphazero-simulator-max-choices", type=int, default=8)
    parser.add_argument(
        "--alphazero-simulator-opponent-policy",
        choices=["minimax", "mean", "robust"],
        default="robust",
    )
    parser.add_argument("--alphazero-simulator-robust-worst-weight", type=float, default=0.35)
    parser.add_argument(
        "--alphazero-require-simulator",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--cfr-checkpoint",
        type=str,
        default=first_existing(
            [
                "checkpoints/cfr_tabular_neural_d2_zero_20260510/best.json",
                "checkpoints/cfr_tabular_neural_20260510/best.json",
            ]
        ),
    )
    parser.add_argument("--cfr-max-candidates", type=int, default=32)
    parser.add_argument("--cfr-temperature", type=float, default=0.0)
    parser.add_argument("--cfr-fallback", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument(
        "--cfr-neural-checkpoint",
        type=str,
        default=first_existing(
            [
                "checkpoints/cfr_tabular_neural_d2_zero_20260510/prior.pt",
                "checkpoints/cfr_tabular_neural_20260510/prior.pt",
            ]
        ),
    )
    parser.add_argument("--cfr-neural-weight", type=float, default=0.70)
    parser.add_argument("--cfr-min-average-visits", type=int, default=3)
    parser.add_argument("--cfr-neural-device", type=str, default="cpu")

    parser.add_argument("--ppo-checkpoint", type=str, default="checkpoints/vgc_final.zip")
    parser.add_argument("--ppo-device", type=str, default="cpu")
    parser.add_argument(
        "--ppo-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ppo-strict-actions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games_per_pair <= 0:
        raise SystemExit("--games-per-pair debe ser mayor que 0")
    asyncio.run(run_tournament(args))


if __name__ == "__main__":
    main()
