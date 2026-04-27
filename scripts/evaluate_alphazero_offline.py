"""Evaluate AlphaZero MCTS in exact offline Showdown battles.

This script is intentionally separate from play.py: play.py talks to a live
websocket battle, while this path keeps the whole battle inside the JS
simulator and advances the serialized state directly.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from login import load_team, should_use_team
from src.alphazero.network import load_checkpoint
from src.alphazero.offline_selfplay import (
    OfflineAlphaZeroSearch,
    OfflineShowdownClient,
    _baseline_choice,
    _team_order,
)
from src.format_resolver import resolve_format


def _active_summary(snapshot: dict, side: str) -> str:
    active = ((snapshot.get("sides") or {}).get(side) or {}).get("active") or []
    labels = []
    for pokemon in active:
        if not pokemon:
            continue
        hp = int(round(float(pokemon.get("hp_fraction") or 0.0) * 100))
        suffix = ", fnt" if pokemon.get("fainted") else ""
        status = pokemon.get("status") or ""
        if status:
            suffix += f", {status}"
        labels.append(f"{pokemon.get('species') or pokemon.get('name')} ({hp}%{suffix})")
    return ", ".join(labels) if labels else "-"


def _print_decision(verbose: bool, side_label: str, choice: str) -> None:
    if verbose:
        print(f"    Decision [{side_label}]: /choose {choice}", flush=True)


def _winner_text(winner_side: str) -> str:
    if winner_side == "p1":
        return "VICTORIA"
    if winner_side == "p2":
        return "DERROTA"
    return "EMPATE"


def play_game(args, game_index: int, searcher: OfflineAlphaZeroSearch, client: OfflineShowdownClient) -> str:
    battle_format = resolve_format(args.format)
    team_text = load_team(args.team)
    rng = random.Random(int(args.offline_seed) + game_index)
    seed = [
        rng.randrange(1, 0xFFFFFFFF),
        rng.randrange(1, 0xFFFFFFFF),
        rng.randrange(1, 0xFFFFFFFF),
        rng.randrange(1, 0xFFFFFFFF),
    ]
    searcher.rng = np.random.default_rng(int(args.offline_seed) + game_index)

    snapshot = client.start(
        battle_format=battle_format,
        team_text=team_text,
        seed=seed,
        team_choice_p1=_team_order(rng, team_text, args.offline_team_preview),
        team_choice_p2=_team_order(rng, team_text, args.offline_team_preview),
        max_choices=args.max_candidates,
    )

    if args.verbose:
        print(f"\n  Batalla offline {game_index + 1}: seed={seed}")

    for _ in range(max(1, int(args.offline_max_turns))):
        if snapshot.get("ended"):
            break
        if args.verbose:
            print(
                f"  Turno {snapshot.get('turn', 0)}: "
                f"propios: {_active_summary(snapshot, 'p1')} | "
                f"rival: {_active_summary(snapshot, 'p2')}",
                flush=True,
            )

        legal = snapshot.get("legal") or {}
        choices: dict[str, str] = {}
        p1_legal = list(legal.get("p1") or [])
        p2_legal = list(legal.get("p2") or [])

        if p1_legal:
            result = searcher.search(snapshot, "p1")
            if result is not None:
                choices["p1"] = result.choice
                _print_decision(args.verbose, "Principal", result.choice)

        if p2_legal:
            if args.p2 == "self":
                result = searcher.search(snapshot, "p2")
                if result is not None:
                    choices["p2"] = result.choice
                    _print_decision(args.verbose, "Rival", result.choice)
            else:
                choices["p2"] = _baseline_choice(args.p2, p2_legal, rng)
                _print_decision(args.verbose, "Rival", choices["p2"])

        if not choices:
            break
        snapshot = client.choose(
            state=snapshot["state"],
            choices=choices,
            max_choices=args.max_candidates,
        )

    winner_side = snapshot.get("winner_side") or ""
    if not winner_side:
        p1_score = float((snapshot.get("score") or {}).get("p1") or 0.0)
        p2_score = float((snapshot.get("score") or {}).get("p2") or 0.0)
        if p1_score > p2_score:
            winner_side = "p1"
        elif p2_score > p1_score:
            winner_side = "p2"

    print(
        f"  Final: offline-{game_index + 1} -> {_winner_text(winner_side)} "
        f"en {snapshot.get('turn', 0)} turnos.",
        flush=True,
    )
    return winner_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero in exact offline Showdown battles")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--p2", choices=["random", "greedy", "self"], default="random")
    parser.add_argument("--format", type=str, default="gen9vgc2026regi")
    parser.add_argument("--team", type=str, default="team.txt")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--simulator-url", type=str, default="http://showdown-sim:9001")
    parser.add_argument("--simulator-timeout", type=float, default=180.0)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=96)
    parser.add_argument("--simulator-max-choices", type=int, default=8)
    parser.add_argument(
        "--simulator-opponent-policy",
        choices=["minimax", "mean", "robust"],
        default="robust",
    )
    parser.add_argument("--simulator-robust-worst-weight", type=float, default=0.35)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--offline-max-turns", type=int, default=120)
    parser.add_argument("--offline-seed", type=int, default=7)
    parser.add_argument("--offline-team-preview", choices=["default", "random"], default="random")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    battle_format = resolve_format(args.format)
    if not should_use_team(battle_format):
        raise ValueError("Offline evaluation requires a fixed team format")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    client = OfflineShowdownClient(args.simulator_url, timeout=args.simulator_timeout)
    model = load_checkpoint(args.checkpoint, device=args.device)
    searcher = OfflineAlphaZeroSearch(
        model=model,
        client=client,
        simulations=args.simulations,
        depth=args.depth,
        max_candidates=args.max_candidates,
        cpuct=args.cpuct,
        temperature=args.temperature,
        device=args.device,
        simulator_max_choices=args.simulator_max_choices,
        simulator_opponent_policy=args.simulator_opponent_policy,
        simulator_robust_worst_weight=args.simulator_robust_worst_weight,
        require_simulator=True,
    )

    print("=" * 55)
    print("  VGC Bot - Offline Eval")
    print("=" * 55)
    print(f"  Formato:  {battle_format}")
    print(f"  Principal: ALPHAZERO_MCTS")
    print(f"  Rival:     {args.p2.upper()}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  MCTS: sims={args.simulations} depth={args.depth}")
    print(f"  Partidas: {args.n}")
    print("=" * 55)

    wins = 0
    losses = 0
    draws = 0
    started = time.time()
    for game_index in range(args.n):
        winner_side = play_game(args, game_index, searcher, client)
        if winner_side == "p1":
            wins += 1
        elif winner_side == "p2":
            losses += 1
        else:
            draws += 1

    print("\n" + "=" * 55)
    print("  RESULTADOS OFFLINE")
    print("=" * 55)
    print(f"  AlphaZero victorias: {wins} / {args.n}")
    print(f"  Rival {args.p2.upper()} victorias: {losses} / {args.n}")
    if draws:
        print(f"  Empates/por score neutro: {draws} / {args.n}")
    print(f"  Tiempo: {time.time() - started:.1f}s")
    print("=" * 55)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
