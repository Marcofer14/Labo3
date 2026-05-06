"""
src.training.tournament · Round-robin entre miembros del league.

Cada par (A, B) juega `n_battles` partidas. Se actualizan los win-rates
de cada miembro en el league. Útil para:
  · seedear la métrica de win-rate cuando agregás un snapshot nuevo
  · evaluar el league al cierre del entrenamiento (parte del reporte)

NOTA: requiere conexión a Showdown. Cada batalla consume ~30s en server local.
Costo: O(N² * n_battles). Con N=6, n_battles=10 → 300 batallas → ~2.5h.
Ajustar `n_battles` y/o `pairs` según presupuesto.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Callable, Optional

from src.training.league import SelfPlayLeague, LeagueEntry


# ── Helpers ──────────────────────────────────────────────────────

def _load_model(path: str, algorithm: str):
    """Carga un modelo desde disco según el algoritmo."""
    if algorithm == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path)
    if algorithm == "maskable_ppo":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path)
    raise ValueError(f"algorithm desconocido: {algorithm}")


# ── Tournament ───────────────────────────────────────────────────

def run_tournament(
    league:       SelfPlayLeague,
    encoder_env_factory: Callable,   # () → VGCEnv (proxy para encoding)
    server_cfg,
    battle_format: str,
    rival_pool,                       # Teambuilder para los rivales del player
    algorithm:    str   = "recurrent_ppo",
    n_battles_per_pair: int = 10,
    only_new:     bool  = False,     # solo evaluar snapshots con < min_battles
    min_battles:  int   = 20,
    verbose:      bool  = True,
) -> dict:
    """
    Round-robin entre los miembros del league.

    Devuelve: dict con resultados por par + actualiza league internamente.

    Implementación: usa `Player.battle_against()` de poke-env (asíncrono).
    Si esa API no está disponible, devuelve resultados vacíos (no rompe).
    """
    from src.training.opponents import LeagueOpponent
    from poke_env.player import Player

    if not league.entries or len(league.entries) < 2:
        if verbose:
            print(f"  [tournament] league con < 2 miembros, saltando.")
        return {}

    members = list(league.entries)
    if only_new:
        members = [e for e in members if e.battles < min_battles]
        if not members:
            return {}

    results: dict[tuple[int,int], dict] = {}
    pairs = [(a, b) for a in members for b in league.entries if a.snapshot_id != b.snapshot_id]

    if verbose:
        print(f"\n[tournament] {len(pairs)} pares × {n_battles_per_pair} batallas = "
              f"{len(pairs) * n_battles_per_pair} partidas")

    t0 = time.time()
    for a, b in pairs:
        try:
            wins_a, wins_b = _play_pair(
                a, b, encoder_env_factory, server_cfg, battle_format,
                rival_pool, algorithm, n_battles_per_pair,
            )
            results[(a.snapshot_id, b.snapshot_id)] = {
                "a_wins": wins_a, "b_wins": wins_b, "n": n_battles_per_pair,
            }
            # Actualizar win-rates en el league
            for _ in range(wins_a):
                league.record_result(a.snapshot_id, won=True)
                league.record_result(b.snapshot_id, won=False)
            for _ in range(wins_b):
                league.record_result(b.snapshot_id, won=True)
                league.record_result(a.snapshot_id, won=False)
            if verbose:
                print(f"  · snap{a.snapshot_id} vs snap{b.snapshot_id}: "
                      f"{wins_a}–{wins_b}  ({n_battles_per_pair})")
        except Exception as ex:
            if verbose:
                print(f"  [!] error en par {a.snapshot_id} vs {b.snapshot_id}: {ex}")
    if verbose:
        print(f"  [tournament] terminó en {time.time()-t0:.1f}s")
    return results


def _play_pair(
    a:                 LeagueEntry,
    b:                 LeagueEntry,
    encoder_env_factory: Callable,
    server_cfg,
    battle_format:     str,
    rival_pool,
    algorithm:         str,
    n_battles:         int,
) -> tuple[int, int]:
    """Juega N batallas entre los snapshots A y B. Retorna (wins_a, wins_b)."""
    from src.training.opponents import LeagueOpponent

    # Cargar ambos modelos
    model_a = _load_model(a.path, algorithm)
    model_b = _load_model(b.path, algorithm)

    # Encoder env compartido (no listening, solo para embed_battle)
    enc = encoder_env_factory()

    # Crear los dos players
    player_a = LeagueOpponent(
        model         = model_a,
        encoder_env   = enc,
        algorithm     = algorithm,
        snapshot_id   = a.snapshot_id,
        battle_format = battle_format,
        team          = rival_pool,
        server_configuration = server_cfg,
    )
    player_b = LeagueOpponent(
        model         = model_b,
        encoder_env   = enc,
        algorithm     = algorithm,
        snapshot_id   = b.snapshot_id,
        battle_format = battle_format,
        team          = rival_pool,
        server_configuration = server_cfg,
    )

    # Player.battle_against(opponent, n_battles=N) corre N batallas asíncronas
    try:
        asyncio.get_event_loop().run_until_complete(
            player_a.battle_against(player_b, n_battles=n_battles)
        )
    except RuntimeError:
        # event loop ya corriendo (Jupyter)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(player_a.battle_against(player_b, n_battles=n_battles))
        loop.close()

    wins_a = player_a.n_won_battles
    wins_b = player_b.n_won_battles
    return wins_a, wins_b
