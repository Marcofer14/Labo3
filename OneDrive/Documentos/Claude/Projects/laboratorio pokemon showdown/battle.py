"""
battle.py
─────────────────────────────────────────────────────────────────
Script de batalla rápida — corre batallas sin necesitar el modelo RL.

Crea dos bots con el equipo del proyecto y los hace pelear entre sí
en el servidor local de Showdown. Sirve para:
  1. Verificar que la conexión al servidor funciona
  2. Ver cómo se mueven los bots (random / greedy)
  3. Testear el VGCEnv antes de entrenar

Uso:
  # 3 batallas random vs random en el servidor local:
  python battle.py

  # Más batallas, con otro formato:
  python battle.py --n 10 --format gen9vgc2025regg

  # Bot greedy vs random:
  python battle.py --p1 greedy --p2 random

  # Contra servidor Docker (desde fuera del contenedor):
  python battle.py --server localhost:8000

  # Desde dentro del contenedor Docker:
  python battle.py --server showdown:8000
"""

import argparse
import asyncio
import os
from pathlib import Path

from poke_env import RandomPlayer, MaxBasePowerPlayer, cross_evaluate
from poke_env.ps_client import ServerConfiguration
from src.format_resolver import resolve_format

def build_server_config(host: str) -> ServerConfiguration:
    """Construye el ServerConfiguration con la URL correcta de WebSocket."""
    # Showdown espera conexión WebSocket en /showdown/websocket
    ws_url = f"ws://{host}/showdown/websocket"
    auth_url = "https://play.pokemonshowdown.com/action.php?"
    return ServerConfiguration(ws_url, auth_url)


def make_player(kind: str, fmt: str, team: str, server_cfg: ServerConfiguration):
    """Crea un jugador del tipo indicado."""
    kwargs = dict(
        battle_format        = fmt,
        team                 = team,
        server_configuration = server_cfg,
        max_concurrent_battles = 1,
    )
    if kind == "random":
        return RandomPlayer(**kwargs)
    elif kind == "greedy":
        return MaxBasePowerPlayer(**kwargs)
    else:
        raise ValueError(f"Tipo desconocido: {kind}. Opciones: random, greedy")


async def run_battles(p1, p2, n: int):
    """Hace que p1 y p2 jueguen n batallas entre sí."""
    print(f"\n  Iniciando {n} batalla(s)...\n")
    await p1.battle_against(p2, n_battles=n)


def main():
    parser = argparse.ArgumentParser(description="VGC Bot — Batalla de prueba")
    parser.add_argument(
        "--n", type=int, default=3,
        help="Número de batallas (default: 3)"
    )
    parser.add_argument(
        "--format", type=str, default=None,
        help="Formato de batalla. Si no se pasa, usa VGC_FORMAT o el default automático."
    )
    parser.add_argument(
        "--p1", type=str, default="greedy", choices=["random", "greedy"],
        help="Tipo de bot para el jugador 1 (default: greedy)"
    )
    parser.add_argument(
        "--p2", type=str, default="random", choices=["random", "greedy"],
        help="Tipo de bot para el jugador 2 (default: random)"
    )
    parser.add_argument(
        "--server", type=str,
        default=os.environ.get("SHOWDOWN_SERVER", "localhost:8000"),
        help="Host:puerto del servidor Showdown (default: localhost:8000)"
    )
    args = parser.parse_args()
    args.format = resolve_format(args.format) #BUSCA EL FORMATO MAS ACTUAL
    
    # Leer equipo
    team_path = Path(__file__).resolve().parent / "team.txt"
    with open(team_path, encoding="utf-8") as f:
        team_str = f.read()

    server_cfg = build_server_config(args.server)

    print("═" * 55)
    print("  VGC Bot — Batalla de prueba")
    print("═" * 55)
    print(f"  Servidor: ws://{args.server}/showdown/websocket")
    print(f"  Formato:  {args.format}")
    print(f"  P1:       {args.p1.upper()}")
    print(f"  P2:       {args.p2.upper()}")
    print(f"  Batallas: {args.n}")
    print("═" * 55)

    p1 = make_player(args.p1, args.format, team_str, server_cfg)
    p2 = make_player(args.p2, args.format, team_str, server_cfg)

    # Correr las batallas
    asyncio.run(run_battles(p1, p2, args.n))

    # Resultados
    total = p1.n_won_battles + p2.n_won_battles
    print("\n" + "═" * 55)
    print("  RESULTADOS")
    print("═" * 55)
    print(f"  {args.p1.upper():<8}  victorias: {p1.n_won_battles} / {args.n}")
    print(f"  {args.p2.upper():<8}  victorias: {p2.n_won_battles} / {args.n}")
    print("═" * 55)

    if p1.n_won_battles > p2.n_won_battles:
        print(f"\n  ✓ Ganó {args.p1.upper()}!")
    elif p2.n_won_battles > p1.n_won_battles:
        print(f"\n  ✓ Ganó {args.p2.upper()}!")
    else:
        print(f"\n  ✓ Empate!")

    # Mostrar historial de batallas
    if p1.battles:
        print(f"\n  Batallas jugadas:")
        for tag, battle in p1.battles.items():
            result = "VICTORIA" if battle.won else "DERROTA"
            turns  = battle.turn
            print(f"    [{result}] {tag}  —  {turns} turnos")


if __name__ == "__main__":
    main()
