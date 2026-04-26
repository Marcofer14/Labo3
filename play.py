"""
play.py

Flujo principal actual:
  conectar bot principal -> jugar N partidas -> cerrar

Este archivo deja preparado el corte para un flujo futuro:
  conectar bot principal -> loop de comandos -> cerrar
"""

from __future__ import annotations

import argparse
import asyncio
import inspect

from poke_env.player.player import handle_threaded_coroutines

from login import (
    DEFAULT_TEAM_PATH,
    build_server_config,
    connect_main_bot,
    connect_opponent_bot,
)
from src.format_resolver import resolve_format


async def close_player(player) -> None:
    """Cierra el listener websocket de poke-env si esta disponible."""
    ps_client = getattr(player, "ps_client", None)
    if ps_client is not None and hasattr(ps_client, "stop_listening"):
        await handle_threaded_coroutines(ps_client.stop_listening(), ps_client.loop)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if item is not None]
    return [value]


def _pokemon_label(pokemon) -> str:
    species = getattr(pokemon, "species", None) or "?"
    fainted = getattr(pokemon, "fainted", False)
    hp = getattr(pokemon, "current_hp_fraction", None)
    status = getattr(pokemon, "status", None)

    details = []
    if fainted:
        details.append("KO")
    elif hp is not None:
        details.append(f"{hp * 100:.0f}%")
    if status is not None:
        details.append(getattr(status, "name", str(status)).lower())

    return f"{species} ({', '.join(details)})" if details else species


def _battle_snapshot(battle) -> str:
    own = ", ".join(_pokemon_label(p) for p in _as_list(getattr(battle, "active_pokemon", None)))
    opp = ", ".join(
        _pokemon_label(p) for p in _as_list(getattr(battle, "opponent_active_pokemon", None))
    )
    own = own or "sin activo"
    opp = opp or "sin activo"
    return f"propios: {own} | rival: {opp}"


def _battle_result(battle) -> str:
    if not battle.finished:
        return "EN CURSO"
    if battle.won:
        return "VICTORIA"
    if getattr(battle, "lost", False):
        return "DERROTA"
    return "EMPATE"


def _choice_message(choice) -> str:
    return getattr(choice, "message", None) or str(choice)


def enable_turn_logging(player, label: str) -> None:
    """Loguea estado y decision cada vez que el bot elige movimiento."""
    original_choose_move = player.choose_move
    last_logged_turn: dict[str, int] = {}

    def logged_choose_move(battle):
        tag = battle.battle_tag
        turn = battle.turn
        if tag not in last_logged_turn:
            print(f"\n  Batalla detectada: {tag}", flush=True)
        if last_logged_turn.get(tag) != turn:
            print(f"  Turno {turn} [{label}]: {_battle_snapshot(battle)}", flush=True)
            last_logged_turn[tag] = turn

        choice = original_choose_move(battle)
        if inspect.isawaitable(choice):
            async def log_async_choice():
                resolved = await choice
                print(f"    Decision [{label}]: {_choice_message(resolved)}", flush=True)
                return resolved

            return log_async_choice()

        print(f"    Decision [{label}]: {_choice_message(choice)}", flush=True)
        return choice

    player.choose_move = logged_choose_move


async def _wait_player_logged_in(player) -> None:
    await handle_threaded_coroutines(player.ps_client.logged_in.wait(), player.ps_client.loop)


async def wait_for_players_logged_in(players: list, timeout: float) -> None:
    try:
        await asyncio.wait_for(
            asyncio.gather(*(_wait_player_logged_in(player) for player in players)),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        pending = [
            player.username
            for player in players
            if not player.ps_client.logged_in.is_set()
        ]
        pending_text = ", ".join(pending) if pending else "desconocido"
        raise RuntimeError(f"Timeout esperando login de: {pending_text}") from exc


async def forfeit_unfinished_battles(player, reason: str) -> int:
    """Abandona batallas abiertas y devuelve cuantas encontro."""
    count = 0
    for battle in player.battles.values():
        if not battle.finished:
            print(
                f"  {reason}: {player.username} abandona batalla abierta "
                f"{battle.battle_tag} (turno {battle.turn})."
            )
            await handle_threaded_coroutines(
                player.ps_client.send_message("/forfeit", battle.battle_tag),
                player.ps_client.loop,
            )
            count += 1
    return count


async def wait_until_battles_closed(players: list, timeout: float = 10.0) -> None:
    end_time = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end_time:
        if all(battle.finished for player in players for battle in player.battles.values()):
            return
        await asyncio.sleep(0.5)


async def cleanup_startup_battles(players: list, wait_seconds: float, login_timeout: float) -> None:
    """Limpia batallas viejas antes de iniciar el flujo pedido."""
    print("\n  Conectando cuenta(s)...")
    await wait_for_players_logged_in(players, timeout=login_timeout)
    print("  Cuenta(s) conectada(s).")
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)

    total = 0
    for player in players:
        total += await forfeit_unfinished_battles(player, "Limpieza inicial")

    if total:
        print(f"  Limpieza inicial: esperando cierre de {total} batalla(s) vieja(s)...")
        await wait_until_battles_closed(players)
    else:
        print("  Limpieza inicial: no habia batallas abiertas.")

    for player in players:
        try:
            player.reset_battles()
        except EnvironmentError:
            print(
                f"  Aviso: {player.username} aun tiene batallas abiertas en poke-env; "
                "se ignoraran en el resumen si el servidor las cierra tarde."
            )


async def monitor_battles(players: list, stop_event: asyncio.Event) -> None:
    """Imprime resultado final cuando poke-env detecta cierre de batalla."""
    seen: dict[str, dict[str, object]] = {}

    def scan() -> None:
        for player in players:
            for tag, battle in player.battles.items():
                state = seen.setdefault(tag, {"finished": False})

                if battle.finished and not state["finished"]:
                    state["finished"] = True
                    print(f"  Final: {tag} -> {_battle_result(battle)} en {battle.turn} turnos.")

    while not stop_event.is_set():
        scan()
        await asyncio.sleep(0.5)

    scan()


async def run_challenge_games(main_bot, opponent, n_battles: int) -> None:
    print(f"\n  Iniciando {n_battles} partida(s)...\n")
    await main_bot.battle_against(opponent, n_battles=n_battles)


async def run_ladder_games(main_bot, n_battles: int) -> None:
    print(f"\n  Buscando {n_battles} partida(s) en ladder...\n")
    await main_bot.ladder(n_battles)


async def play(args) -> tuple[object, object | None, bool]:
    battle_format = resolve_format(args.format)
    server_cfg = build_server_config(args.server)

    main_bot = connect_main_bot(
        policy=args.p1,
        battle_format=battle_format,
        server=args.server,
        team_path=args.team,
    )
    opponent = None
    if args.mode == "challenge":
        opponent = connect_opponent_bot(
            policy=args.p2,
            battle_format=battle_format,
            server=args.server,
            team_path=args.team,
        )

    print("=" * 55)
    print("  VGC Bot - Play")
    print("=" * 55)
    print(f"  Servidor: {server_cfg.websocket_url}")
    print(f"  Formato:  {battle_format}")
    print(f"  Principal: {args.p1.upper()} ({main_bot.username})")
    if opponent is None:
        print("  Rival:     ladder")
    else:
        print(f"  Rival:     {args.p2.upper()} ({opponent.username})")
    print(f"  Partidas:  {args.n}")
    print("=" * 55)

    completed = False
    players = [main_bot] if opponent is None else [main_bot, opponent]
    stop_monitor = None
    monitor_task = None
    try:
        await cleanup_startup_battles(players, args.startup_cleanup_wait, args.login_timeout)
        enable_turn_logging(main_bot, "Principal")
        if opponent is not None:
            enable_turn_logging(opponent, "Rival")

        stop_monitor = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_battles(players, stop_monitor))
        runner = (
            run_challenge_games(main_bot, opponent, args.n)
            if opponent is not None
            else run_ladder_games(main_bot, args.n)
        )
        if args.battle_timeout is None:
            await runner
        else:
            await asyncio.wait_for(runner, args.battle_timeout)
        completed = True
    except asyncio.TimeoutError:
        print(f"\n  No se completaron las partidas antes de {args.battle_timeout:g} segundos.")
        await forfeit_unfinished_battles(main_bot, "Timeout")
        if opponent is not None:
            await forfeit_unfinished_battles(opponent, "Timeout")
        await wait_until_battles_closed(players, timeout=5.0)
    finally:
        if stop_monitor is not None and monitor_task is not None:
            stop_monitor.set()
            await monitor_task
        await close_player(main_bot)
        if opponent is not None:
            await close_player(opponent)

    return main_bot, opponent, completed


def print_results(main_bot, opponent, args, completed: bool) -> None:
    print("\n" + "=" * 55)
    print("  RESULTADOS")
    print("=" * 55)
    print(f"  Principal {args.p1.upper():<8} victorias: {main_bot.n_won_battles} / {args.n}")
    if opponent is not None:
        print(f"  Rival     {args.p2.upper():<8} victorias: {opponent.n_won_battles} / {args.n}")
    print("=" * 55)

    if opponent is None:
        if main_bot.n_won_battles:
            print(f"\n  Gano el bot principal ({args.p1.upper()}).")
        elif main_bot.n_lost_battles:
            print(f"\n  Perdio el bot principal ({args.p1.upper()}).")
        else:
            print("\n  Sin resultado.")
    elif main_bot.n_won_battles > opponent.n_won_battles:
        print(f"\n  Gano el bot principal ({args.p1.upper()}).")
    elif opponent.n_won_battles > main_bot.n_won_battles:
        print(f"\n  Gano el rival ({args.p2.upper()}).")
    else:
        print("\n  Empate.")

    if not completed:
        print("\n  La ejecucion termino sin completar todas las partidas.")

    if main_bot.battles:
        print("\n  Partidas jugadas:")
        for tag, battle in main_bot.battles.items():
            if not battle.finished:
                result = "EN CURSO"
            elif battle.won:
                result = "VICTORIA"
            else:
                result = "DERROTA"
            print(f"    [{result}] {tag} - {battle.turn} turnos")


def parse_args():
    parser = argparse.ArgumentParser(description="VGC Bot - conectar, jugar N partidas y cerrar")
    parser.add_argument("--n", type=int, default=3, help="Numero de partidas (default: 3)")
    parser.add_argument(
        "--mode",
        type=str,
        default="challenge",
        choices=["challenge", "ladder"],
        help="challenge entre dos bots o ladder contra rival aleatorio (default: challenge)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        help="Formato de batalla. Si no se pasa, usa VGC_FORMAT o el default.",
    )
    parser.add_argument(
        "--p1",
        type=str,
        default="greedy",
        choices=["random", "greedy"],
        help="Politica del bot principal (default: greedy)",
    )
    parser.add_argument(
        "--p2",
        type=str,
        default="random",
        choices=["random", "greedy"],
        help="Politica del segundo bot (default: random)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="official",
        help="Servidor: official o host:puerto local (default: official)",
    )
    parser.add_argument(
        "--team",
        type=str,
        default=str(DEFAULT_TEAM_PATH),
        help="Path al equipo en formato Showdown (default: team.txt)",
    )
    parser.add_argument(
        "--battle-timeout",
        type=float,
        default=None,
        help="Segundos maximos para esperar las partidas (default: sin limite)",
    )
    parser.add_argument(
        "--startup-cleanup-wait",
        type=float,
        default=2.0,
        help="Segundos a esperar tras login antes de limpiar batallas viejas (default: 2)",
    )
    parser.add_argument(
        "--login-timeout",
        type=float,
        default=30.0,
        help="Segundos maximos para esperar login inicial (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_bot, opponent, completed = asyncio.run(play(args))
    print_results(main_bot, opponent, args, completed)


if __name__ == "__main__":
    main()
