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
import os

from poke_env.player.player import handle_threaded_coroutines

from login import (
    DEFAULT_TEAM_PATH,
    build_server_config,
    connect_main_bot,
    connect_opponent_bot,
    load_team,
    should_use_team,
)
from src.alphazero.showdown_simulator import ShowdownSimulationTracker, attach_simulation_tracking
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


def alphazero_policy_kwargs(args, tracker: ShowdownSimulationTracker | None = None) -> dict:
    return {
        "checkpoint_path": args.alphazero_checkpoint,
        "simulations": args.alphazero_simulations,
        "search_depth": args.alphazero_depth,
        "max_candidates": args.alphazero_max_candidates,
        "cpuct": args.alphazero_cpuct,
        "temperature": args.alphazero_temperature,
        "heuristic_weight": args.alphazero_heuristic_weight,
        "depth2_weight": args.alphazero_depth2_weight,
        "showdown_simulator_url": args.alphazero_simulator_url,
        "live_state_url": args.alphazero_live_state_url,
        "simulation_tracker": tracker,
        "simulator_timeout": args.alphazero_simulator_timeout,
        "simulator_max_choices": args.alphazero_simulator_max_choices,
        "simulator_opponent_policy": args.alphazero_simulator_opponent_policy,
        "simulator_robust_worst_weight": args.alphazero_simulator_robust_worst_weight,
        "require_showdown_simulator": args.alphazero_require_simulator,
        "device": args.alphazero_device,
    }


def cfr_policy_kwargs(args) -> dict:
    return {
        "checkpoint_path": args.cfr_checkpoint,
        "max_candidates": args.cfr_max_candidates,
        "temperature": args.cfr_temperature,
        "fallback": args.cfr_fallback,
        "neural_checkpoint_path": args.cfr_neural_checkpoint,
        "neural_weight": args.cfr_neural_weight,
        "min_average_visits": args.cfr_min_average_visits,
        "neural_device": args.cfr_neural_device,
    }


def ppo_policy_kwargs(args) -> dict:
    return {
        "checkpoint_path": args.ppo_checkpoint,
        "team_path": args.team,
        "device": args.ppo_device,
        "deterministic": args.ppo_deterministic,
        "strict_actions": args.ppo_strict_actions,
    }


def policy_kwargs_for(
    policy: str,
    args,
    tracker: ShowdownSimulationTracker | None = None,
) -> dict | None:
    if policy == "alphazero_mcts":
        return alphazero_policy_kwargs(args, tracker)
    if policy == "cfr":
        return cfr_policy_kwargs(args)
    if policy in {"ppo", "ppo_recurrent"}:
        return ppo_policy_kwargs(args)
    return None


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
    uses_alphazero = args.p1 == "alphazero_mcts" or args.p2 == "alphazero_mcts"
    uses_cfr = args.p1 == "cfr" or args.p2 == "cfr"
    uses_ppo = args.p1 in {"ppo", "ppo_recurrent"} or args.p2 in {"ppo", "ppo_recurrent"}
    tracker = None
    if (
        uses_alphazero
        and args.alphazero_depth >= 2
        and should_use_team(battle_format)
        and not args.alphazero_live_state_url
    ):
        tracker = ShowdownSimulationTracker(
            battle_format=battle_format,
            team_text=load_team(args.team),
        )

    main_bot = connect_main_bot(
        policy=args.p1,
        battle_format=battle_format,
        server=args.server,
        team_path=args.team,
        policy_kwargs=policy_kwargs_for(args.p1, args, tracker),
    )
    opponent = None
    if args.mode == "challenge":
        opponent = connect_opponent_bot(
            policy=args.p2,
            battle_format=battle_format,
            server=args.server,
            team_path=args.team,
            policy_kwargs=policy_kwargs_for(args.p2, args, tracker),
        )

    if tracker is not None:
        attach_simulation_tracking(main_bot, tracker)
        if opponent is not None:
            attach_simulation_tracking(opponent, tracker)

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
    if uses_alphazero:
        print(
            "  AlphaZero: "
            f"checkpoint={args.alphazero_checkpoint or 'sin checkpoint'} | "
            f"sims={args.alphazero_simulations} | depth={args.alphazero_depth}"
        )
        if args.alphazero_depth >= 2:
            simulator = args.alphazero_simulator_url or "desactivado"
            live_state = args.alphazero_live_state_url or "tracker por historial"
            print(
                f"  Simulador: {simulator} | "
                f"estado={live_state} | "
                f"policy={args.alphazero_simulator_opponent_policy} | "
                f"worst_weight={args.alphazero_simulator_robust_worst_weight} | "
                f"required={args.alphazero_require_simulator}"
            )
    if uses_cfr:
        print(
            "  CFR: "
            f"checkpoint={args.cfr_checkpoint or 'sin checkpoint'} | "
            f"neural={args.cfr_neural_checkpoint or 'sin red'} | "
            f"max_candidates={args.cfr_max_candidates} | "
            f"fallback={args.cfr_fallback}"
        )
    if uses_ppo:
        print(
            "  PPO recurrente: "
            f"checkpoint={args.ppo_checkpoint} | "
            f"device={args.ppo_device} | "
            f"deterministic={args.ppo_deterministic}"
        )
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
        choices=["random", "greedy", "alphazero_mcts", "cfr", "ppo", "ppo_recurrent"],
        help="Politica del bot principal (default: greedy)",
    )
    parser.add_argument(
        "--p2",
        type=str,
        default="random",
        choices=["random", "greedy", "alphazero_mcts", "cfr", "ppo", "ppo_recurrent"],
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
    parser.add_argument(
        "--alphazero-checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pt para la politica alphazero_mcts",
    )
    parser.add_argument(
        "--alphazero-device",
        type=str,
        default="cpu",
        help="Dispositivo para alphazero_mcts: cpu o cuda (default: cpu)",
    )
    parser.add_argument(
        "--alphazero-simulations",
        type=int,
        default=64,
        help="Simulaciones MCTS por decision (default: 64)",
    )
    parser.add_argument(
        "--alphazero-depth",
        type=int,
        default=1,
        help="Profundidad de busqueda MCTS (default: 1)",
    )
    parser.add_argument(
        "--alphazero-max-candidates",
        type=int,
        default=96,
        help="Maximo de acciones dobles candidatas a rankear (0 = todas, default: 96)",
    )
    parser.add_argument(
        "--alphazero-cpuct",
        type=float,
        default=1.5,
        help="Constante de exploracion PUCT para MCTS (default: 1.5)",
    )
    parser.add_argument(
        "--alphazero-temperature",
        type=float,
        default=0.0,
        help="Temperatura al elegir por visitas MCTS (default: 0, determinista)",
    )
    parser.add_argument(
        "--alphazero-heuristic-weight",
        type=float,
        default=0.75,
        help="Peso del prior tactico mientras el modelo aprende (default: 0.75)",
    )
    parser.add_argument(
        "--alphazero-depth2-weight",
        type=float,
        default=0.65,
        help="Peso del evaluador tactico depth 2 cuando --alphazero-depth >= 2 (default: 0.65)",
    )
    parser.add_argument(
        "--alphazero-simulator-url",
        type=str,
        default=os.environ.get("SHOWDOWN_SIMULATOR_URL", ""),
        help="URL del servicio Showdown real para depth >= 2 (default: SHOWDOWN_SIMULATOR_URL)",
    )
    parser.add_argument(
        "--alphazero-live-state-url",
        type=str,
        default=os.environ.get("SHOWDOWN_LIVE_STATE_URL", ""),
        help=(
            "URL del puente de estado vivo del servidor local Showdown. "
            "Si esta activo, depth >= 2 usa el estado interno real de la batalla "
            "en vez de reconstruir historial."
        ),
    )
    parser.add_argument(
        "--alphazero-simulator-timeout",
        type=float,
        default=10.0,
        help="Timeout por request al simulador Showdown real (default: 10)",
    )
    parser.add_argument(
        "--alphazero-simulator-max-choices",
        type=int,
        default=12,
        help="Maximo de respuestas por lado dentro del simulador real (default: 12)",
    )
    parser.add_argument(
        "--alphazero-simulator-opponent-policy",
        choices=["minimax", "mean", "robust"],
        default="robust",
        help="Como agregar respuestas del rival en el simulador real (default: robust)",
    )
    parser.add_argument(
        "--alphazero-simulator-robust-worst-weight",
        type=float,
        default=0.35,
        help="Peso del peor caso dentro de la politica robust (default: 0.35)",
    )
    parser.add_argument(
        "--alphazero-require-simulator",
        action="store_true",
        help="Falla si depth >= 2 no puede usar el simulador Showdown real.",
    )
    parser.add_argument(
        "--cfr-checkpoint",
        type=str,
        default=None,
        help="Checkpoint JSON para la politica CFR tabular.",
    )
    parser.add_argument(
        "--cfr-max-candidates",
        type=int,
        default=32,
        help="Maximo de acciones dobles candidatas para CFR (default: 32).",
    )
    parser.add_argument(
        "--cfr-temperature",
        type=float,
        default=0.0,
        help="Temperatura al samplear la estrategia promedio CFR (default: 0, determinista).",
    )
    parser.add_argument(
        "--cfr-fallback",
        choices=["heuristic", "random"],
        default="heuristic",
        help="Fallback cuando CFR no conoce el estado (default: heuristic).",
    )
    parser.add_argument(
        "--cfr-neural-checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pt de la red prior CFR para estados poco visitados.",
    )
    parser.add_argument(
        "--cfr-neural-weight",
        type=float,
        default=0.70,
        help="Peso de la red prior cuando se mezcla con la tabla CFR (default: 0.70).",
    )
    parser.add_argument(
        "--cfr-min-average-visits",
        type=int,
        default=3,
        help="Visitas minimas para usar estrategia promedio tabular pura (default: 3).",
    )
    parser.add_argument(
        "--cfr-neural-device",
        type=str,
        default="cpu",
        help="Dispositivo de la red CFR prior (default: cpu).",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        type=str,
        default="checkpoints/vgc_final.zip",
        help="Checkpoint .zip para PPO/RecurrentPPO (default: checkpoints/vgc_final.zip).",
    )
    parser.add_argument(
        "--ppo-device",
        type=str,
        default="cpu",
        help="Dispositivo para PPO/RecurrentPPO: cpu o cuda (default: cpu).",
    )
    parser.add_argument(
        "--ppo-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usar prediccion determinista para PPO/RecurrentPPO (default: true).",
    )
    parser.add_argument(
        "--ppo-strict-actions",
        action="store_true",
        help="Fallar si PPO genera una accion ilegal, en vez de caer a fallback legal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_bot, opponent, completed = asyncio.run(play(args))
    print_results(main_bot, opponent, args, completed)


if __name__ == "__main__":
    main()
