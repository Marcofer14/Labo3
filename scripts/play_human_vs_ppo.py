"""Play one local Showdown battle: human browser vs PPO/RecurrentPPO bot."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import sys
from pathlib import Path


def _find_project_root() -> Path:
    """Return the repository root so this script works from scripts/ or root."""
    current_file = Path(__file__).resolve()
    for candidate in (current_file.parent, *current_file.parents):
        if (candidate / "login.py").is_file() and (candidate / "play.py").is_file() and (candidate / "src").is_dir():
            return candidate

    # Normal repository layout: <root>/scripts/play_human_vs_ppo.py.
    return current_file.parents[1]


PROJECT_ROOT = _find_project_root()
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)


def _project_path(path_value: str | os.PathLike[str]) -> str:
    """Resolve relative paths from cwd first, then from the repository root."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return str(cwd_path)

    return str(PROJECT_ROOT / path)


from login import build_server_config, connect_bot
from play import (
    cleanup_startup_battles,
    close_player,
    enable_turn_logging,
    forfeit_unfinished_battles,
    monitor_battles,
    wait_until_battles_closed,
)
from src.format_resolver import resolve_format


def _ppo_policy_kwargs(args: argparse.Namespace) -> dict:
    return {
        "checkpoint_path": args.ppo_checkpoint,
        "team_path": args.team,
        "device": args.ppo_device,
        "deterministic": args.ppo_deterministic,
        "strict_actions": args.ppo_strict_actions,
        "record_decisions": args.record_decisions,
    }


async def _run_challenge_method(method, opponent: str, n_challenges: int) -> None:
    """Call poke-env challenge helpers across minor signature differences."""
    signature = inspect.signature(method)
    parameters = signature.parameters
    kwargs = {}
    if "opponent" in parameters:
        kwargs["opponent"] = opponent
    if "username" in parameters:
        kwargs["username"] = opponent
    if "n_challenges" in parameters:
        kwargs["n_challenges"] = n_challenges
    if "n_battles" in parameters:
        kwargs["n_battles"] = n_challenges

    if kwargs:
        await method(**kwargs)
    else:
        await method(opponent, n_challenges)


async def play_human_vs_ppo(args: argparse.Namespace) -> bool:
    args.team = _project_path(args.team)
    args.ppo_checkpoint = _project_path(args.ppo_checkpoint)

    battle_format = resolve_format(args.format)
    server_cfg = build_server_config(args.server)
    password = args.bot_password
    if password is None:
        password = os.environ.get("SHOWDOWN_PASSWORD")

    bot = connect_bot(
        policy="ppo_recurrent",
        battle_format=battle_format,
        server=args.server,
        username=args.bot_name,
        password=password,
        team_path=args.team,
        policy_kwargs=_ppo_policy_kwargs(args),
    )

    print("=" * 70)
    print("  Partida local: humano vs PPO recurrente")
    print("=" * 70)
    print(f"  Servidor websocket: {server_cfg.websocket_url}")
    print(f"  Pagina local:       {args.browser_url}")
    print(f"  Formato:            {battle_format}")
    print(f"  Bot:                {args.bot_name}")
    print(f"  Humano esperado:    {args.human_name}")
    print(f"  Checkpoint PPO:     {args.ppo_checkpoint}")
    print(f"  Equipo del bot:     {args.team}")
    print("=" * 70)

    completed = False
    stop_monitor = None
    monitor_task = None
    try:
        await cleanup_startup_battles([bot], args.startup_cleanup_wait, args.login_timeout)
        enable_turn_logging(bot, "PPO")

        print("\n  Instrucciones:")
        print(f"    1. Abrir {args.browser_url}")
        print(f"    2. Elegir el nombre: {args.human_name}")
        print("    3. Importar/elegir un equipo valido para el formato.")
        if args.challenge_direction == "accept":
            print(f"    4. Retar a {args.bot_name} en el formato {battle_format}.")
            print("\n  El bot queda esperando tu challenge...\n")
        else:
            print(f"    4. Esperar el challenge enviado por {args.bot_name}.")
            print(f"\n  El bot enviara el challenge en {args.challenge_delay:g} segundos...\n")
            if args.challenge_delay > 0:
                await asyncio.sleep(args.challenge_delay)

        stop_monitor = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_battles([bot], stop_monitor))

        runner = (
            _run_challenge_method(bot.accept_challenges, args.human_name, 1)
            if args.challenge_direction == "accept"
            else _run_challenge_method(bot.send_challenges, args.human_name, 1)
        )
        await asyncio.wait_for(runner, timeout=args.battle_timeout)
        completed = True
        await wait_until_battles_closed([bot], timeout=args.post_battle_wait)
    except asyncio.TimeoutError:
        print(f"\n  Timeout: no se completo la partida antes de {args.battle_timeout:g} segundos.")
        await forfeit_unfinished_battles(bot, "Timeout")
        await wait_until_battles_closed([bot], timeout=5.0)
    finally:
        if stop_monitor is not None and monitor_task is not None:
            stop_monitor.set()
            await monitor_task
        await close_player(bot)

    print("\n" + "=" * 70)
    print("  Resultado")
    print("=" * 70)
    if bot.battles:
        for tag, battle in bot.battles.items():
            if not battle.finished:
                result = "EN CURSO"
            elif battle.won:
                result = "GANO PPO"
            elif getattr(battle, "lost", False):
                result = "GANO HUMANO"
            else:
                result = "SIN RESULTADO"
            print(f"  {tag}: {result} en {battle.turn} turnos")
    else:
        print("  No se registro ninguna batalla.")
    print("=" * 70)
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conecta el PPO recurrente al Showdown local para jugar contra un humano."
    )
    parser.add_argument("--server", default="showdown:8000", help="Host local del servidor Showdown.")
    parser.add_argument(
        "--browser-url",
        default="http://localhost:8000",
        help="URL que debe abrir el humano en el navegador.",
    )
    parser.add_argument("--format", default="gen9vgc2026regi", help="Formato de batalla.")
    parser.add_argument("--team", default="team.txt", help="Equipo Showdown del bot PPO.")
    parser.add_argument("--bot-name", default="PPOBotLocal", help="Nombre del bot en Showdown.")
    parser.add_argument(
        "--bot-password",
        default=None,
        help="Password del bot. En local normalmente puede omitirse.",
    )
    parser.add_argument(
        "--human-name",
        default="HumanoLocal",
        help="Nombre que debe elegir el jugador humano en el navegador.",
    )
    parser.add_argument(
        "--challenge-direction",
        choices=["accept", "send"],
        default="accept",
        help="accept: el bot espera tu reto. send: el bot te reta a vos.",
    )
    parser.add_argument(
        "--challenge-delay",
        type=float,
        default=15.0,
        help="Segundos de espera antes de enviar challenge cuando direction=send.",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        default="checkpoints/vgc_final.zip",
        help="Checkpoint .zip de PPO/RecurrentPPO.",
    )
    parser.add_argument("--ppo-device", default="cpu", help="Dispositivo de inferencia.")
    parser.add_argument(
        "--ppo-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usar acciones deterministicas del modelo.",
    )
    parser.add_argument(
        "--ppo-strict-actions",
        action="store_true",
        help="Fallar si el PPO emite una accion ilegal en vez de proyectarla a una legal.",
    )
    parser.add_argument(
        "--record-decisions",
        action="store_true",
        help="Guardar decisiones en memoria para debug durante la ejecucion.",
    )
    parser.add_argument("--login-timeout", type=float, default=60.0)
    parser.add_argument("--startup-cleanup-wait", type=float, default=1.0)
    parser.add_argument("--battle-timeout", type=float, default=3600.0)
    parser.add_argument("--post-battle-wait", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    completed = asyncio.run(play_human_vs_ppo(parse_args()))
    raise SystemExit(0 if completed else 1)


if __name__ == "__main__":
    main()
