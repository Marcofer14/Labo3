"""
login.py

Construccion explicita de bots autenticados.

Importar este modulo no conecta nada: el caller debe llamar a
connect_main_bot(...) o connect_opponent_bot(...) cuando quiera iniciar el
flujo.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from poke_env import AccountConfiguration, MaxBasePowerPlayer, RandomPlayer
from poke_env.ps_client import ServerConfiguration, ShowdownServerConfiguration


BOT_USERNAME = "Laboratorio3IA"
BOT_PASSWORD = "123456789"
OPPONENT_USERNAME = "Laboratorio3IA-B"
OPPONENT_PASSWORD = "123456789"
DEFAULT_TEAM_PATH = Path(__file__).resolve().parent / "team.txt"
OFFICIAL_SERVER_ALIASES = {"official", "showdown", "pokemonshowdown", "ps"}


def build_server_config(host: str) -> ServerConfiguration:
    """Construye la configuracion del servidor Showdown oficial o local."""
    if host.lower() in OFFICIAL_SERVER_ALIASES:
        return ShowdownServerConfiguration

    ws_url = f"ws://{host}/showdown/websocket"
    auth_url = "https://play.pokemonshowdown.com/action.php?"
    return ServerConfiguration(ws_url, auth_url)


def load_team(team_path: str | Path = DEFAULT_TEAM_PATH) -> str:
    """Lee team.txt en formato Pokepaste/Showdown."""
    with open(team_path, encoding="utf-8") as f:
        return f.read()


def should_use_team(battle_format: str) -> bool:
    """Los formatos random de Showdown generan equipo del lado del servidor."""
    return "random" not in battle_format.lower()


def make_policy_player(
    kind: str,
    *,
    battle_format: str,
    team: str | None,
    server_configuration: ServerConfiguration,
    account_configuration: AccountConfiguration | None = None,
    policy_kwargs: dict | None = None,
):
    """Crea un jugador random o greedy con parametros compartidos."""
    kwargs = dict(
        account_configuration=account_configuration,
        battle_format=battle_format,
        team=team,
        server_configuration=server_configuration,
        max_concurrent_battles=1,
    )

    if kind == "random":
        return RandomPlayer(**kwargs)
    if kind == "greedy":
        return MaxBasePowerPlayer(**kwargs)
    if kind == "alphazero_mcts":
        from src.alphazero.player import AlphaZeroMCTSPlayer

        return AlphaZeroMCTSPlayer(**kwargs, **(policy_kwargs or {}))

    raise ValueError(f"Tipo desconocido: {kind}. Opciones: random, greedy, alphazero_mcts")


def make_account(username: str, password: str | None) -> AccountConfiguration:
    """Construye credenciales de una cuenta de Showdown."""
    return AccountConfiguration(username, password)


def make_anonymous_account(prefix: str = "Rival") -> AccountConfiguration:
    """Crea una identidad anonima y unica para evitar choques de nombres."""
    username = f"{prefix}-{uuid.uuid4().hex[:8]}"
    return AccountConfiguration(username, None)


def connect_bot(
    *,
    policy: str,
    battle_format: str,
    server: str,
    username: str,
    password: str | None,
    team_path: str | Path = DEFAULT_TEAM_PATH,
    policy_kwargs: dict | None = None,
):
    """Crea un bot autenticado con la politica indicada."""
    account = make_account(username, password)
    server_cfg = build_server_config(server)
    team = load_team(team_path) if should_use_team(battle_format) else None

    return make_policy_player(
        policy,
        battle_format=battle_format,
        team=team,
        server_configuration=server_cfg,
        account_configuration=account,
        policy_kwargs=policy_kwargs,
    )


def connect_main_bot(
    *,
    policy: str,
    battle_format: str,
    server: str,
    team_path: str | Path = DEFAULT_TEAM_PATH,
    policy_kwargs: dict | None = None,
):
    """
    Crea/conecta el bot principal con identidad fija.

    SHOWDOWN_USERNAME y SHOWDOWN_PASSWORD permiten override local sin cambiar el
    codigo, pero los defaults son la identidad fija del proyecto.
    """
    return connect_bot(
        policy=policy,
        battle_format=battle_format,
        server=server,
        username=os.environ.get("SHOWDOWN_USERNAME", BOT_USERNAME),
        password=os.environ.get("SHOWDOWN_PASSWORD", BOT_PASSWORD),
        team_path=team_path,
        policy_kwargs=policy_kwargs,
    )


def connect_opponent_bot(
    *,
    policy: str,
    battle_format: str,
    server: str,
    team_path: str | Path = DEFAULT_TEAM_PATH,
    policy_kwargs: dict | None = None,
):
    """
    Crea/conecta el segundo bot con cuenta fija.

    SHOWDOWN_OPPONENT_USERNAME y SHOWDOWN_OPPONENT_PASSWORD permiten override.
    """
    return connect_bot(
        policy=policy,
        battle_format=battle_format,
        server=server,
        username=os.environ.get("SHOWDOWN_OPPONENT_USERNAME", OPPONENT_USERNAME),
        password=os.environ.get("SHOWDOWN_OPPONENT_PASSWORD", OPPONENT_PASSWORD),
        team_path=team_path,
        policy_kwargs=policy_kwargs,
    )
