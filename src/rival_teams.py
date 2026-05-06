"""
src.rival_teams
─────────────────────────────────────────────────────────────────
Pool de equipos rivales rotando aleatoriamente entre batallas.

Uso:
    from src.rival_teams import RandomTeamPool, load_rival_pool
    from poke_env import RandomPlayer

    pool = load_rival_pool("rivalteams")          # carga todos los .txt
    rival = RandomPlayer(
        battle_format       = "gen9vgc2025regi",
        team                = pool,                # Teambuilder (no string)
        server_configuration= server_cfg,
    )

Cada vez que el rival inicia una batalla, `pool.yield_team()` devuelve
uno de los 10 equipos al azar (uniforme por defecto).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from poke_env.teambuilder import Teambuilder


class RandomTeamPool(Teambuilder):
    """
    Teambuilder que rota uniformemente entre N equipos en cada batalla.

    Args:
        team_strings: lista de strings en formato Pokepaste
        seed:         semilla del RNG (para reproducibilidad de tests)
        weights:      probabilidades por equipo (default: uniforme)
    """

    def __init__(
        self,
        team_strings: list[str],
        seed:    Optional[int]     = None,
        weights: Optional[list[float]] = None,
    ):
        if not team_strings:
            raise ValueError("RandomTeamPool: lista de equipos vacía.")

        # parse_showdown_team es heredado de Teambuilder
        self._parsed = [self.parse_showdown_team(t) for t in team_strings]
        self._raw    = list(team_strings)
        self._rng    = random.Random(seed)

        if weights is None:
            self._weights = [1.0 / len(team_strings)] * len(team_strings)
        else:
            if len(weights) != len(team_strings):
                raise ValueError("len(weights) debe coincidir con len(team_strings)")
            s = float(sum(weights))
            self._weights = [w / s for w in weights]

        # Estadística para debug / metrics
        self._yield_count: dict[int, int] = {i: 0 for i in range(len(team_strings))}
        self._last_idx: int = -1

    @property
    def num_teams(self) -> int:
        return len(self._parsed)

    @property
    def last_team_idx(self) -> int:
        """Índice del último equipo entregado (para logging)."""
        return self._last_idx

    @property
    def yield_counts(self) -> dict[int, int]:
        return dict(self._yield_count)

    def yield_team(self) -> str:
        idx = self._rng.choices(range(len(self._parsed)), weights=self._weights, k=1)[0]
        self._yield_count[idx] += 1
        self._last_idx = idx
        return self.join_team(self._parsed[idx])


# ── Loader ───────────────────────────────────────────────────────

def load_rival_pool(
    folder:  str | Path = "rivalteams",
    pattern: str        = "*.txt",
    seed:    Optional[int] = None,
    weights: Optional[list[float]] = None,
) -> RandomTeamPool:
    """
    Carga todos los .txt de la carpeta y devuelve un RandomTeamPool.

    Los archivos se ordenan alfabéticamente para que el orden sea determinista.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Carpeta de rivalteams no encontrada: {folder}")

    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron equipos ({pattern}) en {folder}")

    team_strings = []
    for f in files:
        with open(f, encoding="utf-8") as fp:
            team_strings.append(fp.read())

    return RandomTeamPool(team_strings, seed=seed, weights=weights)


# ── Entry point para verificación ────────────────────────────────

if __name__ == "__main__":
    pool = load_rival_pool(Path(__file__).resolve().parent.parent / "rivalteams")
    print(f"Cargados {pool.num_teams} equipos:")
    for i in range(pool.num_teams):
        # Showdown team text está en self._raw
        first_line = pool._raw[i].splitlines()[0]
        print(f"  [{i}]  {first_line}")
    print()
    # Smoke: yield 30 veces y conteo
    for _ in range(30):
        pool.yield_team()
    print("Yield counts (30 sorteos):")
    for k, v in pool.yield_counts.items():
        print(f"  team {k}: {v}")
