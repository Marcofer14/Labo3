"""
src.training.league · Self-play league acotado.

Mantiene un pool de hasta N snapshots del modelo. Cuando se agrega
uno nuevo y el pool está lleno, **se evict el snapshot con menor
win-rate** (la idea: nos quedamos con los más duros).

Variantes posibles (configurables vía TrainingConfig.league_eviction):
  · "lowest_winrate"   → evict el más fácil de ganar (recomendado)
  · "highest_winrate"  → evict el más difícil (mantiene diversidad)
  · "fifo"             → evict el más viejo
  · "random"
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LeagueEntry:
    """Un snapshot del modelo dentro del league."""
    snapshot_id: int
    timestep:    int
    path:        str
    wins:        int = 0
    losses:      int = 0
    battles:     int = 0
    label:       str = ""        # ej: "stage3_t800k"

    @property
    def win_rate(self) -> float:
        return self.wins / self.battles if self.battles > 0 else 0.5    # neutral antes de jugar

    def record(self, won: bool) -> None:
        self.battles += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1


class SelfPlayLeague:
    """Pool acotado con eviction por win-rate (default)."""

    def __init__(
        self,
        max_size:       int  = 6,
        eviction_kind:  str  = "lowest_winrate",
        min_battles_for_eviction: int = 20,
        seed:           Optional[int] = None,
    ):
        self.max_size = max_size
        self.eviction_kind = eviction_kind
        self.min_battles_for_eviction = min_battles_for_eviction
        self._rng = random.Random(seed)

        self.entries: list[LeagueEntry] = []
        self._next_id: int = 0

    # ── API ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.entries)

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def add(self, path: str, timestep: int, label: str = "") -> LeagueEntry:
        """Agrega un snapshot. Si el pool está lleno, expulsa según política."""
        new = LeagueEntry(
            snapshot_id = self._next_id,
            timestep    = timestep,
            path        = path,
            label       = label or f"snapshot_{self._next_id}",
        )
        self._next_id += 1

        if len(self.entries) < self.max_size:
            self.entries.append(new)
            return new

        # Pool lleno → expulsar
        evict_idx = self._select_evict()
        evicted = self.entries[evict_idx]
        # Borrar archivo del disco para no acumular
        try:
            if os.path.isfile(evicted.path):
                os.remove(evicted.path)
        except OSError:
            pass
        self.entries[evict_idx] = new
        return new

    def sample(self) -> Optional[LeagueEntry]:
        """Devuelve un snapshot al azar uniforme. None si vacío."""
        if not self.entries:
            return None
        return self._rng.choice(self.entries)

    def sample_pfsp(self, alpha: float = 1.0) -> Optional[LeagueEntry]:
        """
        Prioritized Fictitious Self-Play: sampling proporcional a (1 - winrate) ^ alpha.
        Privilegia oponentes que aún nos están ganando o están parejos.
        """
        if not self.entries:
            return None
        weights = [max(1e-3, (1.0 - e.win_rate)) ** alpha for e in self.entries]
        return self._rng.choices(self.entries, weights=weights, k=1)[0]

    def record_result(self, snapshot_id: int, won: bool) -> None:
        for e in self.entries:
            if e.snapshot_id == snapshot_id:
                e.record(won)
                return

    def stats(self) -> list[dict]:
        return [
            {
                "id":        e.snapshot_id,
                "label":     e.label,
                "timestep":  e.timestep,
                "battles":   e.battles,
                "wins":      e.wins,
                "losses":    e.losses,
                "win_rate":  round(e.win_rate, 3),
                "path":      e.path,
            }
            for e in self.entries
        ]

    # ── Eviction policies ───────────────────────────────────────

    def _select_evict(self) -> int:
        kind = self.eviction_kind
        eligible = [
            i for i, e in enumerate(self.entries)
            if e.battles >= self.min_battles_for_eviction
        ]
        # Si nadie tiene suficientes batallas, fallback a FIFO
        if not eligible:
            return 0

        if kind == "lowest_winrate":
            return min(eligible, key=lambda i: self.entries[i].win_rate)
        if kind == "highest_winrate":
            return max(eligible, key=lambda i: self.entries[i].win_rate)
        if kind == "fifo":
            return min(eligible, key=lambda i: self.entries[i].timestep)
        if kind == "random":
            return self._rng.choice(eligible)
        # Default
        return min(eligible, key=lambda i: self.entries[i].win_rate)


# ── Helper para snapshot en disco ────────────────────────────────

def make_snapshot_dir(base: str | Path) -> Path:
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p
