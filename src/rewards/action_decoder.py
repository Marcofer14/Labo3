"""
src.rewards.action_decoder
─────────────────────────────────────────────────────────────────
Decodifica los enteros del action space de DoublesEnv (Gen 9) en
acciones semánticas que los reward modules pueden interpretar.

Espacio de acciones (DoublesEnv, Gen 9), por slot:
  -2     : default  (poke-env elige)
  -1     : forfeit
   0     : pass / no-op
   1–6   : switch al Pokémon de equipo[N-1]
   7–26  : move 1..4 con target -2..+2  (4 moves × 5 targets)
  27–46  : + mega evolve  (no aplica en gen 9 pero ocupa el slot)
  47–66  : + z-move       (idem)
  67–86  : + dynamax      (idem)
  87–106 : + Terastal     (Gen 9)

Convención de target en poke-env:
  -2 = rival 2,  -1 = rival 1,  0 = ningún target específico (move spread / self)
  +1 = aliado 1, +2 = aliado 2 (autoinflingido o helping hand)
"""

from __future__ import annotations
from dataclasses import dataclass


# Constantes — derivadas del esquema de DoublesEnv
N_TARGETS_PER_MOVE = 5         # -2,-1,0,+1,+2
N_MOVES            = 4
MOVES_BLOCK_SIZE   = N_MOVES * N_TARGETS_PER_MOVE   # 20
MOVE_BLOCK_OFFSET  = 7         # primer entero de la región "move"


@dataclass
class DecodedAction:
    """Acción semántica decodificada de un entero del action space."""
    kind: str              # "pass" | "switch" | "move" | "forfeit" | "default"
    move_idx: int = -1     # 0..3  (slot de movimiento)
    target:   int = 0      # -2..+2
    tera:     bool = False
    switch_idx: int = -1   # 0..5  (índice en el equipo)
    raw:      int = 0

    def is_move(self) -> bool:    return self.kind == "move"
    def is_switch(self) -> bool:  return self.kind == "switch"

    def __repr__(self) -> str:
        if self.kind == "move":
            tag = "+TERA" if self.tera else ""
            return f"<Move m{self.move_idx} t{self.target:+d}{tag} raw={self.raw}>"
        if self.kind == "switch":
            return f"<Switch slot{self.switch_idx} raw={self.raw}>"
        return f"<{self.kind.upper()} raw={self.raw}>"


def decode_action(action: int) -> DecodedAction:
    """Convierte un entero de DoublesEnv en DecodedAction."""
    a = int(action)
    if a == -2: return DecodedAction(kind="default", raw=a)
    if a == -1: return DecodedAction(kind="forfeit", raw=a)
    if a ==  0: return DecodedAction(kind="pass",    raw=a)

    if 1 <= a <= 6:
        return DecodedAction(kind="switch", switch_idx=a - 1, raw=a)

    # Región de movimientos: 7..106
    rel = a - MOVE_BLOCK_OFFSET
    block = rel // MOVES_BLOCK_SIZE     # 0=normal, 1=mega, 2=z, 3=dyna, 4=tera
    within = rel % MOVES_BLOCK_SIZE
    move_idx = within // N_TARGETS_PER_MOVE
    target   = (within % N_TARGETS_PER_MOVE) - 2

    return DecodedAction(
        kind="move",
        move_idx=move_idx,
        target=target,
        tera=(block == 4),
        raw=a,
    )


def decode_pair(pair) -> tuple[DecodedAction, DecodedAction]:
    """Decodifica una tupla/lista de dos acciones (uno por slot activo)."""
    a, b = pair
    return decode_action(a), decode_action(b)
