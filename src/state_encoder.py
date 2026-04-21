"""
state_encoder.py
─────────────────────────────────────────────────────────────────
Convierte el estado de una batalla VGC (proveniente de poke-env)
en un vector numérico normalizado [0, 1] para ser procesado por la
red neuronal del agente.

Estructura del vector de observación (total: ~380 valores):
─────────────────────────────────────────────────────────────────

  [A] PROPIOS EN CAMPO (2 Pokémon × ~60 valores = ~120)
      • HP actual / HP máximo                         → 1 float
      • Tipos (one-hot 18 tipos × 2 slots)            → 36 bools
      • Stats normalizados (atk/def/spa/spd/spe)      → 5 floats
      • Modificadores de stat (-6/+6 → 0/1) × 5      → 5 floats
      • Estado alterado (quem/par/sue/hie/env/+)      → 7 bools
      • Movimientos disponibles × 4:
          - Tipo (one-hot 18)                         → 18 bools
          - Potencia normalizada (0–250)               → 1 float
          - Categoría (phys/spec/status one-hot)      → 3 bools
          - PP restantes normalizados                  → 1 float
          - Efectividad vs cada rival (2 valores)     → 2 floats
      • Tera disponible (bool) + tipo tera (one-hot)  → 19 bools
      • Item (one-hot de items más comunes)            → 25 bools

  [B] RIVALES EN CAMPO (2 Pokémon × ~45 valores = ~90)
      • HP actual / HP máximo (visible)               → 1 float
      • Tipos (one-hot × 18 × 2 slots)                → 36 bools
      • Stats estimados normalizados (base stats)     → 5 floats
      • Modificadores de stat visibles × 5            → 5 floats
      • Estado alterado visible                       → 7 bools
      • Movimientos observados (max 4):
          - Tipo si se vio, else 0                    → 18 bools
      • Item revelado (one-hot) o desconocido         → 25 bools

  [C] CONDICIONES DEL CAMPO (~20 valores)
      • Clima activo (one-hot: none/rain/sun/sand/snow) → 5 bools
      • Terreno activo (none/grass/elec/psyc/mist)      → 5 bools
      • Trick Room activo                               → 1 bool
      • Número de turno normalizado (/ 50)              → 1 float

  [D] POKÉMON BANQUEADOS PROPIOS (2 restantes × ~10 = ~20)
      • HP normalizado                                  → 1 float
      • Tipos (one-hot × 18)                            → 18 bools
      • Estado alterado                                 → 7 bools
"""

import numpy as np
from typing import Optional

# ── Constantes de tipos ───────────────────────────────────────────

ALL_TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "dark", "steel", "fairy",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(ALL_TYPES)}
N_TYPES     = len(ALL_TYPES)   # 18

# ── Items más comunes en VGC (para encoding) ──────────────────────

VGC_COMMON_ITEMS = [
    "choice-scarf", "choice-specs", "choice-band",
    "assault-vest", "focus-sash", "life-orb",
    "leftovers", "rocky-helmet", "safety-goggles",
    "booster-energy", "mystic-water", "lum-berry",
    "sitrus-berry", "weakness-policy", "clear-amulet",
    "covert-cloak", "mirror-herb", "loaded-dice",
    "punching-glove", "throat-spray",
    "expert-belt", "wide-lens", "zoom-lens",
    "eviolite", "unknown",   # "unknown" = item no revelado
]
ITEM_TO_IDX  = {item: i for i, item in enumerate(VGC_COMMON_ITEMS)}
N_ITEMS      = len(VGC_COMMON_ITEMS)   # 25

# ── Estados alterados ─────────────────────────────────────────────

STATUS_CONDITIONS = ["burn", "paralysis", "sleep", "freeze", "poison", "toxic", "none"]
STATUS_TO_IDX = {s: i for i, s in enumerate(STATUS_CONDITIONS)}
N_STATUS = len(STATUS_CONDITIONS)   # 7

# ── Categorías de movimiento ──────────────────────────────────────

MOVE_CATEGORIES  = ["physical", "special", "status"]
N_MOVE_CATS      = 3

# ── Helpers de encoding ───────────────────────────────────────────

def _one_hot(index: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        v[index] = 1.0
    return v

def _encode_types(types: list[str]) -> np.ndarray:
    """Codifica hasta 2 tipos como vector de 36 floats (18 × 2 slots)."""
    vec = np.zeros(N_TYPES * 2, dtype=np.float32)
    for slot, t in enumerate(types[:2]):
        idx = TYPE_TO_IDX.get(t, -1)
        if idx >= 0:
            vec[slot * N_TYPES + idx] = 1.0
    return vec

def _encode_item(item: Optional[str]) -> np.ndarray:
    """One-hot de item. Desconocido → slot 'unknown'."""
    key = item if item in ITEM_TO_IDX else "unknown"
    return _one_hot(ITEM_TO_IDX[key], N_ITEMS)

def _encode_status(status: Optional[str]) -> np.ndarray:
    """One-hot del estado alterado. None / no estado → 'none'."""
    key = status if status in STATUS_TO_IDX else "none"
    return _one_hot(STATUS_TO_IDX[key], N_STATUS)

def _encode_stat_modifier(stage: int) -> float:
    """Normaliza un modificador de stat (-6 a +6) al rango [0, 1]."""
    return (stage + 6) / 12.0

def _encode_move(
    move: Optional[dict],
    effectiveness_vs: list[float],
) -> np.ndarray:
    """
    Codifica un movimiento individual como vector.

    Incluye:
      • tipo (one-hot 18)
      • potencia normalizada (power / 250)
      • categoría (one-hot 3)
      • PP restantes normalizados
      • efectividad vs cada rival en campo (2 floats)

    Total: 18 + 1 + 3 + 1 + 2 = 25 valores
    """
    if move is None:
        return np.zeros(18 + 1 + 3 + 1 + 2, dtype=np.float32)

    type_vec = _one_hot(TYPE_TO_IDX.get(move.get("type", ""), -1), N_TYPES)
    power    = float(move.get("power") or 0) / 250.0
    cat_vec  = _one_hot(MOVE_CATEGORIES.index(move.get("category", "status")), N_MOVE_CATS)
    pp_max   = float(move.get("pp") or 1)
    pp_left  = float(move.get("pp_left", move.get("pp") or 1))
    pp_norm  = pp_left / pp_max if pp_max > 0 else 0.0

    eff_vec = np.array(effectiveness_vs[:2] + [0.0] * (2 - len(effectiveness_vs)),
                       dtype=np.float32) / 4.0   # normalizar sobre 4x máximo

    return np.concatenate([type_vec, [power], cat_vec, [pp_norm], eff_vec])


# ── Encoder principal ─────────────────────────────────────────────

class StateEncoder:
    """
    Transforma el estado de una batalla VGC en un vector numpy [0, 1].

    Uso con poke-env:
        encoder   = StateEncoder(type_chart, moves_data)
        obs_array = encoder.encode(battle)

    Uso standalone (pruebas):
        obs_array = encoder.encode_manual(my_pokemon, enemy_pokemon, conditions)
    """

    def __init__(self, type_chart: dict, moves_data: dict):
        self.type_chart  = type_chart
        self.moves_data  = moves_data
        self.obs_size    = self._calculate_obs_size()

    def _calculate_obs_size(self) -> int:
        """
        Calcula el tamaño total del vector de observación.
        Sirve para definir el observation_space en Gymnasium.
        """
        # Pokémon propio en campo
        own_poke  = (
            1           # HP %
            + N_TYPES*2 # tipos (2 slots)
            + 5         # stats normalizados
            + 5         # modificadores de stat
            + N_STATUS  # estado alterado
            + 4 * 25    # 4 movimientos × 25 valores cada uno
            + 1         # tera disponible
            + N_TYPES   # tipo tera (one-hot)
            + N_ITEMS   # item
        )
        # Pokémon rival en campo (sin item ni tera en algunos casos)
        rival_poke = (
            1
            + N_TYPES*2
            + 5
            + 5
            + N_STATUS
            + 4 * 25
            + N_ITEMS
        )
        # Condiciones del campo
        conditions = (
            5   # clima (5 opciones)
            + 5 # terreno (5 opciones)
            + 1 # trick room
            + 1 # turno
        )
        # Pokémon banqueados propios (2)
        bench = 2 * (1 + N_TYPES*2 + N_STATUS)

        return own_poke * 2 + rival_poke * 2 + conditions + bench

    def _get_effectiveness_vs_field(
        self, move_type: str, rivals_types: list[list[str]]
    ) -> list[float]:
        """
        Retorna la efectividad del tipo de movimiento contra cada rival en campo.
        Normalizado a [0,1] (dividido por 4.0 máximo).
        """
        result = []
        from src.utils import get_effectiveness
        for r_types in rivals_types:
            mult = get_effectiveness(move_type, r_types, self.type_chart)
            result.append(mult)
        return result

    def encode_pokemon(
        self,
        hp_pct:        float,
        types:         list[str],
        stats:         dict[str, float],
        stat_mods:     dict[str, int],
        status:        Optional[str],
        moves:         list[dict],
        tera_available: bool,
        tera_type:     Optional[str],
        item:          Optional[str],
        rival_types:   list[list[str]],
        is_ally:       bool = True,
    ) -> np.ndarray:
        """
        Codifica un Pokémon individual en campo como vector numérico.

        Args:
            hp_pct:         HP actual / HP máximo (0.0 a 1.0)
            types:          lista de tipos del Pokémon
            stats:          dict de stats normalizados (entre 0 y 1)
            stat_mods:      modificadores de stat activos (-6 a +6)
            status:         estado alterado activo o None
            moves:          lista de dicts de movimientos (con pp_left si disponible)
            tera_available: si todavía no usó Terastal
            tera_type:      tipo tera (o None si ya lo usó / desconocido)
            item:           objeto que porta (o None)
            rival_types:    tipos de los rivales en campo (para calcular efectividad)
            is_ally:        True si es nuestro Pokémon, False si es del rival

        Returns:
            np.ndarray de longitud variable (ver _calculate_obs_size)
        """
        STAT_ORDER = ["attack", "defense", "special-attack", "special-defense", "speed"]
        MAX_STATS  = [400, 350, 400, 350, 400]  # máximos aproximados a nivel 50

        parts = [
            np.array([hp_pct], dtype=np.float32),
            _encode_types(types),
            np.array(
                [stats.get(s, 0) / mx for s, mx in zip(STAT_ORDER, MAX_STATS)],
                dtype=np.float32
            ),
            np.array(
                [_encode_stat_modifier(stat_mods.get(s, 0)) for s in STAT_ORDER],
                dtype=np.float32
            ),
            _encode_status(status),
        ]

        # Movimientos (4 slots, paddeados con zeros si hay menos)
        for i in range(4):
            move = moves[i] if i < len(moves) else None
            if move:
                eff_vs = self._get_effectiveness_vs_field(
                    move.get("type", ""), rival_types
                )
            else:
                eff_vs = [0.0, 0.0]
            parts.append(_encode_move(move, eff_vs))

        if is_ally:
            parts.append(np.array([float(tera_available)], dtype=np.float32))
            tera_idx = TYPE_TO_IDX.get(tera_type or "", -1)
            parts.append(_one_hot(tera_idx, N_TYPES))

        parts.append(_encode_item(item))

        return np.concatenate(parts)

    def encode_manual(
        self,
        own_field:    list[dict],
        rival_field:  list[dict],
        benched_own:  list[dict],
        conditions:   dict,
    ) -> np.ndarray:
        """
        Codifica el estado de batalla completo como vector numpy.

        Args:
            own_field:   lista de hasta 2 dicts con datos de nuestros Pokémon en campo:
                         { "hp_pct", "types", "stats", "stat_mods", "status",
                           "moves", "tera_available", "tera_type", "item" }
            rival_field: lista de hasta 2 dicts con datos de los rivales en campo
            benched_own: lista de hasta 2 dicts de banqueados propios
                         { "hp_pct", "types", "status" }
            conditions:  { "weather", "terrain", "trick_room", "turn" }

        Returns:
            np.ndarray shape (obs_size,) con valores en [0, 1]
        """
        parts = []
        rival_types = [p.get("types", []) for p in rival_field]

        # Tamaños exactos por tipo de Pokémon (calculados en _calculate_obs_size)
        # ally:  1+36+5+5+7+100+1+18+25 = 198
        # rival: 1+36+5+5+7+100+25      = 179
        OWN_POKE_SIZE   = 1 + N_TYPES*2 + 5 + 5 + N_STATUS + 4*25 + 1 + N_TYPES + N_ITEMS
        RIVAL_POKE_SIZE = 1 + N_TYPES*2 + 5 + 5 + N_STATUS + 4*25 + N_ITEMS

        # Propios en campo
        for i in range(2):
            if i < len(own_field):
                p = own_field[i]
                vec = self.encode_pokemon(
                    hp_pct         = p.get("hp_pct", 0.0),
                    types          = p.get("types", []),
                    stats          = p.get("stats", {}),
                    stat_mods      = p.get("stat_mods", {}),
                    status         = p.get("status"),
                    moves          = [self.moves_data.get(m, {"name": m}) for m in p.get("moves", [])],
                    tera_available = p.get("tera_available", True),
                    tera_type      = p.get("tera_type"),
                    item           = p.get("item"),
                    rival_types    = rival_types,
                    is_ally        = True,
                )
            else:
                vec = np.zeros(OWN_POKE_SIZE, dtype=np.float32)
            parts.append(vec)

        # Rivales en campo
        own_types = [p.get("types", []) for p in own_field]
        for i in range(2):
            if i < len(rival_field):
                p = rival_field[i]
                vec = self.encode_pokemon(
                    hp_pct         = p.get("hp_pct", 0.0),
                    types          = p.get("types", []),
                    stats          = p.get("stats", {}),
                    stat_mods      = p.get("stat_mods", {}),
                    status         = p.get("status"),
                    moves          = [self.moves_data.get(m, {"name": m}) for m in p.get("moves", [])],
                    tera_available = p.get("tera_available", True),
                    tera_type      = p.get("tera_type"),
                    item           = p.get("item"),
                    rival_types    = own_types,
                    is_ally        = False,
                )
            else:
                vec = np.zeros(RIVAL_POKE_SIZE, dtype=np.float32)
            parts.append(vec)

        # Condiciones del campo
        WEATHERS  = ["none", "rain", "sun", "sandstorm", "snow"]
        TERRAINS  = ["none", "grassy", "electric", "psychic", "misty"]
        weather   = conditions.get("weather", "none")
        terrain   = conditions.get("terrain", "none")
        turn      = conditions.get("turn", 0)

        cond_vec = np.concatenate([
            _one_hot(WEATHERS.index(weather) if weather in WEATHERS else 0, 5),
            _one_hot(TERRAINS.index(terrain) if terrain in TERRAINS else 0, 5),
            np.array([float(conditions.get("trick_room", False))], dtype=np.float32),
            np.array([min(turn, 50) / 50.0], dtype=np.float32),
        ])
        parts.append(cond_vec)

        # Banqueados propios
        for i in range(2):
            if i < len(benched_own):
                b = benched_own[i]
                bench_vec = np.concatenate([
                    np.array([b.get("hp_pct", 0.0)], dtype=np.float32),
                    _encode_types(b.get("types", [])),
                    _encode_status(b.get("status")),
                ])
            else:
                bench_vec = np.zeros(1 + N_TYPES*2 + N_STATUS, dtype=np.float32)
            parts.append(bench_vec)

        return np.concatenate(parts)

    def get_obs_shape(self) -> tuple[int]:
        """Retorna la forma del vector de observación para Gymnasium."""
        # Hacer un encode dummy para obtener el tamaño real
        dummy = self.encode_manual(
            own_field   = [],
            rival_field = [],
            benched_own = [],
            conditions  = {},
        )
        return dummy.shape


# ── Entry point de prueba ─────────────────────────────────────────

if __name__ == "__main__":
    from src.utils import load_all_data, parse_team, calc_all_stats, get_pokemon
    from pathlib import Path

    data      = load_all_data()
    team_path = Path(__file__).resolve().parent.parent / "team.txt"
    team      = parse_team(team_path)

    encoder = StateEncoder(data["type_chart"], data["moves"])

    # Simular turno 1: Kyogre + Calyrex vs Incineroar + Rillaboom (uso training interno)
    kyogre_info = get_pokemon("kyogre", data["pokemon"])
    kyogre_stats = calc_all_stats(team[0], data["pokemon"])

    obs = encoder.encode_manual(
        own_field = [
            {
                "hp_pct": 1.0,
                "types":  ["water"],
                "stats":  kyogre_stats,
                "stat_mods": {},
                "status": None,
                "moves":  team[0]["moves"],
                "tera_available": True,
                "tera_type": "grass",
                "item": "mystic-water",
            }
        ],
        rival_field = [
            {"hp_pct": 1.0, "types": ["fire"], "stats": {}, "stat_mods": {},
             "status": None, "moves": [], "tera_available": True, "tera_type": None, "item": None},
        ],
        benched_own = [
            {"hp_pct": 1.0, "types": ["psychic", "ghost"], "status": None},
        ],
        conditions = {"weather": "rain", "terrain": "none", "trick_room": False, "turn": 1},
    )

    print(f"Forma del vector de observación: {obs.shape}")
    print(f"Min valor: {obs.min():.4f}  Max valor: {obs.max():.4f}")
    print(f"Primeros 20 valores: {obs[:20]}")
    print(f"✓ StateEncoder funciona correctamente")
