"""
src.rewards.state
─────────────────────────────────────────────────────────────────
Snapshots turn-to-turn de un DoubleBattle de poke-env.

Los reward modules consumen (prev_snapshot, curr_snapshot,
last_actions, data) y devuelven (valor, breakdown_dict).

El BattleStateTracker mantiene el snapshot anterior por battle_tag
y se limpia automáticamente cuando la batalla termina.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.weather       import Weather
from poke_env.battle.field         import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.pokemon       import Pokemon as PokemonObj


# ── Mapeos de enums a strings ─────────────────────────────────────

WEATHER_MAP = {
    Weather.RAINDANCE:     "rain",
    Weather.PRIMORDIALSEA: "rain",
    Weather.SUNNYDAY:      "sun",
    Weather.DESOLATELAND:  "sun",
    Weather.SANDSTORM:     "sandstorm",
    Weather.SNOWSCAPE:     "snow",
    Weather.HAIL:          "snow",
}

TERRAIN_MAP = {
    Field.GRASSY_TERRAIN:   "grassy",
    Field.ELECTRIC_TERRAIN: "electric",
    Field.PSYCHIC_TERRAIN:  "psychic",
    Field.MISTY_TERRAIN:    "misty",
}

# Stat boosts que tracking
BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


# ── Snapshots ─────────────────────────────────────────────────────

@dataclass
class PokemonSnapshot:
    species:        str            = ""
    hp_pct:         float          = 1.0
    fainted:        bool           = False
    status:         Optional[str]  = None
    types:          list[str]      = field(default_factory=list)
    boosts:         dict[str,int]  = field(default_factory=dict)
    item:           Optional[str]  = None
    terastallized:  bool           = False
    tera_type:      Optional[str]  = None
    base_atk:       int            = 0
    base_spa:       int            = 0
    base_def:       int            = 0
    base_spd:       int            = 0
    base_spe:       int            = 0
    speed_stat:     int            = 0
    moves_revealed: list[str]      = field(default_factory=list)
    last_move:      Optional[str]  = None

    @property
    def is_offensive_attacker(self) -> bool:
        """True si es atacante físico/especial relevante (base atk o spa > 90)."""
        return self.base_atk > 90 or self.base_spa > 90

    @property
    def is_physical_attacker(self) -> bool:
        return self.base_atk > 90

    @property
    def is_wall(self) -> bool:
        """True si es defensor: def o spd base > 100."""
        return self.base_def > 100 or self.base_spd > 100


@dataclass
class FieldSnapshot:
    weather:           str  = "none"
    terrain:           str  = "none"
    trick_room:        bool = False
    own_tailwind:      bool = False
    rival_tailwind:    bool = False
    own_reflect:       bool = False
    own_light_screen:  bool = False
    own_aurora_veil:   bool = False
    rival_reflect:     bool = False
    rival_light_screen: bool = False
    rival_aurora_veil: bool = False


@dataclass
class TurnSnapshot:
    turn:           int                 = 0
    finished:       bool                = False
    won:            bool                = False
    own_active:     list[PokemonSnapshot] = field(default_factory=list)
    rival_active:   list[PokemonSnapshot] = field(default_factory=list)
    own_bench:      list[PokemonSnapshot] = field(default_factory=list)
    rival_bench:    list[PokemonSnapshot] = field(default_factory=list)
    own_team_hp:    float = 0.0    # promedio HP% sobre 6
    rival_team_hp:  float = 0.0
    own_ko_count:   int   = 0
    rival_ko_count: int   = 0
    own_alive:      int   = 6
    rival_alive:    int   = 6
    field:          FieldSnapshot = field(default_factory=FieldSnapshot)


# ── Helpers de extracción ─────────────────────────────────────────

def _normalize_status(p: PokemonObj) -> Optional[str]:
    if not p or not p.status:
        return None
    return p.status.name.lower()


def _extract_boosts(p: PokemonObj) -> dict[str, int]:
    if not p or not getattr(p, "boosts", None):
        return {}
    out: dict[str, int] = {}
    for k in BOOST_KEYS:
        v = getattr(p.boosts, k, 0) if hasattr(p.boosts, k) else p.boosts.get(k, 0) if isinstance(p.boosts, dict) else 0
        if v:
            out[k] = v
    return out


def _last_move_id(p: PokemonObj) -> Optional[str]:
    """Move usado más recientemente por el Pokemon (si poke-env lo expone)."""
    moves = getattr(p, "moves", {}) or {}
    last = None
    for m in moves.values():
        # Cuando un movimiento se usó, su current_pp baja. El último usado es heurístico.
        pass
    # poke-env >= 0.8 expone last_move en algunos casos; si no, lo dejamos None
    return getattr(p, "last_move", None) and p.last_move.id


def _snapshot_pokemon(
    p: Optional[PokemonObj],
    team_stats: dict,    # nuestro equipo: {species: stats_dict}
    pokemon_data: dict,
    is_own: bool,
) -> PokemonSnapshot:
    if p is None:
        return PokemonSnapshot()

    species = p.species
    types   = [t.name.lower() for t in (p.types or []) if t]

    # Stats: reales si es nuestro, base stats si es rival
    if is_own and species in team_stats:
        stats = team_stats[species]
    else:
        info  = pokemon_data.get(species)
        stats = dict(info["stats"]) if info else {}

    moves_revealed = []
    for m in (getattr(p, "moves", {}) or {}).values():
        moves_revealed.append(m.id)

    return PokemonSnapshot(
        species        = species,
        hp_pct         = p.current_hp_fraction,
        fainted        = p.fainted,
        status         = _normalize_status(p),
        types          = types,
        boosts         = _extract_boosts(p),
        item           = getattr(p, "item", None),
        terastallized  = bool(getattr(p, "terastallized", False)),
        tera_type      = (p.tera_type.name.lower() if getattr(p, "tera_type", None) else None),
        base_atk       = stats.get("attack", 0),
        base_spa       = stats.get("special-attack", 0),
        base_def       = stats.get("defense", 0),
        base_spd       = stats.get("special-defense", 0),
        base_spe       = stats.get("speed", 0),
        speed_stat     = stats.get("speed", 0),
        moves_revealed = moves_revealed,
        last_move      = _last_move_id(p),
    )


def _snapshot_field(battle: DoubleBattle) -> FieldSnapshot:
    weather_enum = next(iter(battle.weather), None) if battle.weather else None
    weather_str  = WEATHER_MAP.get(weather_enum, "none") if weather_enum else "none"

    terrain_str = "none"
    for f in (battle.fields or {}):
        t = TERRAIN_MAP.get(f)
        if t:
            terrain_str = t
            break
    trick_room = Field.TRICK_ROOM in (battle.fields or {})

    # side_conditions: dicts {SideCondition: turns}
    own_sc   = battle.side_conditions     or {}
    rival_sc = battle.opponent_side_conditions or {}

    return FieldSnapshot(
        weather             = weather_str,
        terrain             = terrain_str,
        trick_room          = trick_room,
        own_tailwind        = SideCondition.TAILWIND      in own_sc,
        rival_tailwind      = SideCondition.TAILWIND      in rival_sc,
        own_reflect         = SideCondition.REFLECT       in own_sc,
        own_light_screen    = SideCondition.LIGHT_SCREEN  in own_sc,
        own_aurora_veil     = SideCondition.AURORA_VEIL   in own_sc,
        rival_reflect       = SideCondition.REFLECT       in rival_sc,
        rival_light_screen  = SideCondition.LIGHT_SCREEN  in rival_sc,
        rival_aurora_veil   = SideCondition.AURORA_VEIL   in rival_sc,
    )


def snapshot_battle(
    battle: DoubleBattle,
    team_stats:    dict,
    pokemon_data:  dict,
) -> TurnSnapshot:
    """
    Construye un TurnSnapshot completo de la batalla actual.

    `team_stats` es {species: stats_dict} de nuestro equipo (calc_all_stats por Pokemon).
    `pokemon_data` es el dict completo cargado de pokemon.json.
    """
    own_active = [
        _snapshot_pokemon(p, team_stats, pokemon_data, is_own=True)
        for p in (battle.active_pokemon or [None, None])
    ]
    rival_active = [
        _snapshot_pokemon(p, team_stats, pokemon_data, is_own=False)
        for p in (battle.opponent_active_pokemon or [None, None])
    ]
    while len(own_active)   < 2: own_active.append(PokemonSnapshot())
    while len(rival_active) < 2: rival_active.append(PokemonSnapshot())

    active_own_species   = {p.species for p in (battle.active_pokemon          or []) if p}
    active_rival_species = {p.species for p in (battle.opponent_active_pokemon or []) if p}

    own_bench = [
        _snapshot_pokemon(p, team_stats, pokemon_data, is_own=True)
        for p in (battle.team or {}).values()
        if p.species not in active_own_species
    ]
    rival_bench = [
        _snapshot_pokemon(p, team_stats, pokemon_data, is_own=False)
        for p in (battle.opponent_team or {}).values()
        if p.species not in active_rival_species
    ]

    own_ko    = sum(1 for p in (battle.team          or {}).values() if p.fainted)
    rival_ko  = sum(1 for p in (battle.opponent_team or {}).values() if p.fainted)
    own_alive = max(0, len(battle.team or {}) - own_ko)
    rival_alive = max(0, len(battle.opponent_team or {}) - rival_ko)

    own_hp_total   = sum(p.current_hp_fraction for p in (battle.team          or {}).values())
    rival_hp_total = sum(p.current_hp_fraction for p in (battle.opponent_team or {}).values())
    own_team_hp    = own_hp_total   / max(1, len(battle.team or {}))
    rival_team_hp  = rival_hp_total / max(1, len(battle.opponent_team or {}))

    return TurnSnapshot(
        turn          = battle.turn,
        finished      = battle.finished,
        won           = bool(battle.won) if battle.finished else False,
        own_active    = own_active,
        rival_active  = rival_active,
        own_bench     = own_bench,
        rival_bench   = rival_bench,
        own_team_hp   = own_team_hp,
        rival_team_hp = rival_team_hp,
        own_ko_count  = own_ko,
        rival_ko_count= rival_ko,
        own_alive     = own_alive,
        rival_alive   = rival_alive,
        field         = _snapshot_field(battle),
    )


# ── Tracker ───────────────────────────────────────────────────────

class BattleStateTracker:
    """Mantiene el TurnSnapshot anterior por battle_tag."""

    def __init__(self):
        self._snapshots: dict[str, TurnSnapshot] = {}

    def previous(self, tag: str) -> Optional[TurnSnapshot]:
        return self._snapshots.get(tag)

    def update(self, tag: str, snap: TurnSnapshot) -> None:
        self._snapshots[tag] = snap

    def clear(self, tag: str) -> None:
        self._snapshots.pop(tag, None)

    def reset(self) -> None:
        self._snapshots.clear()
