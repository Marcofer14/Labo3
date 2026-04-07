"""
damage_calc.py
─────────────────────────────────────────────────────────────────
Calculadora de daño para VGC (Gen 9, nivel 50).

Implementa la fórmula oficial de daño de Pokémon gen 3+ con todos
los modificadores relevantes para el formato dobles:

  Daño = floor(floor(floor(2*Lv/5 + 2) * Pow * Atk/Def) / 50 + 2)
         × targets × weather × critical × random × STAB × type × burn × otros

Retorna:
  - Daño mínimo / esperado / máximo en puntos absolutos
  - % del HP máximo del defensor

Uso:
    from src.damage_calc import calc_damage, DamageResult
    from src.utils import load_all_data, parse_team, calc_all_stats

    data  = load_all_data()
    team  = parse_team("team.txt")
    kyogre_stats = calc_all_stats(team[0], data["pokemon"])

    result = calc_damage(
        attacker_stats  = kyogre_stats,
        attacker_types  = ["water"],
        attacker_item   = "mystic-water",
        move            = data["moves"]["water-spout"],
        defender_stats  = {...},
        defender_types  = ["fire"],
        conditions      = BattleConditions(weather="rain"),
        defender_hp_pct = 1.0,
    )
    print(result)  # DamageResult(min_dmg=..., avg_dmg=..., ...)
"""

from dataclasses import dataclass, field
from typing import Optional
from src.utils import get_effectiveness, is_stab

# ── Tipos de datos ────────────────────────────────────────────────

@dataclass
class BattleConditions:
    """Estado del campo al momento de calcular el daño."""
    weather:      str  = "none"    # "rain" | "sun" | "sand" | "snow" | "none"
    terrain:      str  = "none"    # "grassy" | "electric" | "psychic" | "misty" | "none"
    trick_room:   bool = False
    is_spread:    bool = False     # True si el movimiento golpea a ambos rivales
    is_critical:  bool = False

@dataclass
class DamageResult:
    """Resultado de un cálculo de daño."""
    move_name:       str
    attacker_name:   str
    defender_name:   str
    min_dmg:         int        # con roll de daño 0.85
    avg_dmg:         float      # con roll esperado 0.925
    max_dmg:         int        # con roll 1.00
    min_pct:         float      # min_dmg como % del HP máximo del defensor
    avg_pct:         float
    max_pct:         float
    effectiveness:   float      # multiplicador de tipo (0 / 0.25 / 0.5 / 1 / 2 / 4)
    is_stab:         bool
    ohko:            bool       # True si min_pct >= 1.0
    two_hit_ko:      bool       # True si min_pct >= 0.5

    def __str__(self) -> str:
        pct_str = f"{self.min_pct*100:.1f}% – {self.max_pct*100:.1f}%"
        eff_str = f"x{self.effectiveness}" if self.effectiveness != 1.0 else ""
        stab_str = " (STAB)" if self.is_stab else ""
        ko_str = " ← OHKO!" if self.ohko else (" ← 2HKO!" if self.two_hit_ko else "")
        return (
            f"  {self.move_name.upper()} → {self.defender_name}  "
            f"{pct_str} del HP  {eff_str}{stab_str}{ko_str}"
        )


# ── Modificadores de clima ────────────────────────────────────────

def _weather_modifier(move_type: str, weather: str) -> float:
    if weather == "rain":
        if move_type == "water":  return 1.5
        if move_type == "fire":   return 0.5
    elif weather == "sun":
        if move_type == "fire":   return 1.5
        if move_type == "water":  return 0.5
    return 1.0


# ── Modificadores de terreno ──────────────────────────────────────

def _terrain_modifier(move_type: str, terrain: str, attacker_grounded: bool = True) -> float:
    if not attacker_grounded:
        return 1.0
    if terrain == "electric" and move_type == "electric": return 1.3
    if terrain == "grassy"   and move_type == "grass":    return 1.3
    if terrain == "psychic"  and move_type == "psychic":  return 1.3
    return 1.0


# ── Modificador de item ofensivo ──────────────────────────────────

ITEM_TYPE_BOOST: dict[str, str] = {
    "mystic-water":     "water",
    "charcoal":         "fire",
    "miracle-seed":     "grass",
    "magnet":           "electric",
    "never-melt-ice":   "ice",
    "twisted-spoon":    "psychic",
    "black-belt":       "fighting",
    "poison-barb":      "poison",
    "soft-sand":        "ground",
    "sharp-beak":       "flying",
    "silver-powder":    "bug",
    "hard-stone":       "rock",
    "spell-tag":        "ghost",
    "dragon-fang":      "dragon",
    "black-glasses":    "dark",
    "metal-coat":       "steel",
    "silk-scarf":       "normal",
    "fairy-feather":    "fairy",
}

def _item_offensive_modifier(item: Optional[str], move_type: str, move_category: str) -> float:
    if item is None:
        return 1.0
    if item == "life-orb":
        return 1.3
    if item == "choice-specs" and move_category == "special":
        return 1.5
    if item == "choice-band" and move_category == "physical":
        return 1.5
    # Items de boost de tipo específico (+20%)
    boosted_type = ITEM_TYPE_BOOST.get(item)
    if boosted_type and boosted_type == move_type:
        return 1.2
    return 1.0


# ── Modificador de item defensivo ────────────────────────────────

def _item_defensive_modifier(item: Optional[str]) -> float:
    if item == "assault-vest":
        return 1.0   # sube sp.def — ya está en los stats calculados
    if item == "eviolite":
        return 1.0   # sube def y sp.def — ya en stats
    return 1.0


# ── Modificadores de habilidad (los más comunes en VGC) ──────────

def _ability_offensive_modifier(
    ability: str, move_type: str, move_category: str,
    attacker_types: list[str], defender_types: list[str]
) -> float:
    # Adaptability: STAB x2 en vez de x1.5 (el ajuste lo hacemos en calc_damage)
    # Technician: x1.5 si potencia <= 60
    return 1.0   # se expande según necesidad

def _ability_defensive_modifier(
    ability: str, move_type: str, effectiveness: float
) -> float:
    if ability == "thick-fat" and move_type in ("fire", "ice"):
        return 0.5
    if ability == "water-absorb" and move_type == "water":
        return 0.0      # inmune
    if ability == "flash-fire" and move_type == "fire":
        return 0.0
    if ability == "levitate" and move_type == "ground":
        return 0.0
    if ability == "storm-drain" and move_type == "water":
        return 0.0
    if ability == "lightning-rod" and move_type == "electric":
        return 0.0
    if ability == "motor-drive" and move_type == "electric":
        return 0.0
    if ability == "sap-sipper" and move_type == "grass":
        return 0.0
    if ability == "wonder-guard":
        return effectiveness if effectiveness > 1.0 else 0.0
    return 1.0


# ── Fórmula principal de daño ─────────────────────────────────────

def calc_damage(
    attacker_stats:   dict[str, int],
    attacker_types:   list[str],
    attacker_ability: str,
    attacker_item:    Optional[str],
    attacker_name:    str,
    move:             dict,
    defender_stats:   dict[str, int],
    defender_types:   list[str],
    defender_ability: str,
    defender_item:    Optional[str],
    defender_name:    str,
    type_chart:       dict,
    conditions:       Optional[BattleConditions] = None,
    attacker_level:   int = 50,
    defender_hp_max:  Optional[int] = None,
    attacker_stat_mods: dict[str, int] = None,   # -6 a +6 por stat
    defender_stat_mods: dict[str, int] = None,
) -> DamageResult:
    """
    Calcula el rango de daño de un movimiento según la fórmula oficial gen 3+.

    Los modificadores de stat (-6 a +6) siguen la tabla oficial:
        -6: x0.25,  -5: x0.28,  -4: x0.33,  -3: x0.40,
        -2: x0.50,  -1: x0.67,   0: x1.00,  +1: x1.50,
        +2: x2.00,  +3: x2.50,  +4: x3.00,  +5: x3.50,  +6: x4.00

    Args:
        attacker_stats:  stats finales del atacante (de calc_all_stats)
        attacker_types:  lista de tipos del atacante
        attacker_ability: nombre de habilidad del atacante
        attacker_item:   nombre del objeto del atacante (o None)
        attacker_name:   nombre del atacante (para el resultado)
        move:            dict del movimiento (de load_moves_data)
        defender_stats:  stats finales del defensor
        defender_types:  lista de tipos del defensor
        defender_ability: nombre de habilidad del defensor
        defender_item:   objeto del defensor (o None)
        defender_name:   nombre del defensor
        type_chart:      tabla de efectividad (de load_type_chart)
        conditions:      estado del campo
        attacker_level:  nivel del atacante (default 50)
        defender_hp_max: HP máximo del defensor (si None, usa defender_stats["hp"])
        attacker_stat_mods: modificadores de stat del atacante (ej. {"attack": +2})
        defender_stat_mods: modificadores de stat del defensor

    Returns:
        DamageResult con mínimo, promedio y máximo de daño y % de HP
    """
    if conditions is None:
        conditions = BattleConditions()
    if attacker_stat_mods is None:
        attacker_stat_mods = {}
    if defender_stat_mods is None:
        defender_stat_mods = {}

    move_type     = move["type"]
    move_category = move["category"]   # "physical" | "special" | "status"
    power         = move.get("power")
    move_name     = move["name"]

    # Movimientos de status o sin potencia → 0 daño
    if move_category == "status" or not power:
        hp_max = defender_hp_max or defender_stats["hp"]
        return DamageResult(
            move_name=move_name, attacker_name=attacker_name,
            defender_name=defender_name,
            min_dmg=0, avg_dmg=0.0, max_dmg=0,
            min_pct=0.0, avg_pct=0.0, max_pct=0.0,
            effectiveness=0.0, is_stab=False, ohko=False, two_hit_ko=False
        )

    # ── Ajuste de Power especial: Water Spout / Eruption ──────────
    # Estos moves tienen potencia variable según el % de HP del atacante
    attacker_hp_pct = attacker_stats["hp"] / (attacker_stats["hp"] or 1)
    if move_name in ("water-spout", "eruption"):
        power = max(1, int(power * attacker_hp_pct))

    # ── Stat ofensivo y defensivo ─────────────────────────────────
    if move_category == "physical":
        atk_stat = "attack"
        def_stat = "defense"
    else:
        atk_stat = "special-attack"
        def_stat = "special-defense"

    atk = attacker_stats[atk_stat] * _stat_modifier_mult(attacker_stat_mods.get(atk_stat, 0))
    dfn = defender_stats[def_stat]  * _stat_modifier_mult(defender_stat_mods.get(def_stat, 0))

    # Grassy Terrain reduce daño de Earthquake a la mitad (útil en este equipo)
    if move_name == "earthquake" and conditions.terrain == "grassy":
        power = power // 2

    # ── Fórmula base ──────────────────────────────────────────────
    base = (2 * attacker_level // 5 + 2) * power * atk // dfn // 50 + 2

    # ── Multiplicadores ───────────────────────────────────────────
    # 1. Spread: movimientos que golpean a los 2 rivales → x0.75
    spread_mult = 0.75 if conditions.is_spread else 1.0

    # 2. Clima
    weather_mult = _weather_modifier(move_type, conditions.weather)

    # 3. Crítico
    crit_mult = 1.5 if conditions.is_critical else 1.0

    # 4. STAB
    stab = is_stab(move_type, attacker_types)
    stab_mult = 1.5 if stab else 1.0
    # Adaptability sube STAB a x2
    if stab and attacker_ability == "adaptability":
        stab_mult = 2.0

    # 5. Efectividad de tipo
    # Tera: si el defensor usó tera, sus tipos cambian
    effectiveness = get_effectiveness(move_type, defender_types, type_chart)

    # 6. Habilidad defensiva
    ab_def_mult = _ability_defensive_modifier(defender_ability, move_type, effectiveness)
    effectiveness *= ab_def_mult  # habilidades de inmunidad → 0

    # 7. Quemadura (physical moves desde un Pokémon quemado → x0.5)
    # (se pasa como modificador externo si aplica)

    # 8. Item ofensivo
    item_off_mult = _item_offensive_modifier(attacker_item, move_type, move_category)

    # 9. Terreno
    terrain_mult = _terrain_modifier(move_type, conditions.terrain)

    # ── Ensamblar con rolls de daño (0.85 a 1.00 en 16 pasos) ────
    combined_no_random = (
        base
        * spread_mult
        * weather_mult
        * crit_mult
        * stab_mult
        * effectiveness
        * item_off_mult
        * terrain_mult
    )

    min_dmg = int(combined_no_random * 0.85)
    max_dmg = int(combined_no_random * 1.00)
    avg_dmg = combined_no_random * 0.925

    hp_max    = defender_hp_max or defender_stats["hp"]
    min_pct   = min_dmg / hp_max
    max_pct   = max_dmg / hp_max
    avg_pct   = avg_dmg / hp_max

    return DamageResult(
        move_name     = move_name,
        attacker_name = attacker_name,
        defender_name = defender_name,
        min_dmg       = max(0, min_dmg),
        avg_dmg       = max(0.0, avg_dmg),
        max_dmg       = max(0, max_dmg),
        min_pct       = max(0.0, min_pct),
        avg_pct       = max(0.0, avg_pct),
        max_pct       = max(0.0, max_pct),
        effectiveness = effectiveness,
        is_stab       = stab,
        ohko          = min_pct >= 1.0,
        two_hit_ko    = min_pct >= 0.5,
    )


def _stat_modifier_mult(stage: int) -> float:
    """
    Convierte modificador de stat (-6 a +6) a multiplicador oficial.
    Tabla: https://bulbapedia.bulbagarden.net/wiki/Stat#Stage_multipliers
    """
    stage = max(-6, min(6, stage))
    table = {
        -6: 2/8, -5: 2/7, -4: 2/6, -3: 2/5,
        -2: 2/4, -1: 2/3,  0: 1.0, +1: 3/2,
        +2: 4/2, +3: 5/2, +4: 6/2, +5: 7/2,  +6: 8/2,
    }
    return table[stage]


# ── Análisis de todas las acciones posibles del equipo ───────────

def calc_all_matchups(
    attacker: dict,
    attacker_stats: dict,
    attacker_types: list[str],
    defenders: list[tuple[str, dict, list[str]]],
    moves_data: dict,
    type_chart: dict,
    conditions: Optional[BattleConditions] = None,
) -> list[DamageResult]:
    """
    Calcula el daño de todos los movimientos de un Pokémon contra
    todos los defensores dados.

    Args:
        attacker:       dict del pokepaste (nombre, item, ability, moves...)
        attacker_stats: stats calculados (de calc_all_stats)
        attacker_types: tipos del atacante
        defenders:      lista de (nombre, stats, tipos) de posibles defensores
        moves_data:     dataset de movimientos
        type_chart:     tabla de tipos
        conditions:     estado del campo

    Returns:
        Lista de DamageResult ordenada de mayor a menor daño esperado
    """
    results = []
    for move_name in attacker["moves"]:
        move = moves_data.get(move_name)
        if not move or move.get("category") == "status":
            continue
        for def_name, def_stats, def_types in defenders:
            result = calc_damage(
                attacker_stats   = attacker_stats,
                attacker_types   = attacker_types,
                attacker_ability = attacker.get("ability", ""),
                attacker_item    = attacker.get("item"),
                attacker_name    = attacker["name"],
                move             = move,
                defender_stats   = def_stats,
                defender_types   = def_types,
                defender_ability = "",
                defender_item    = None,
                defender_name    = def_name,
                type_chart       = type_chart,
                conditions       = conditions,
            )
            results.append(result)

    results.sort(key=lambda r: r.avg_pct, reverse=True)
    return results


# ── Entry point de prueba ─────────────────────────────────────────

if __name__ == "__main__":
    from src.utils import load_all_data, parse_team, calc_all_stats
    from pathlib import Path

    print("Cargando datos...")
    data      = load_all_data()
    team_path = Path(__file__).resolve().parent.parent / "team.txt"
    team      = parse_team(team_path)

    # Stats del equipo
    team_stats = []
    for p in team:
        from src.utils import get_pokemon
        poke_info = get_pokemon(p["name"], data["pokemon"])
        stats     = calc_all_stats(p, data["pokemon"])
        types     = poke_info["types"] if poke_info else []
        team_stats.append((p, stats, types))

    # Ejemplo: Kyogre vs el resto del equipo
    kyogre, kyogre_stats, kyogre_types = team_stats[0]

    defenders = [(p["name"], s, t) for p, s, t in team_stats[1:]]

    print(f"\nMovimientos de {kyogre['name'].upper()} vs el resto del equipo:")
    print("─" * 60)

    rain = BattleConditions(weather="rain")
    results = calc_all_matchups(
        kyogre, kyogre_stats, kyogre_types,
        defenders, data["moves"], data["type_chart"],
        conditions=rain,
    )

    for r in results:
        print(r)
