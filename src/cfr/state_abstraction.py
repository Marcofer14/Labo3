"""Compact public-state abstractions for tabular CFR.

The real VGC state is far too large for exact CFR. These helpers group similar
public states into stable keys so regret tables can be reused across battles.
"""

from __future__ import annotations

from typing import Any


def compact_name(value: Any) -> str:
    """Extract a stable short id/name from Showdown/PokeEnv objects."""
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("id", "name", "species", "baseSpecies", "base_species"):
            item = value.get(key)
            if item and item is not value:
                text = compact_name(item)
                if text:
                    return text
        return ""
    for attr in ("id", "name", "species", "base_species", "baseSpecies"):
        try:
            item = getattr(value, attr)
        except Exception:
            item = None
        if item and item is not value:
            text = compact_name(item)
            if text:
                return text
    return str(value or "")


def normalize_name(value: Any) -> str:
    text = compact_name(value).lower().strip()
    for char in (" ", "_"):
        text = text.replace(char, "-")
    return "".join(ch for ch in text if ch.isalnum() or ch == "-").strip("-") or "unknown"


def hp_bucket(value: Any) -> str:
    try:
        hp = float(value or 0.0)
    except (TypeError, ValueError):
        hp = 0.0
    if hp > 1.0:
        hp /= 100.0
    if hp <= 0.0:
        return "ko"
    if hp <= 0.25:
        return "low"
    if hp <= 0.50:
        return "mid"
    if hp <= 0.75:
        return "high"
    return "full"


def turn_bucket(turn: Any) -> str:
    try:
        value = int(turn or 0)
    except (TypeError, ValueError):
        value = 0
    if value <= 1:
        return "t1"
    if value <= 3:
        return "t2-3"
    if value <= 6:
        return "t4-6"
    if value <= 12:
        return "t7-12"
    return "late"


def _status(value: Any) -> str:
    if not value:
        return "ok"
    return normalize_name(getattr(value, "name", value))


def _offline_mon_token(mon: dict[str, Any] | None) -> str:
    if not mon:
        return "empty"
    species = normalize_name(mon.get("species") or mon.get("name"))
    hp = hp_bucket(mon.get("hp_fraction"))
    status = _status(mon.get("status"))
    tera = normalize_name(mon.get("tera_type") or "")
    return f"{species}:{hp}:{status}:{tera}"


def _battle_mon_token(mon: Any | None) -> str:
    if mon is None:
        return "empty"
    species = normalize_name(getattr(mon, "species", None) or getattr(mon, "name", None))
    hp = hp_bucket(getattr(mon, "current_hp_fraction", 0.0))
    status = _status(getattr(mon, "status", None))
    tera = normalize_name(getattr(mon, "tera_type", None) or getattr(mon, "teraType", None) or "")
    return f"{species}:{hp}:{status}:{tera}"


def _alive_count_offline(side: dict[str, Any]) -> int:
    count = 0
    for mon in side.get("team") or []:
        if mon and not mon.get("fainted") and float(mon.get("hp_fraction") or 0.0) > 0.0:
            count += 1
    return count


def _alive_count_battle(pokemon: list[Any]) -> int:
    count = 0
    for mon in pokemon:
        if mon is None:
            continue
        if getattr(mon, "fainted", False):
            continue
        try:
            if float(getattr(mon, "current_hp_fraction", 0.0) or 0.0) <= 0.0:
                continue
        except (TypeError, ValueError):
            pass
        count += 1
    return count


def offline_state_key(snapshot: dict[str, Any], side: str = "p1") -> str:
    own_id = side
    opp_id = "p2" if side == "p1" else "p1"
    sides = snapshot.get("sides") or {}
    own = sides.get(own_id) or {}
    opp = sides.get(opp_id) or {}
    field = snapshot.get("field") or {}

    own_active = [_offline_mon_token(mon) for mon in (own.get("active") or [])[:2]]
    opp_active = [_offline_mon_token(mon) for mon in (opp.get("active") or [])[:2]]
    while len(own_active) < 2:
        own_active.append("empty")
    while len(opp_active) < 2:
        opp_active.append("empty")

    weather = normalize_name(field.get("weather") or "none")
    terrain = normalize_name(field.get("terrain") or "none")
    return "|".join(
        [
            f"side={side}",
            f"turn={turn_bucket(snapshot.get('turn'))}",
            f"field={weather}/{terrain}",
            f"own={','.join(own_active)}",
            f"opp={','.join(opp_active)}",
            f"alive={_alive_count_offline(own)}-{_alive_count_offline(opp)}",
        ]
    )


def battle_state_key(battle: Any, side: str = "p1") -> str:
    own_active = [_battle_mon_token(mon) for mon in list(getattr(battle, "active_pokemon", []) or [])[:2]]
    opp_active = [
        _battle_mon_token(mon)
        for mon in list(getattr(battle, "opponent_active_pokemon", []) or [])[:2]
    ]
    while len(own_active) < 2:
        own_active.append("empty")
    while len(opp_active) < 2:
        opp_active.append("empty")

    weather = normalize_name(getattr(battle, "weather", None) or "none")
    terrain = normalize_name(getattr(battle, "fields", None) or "none")
    own_team_obj = getattr(battle, "team", {}) or {}
    opp_team_obj = getattr(battle, "opponent_team", {}) or {}
    own_team = list(own_team_obj.values()) if isinstance(own_team_obj, dict) else list(own_team_obj)
    opp_team = list(opp_team_obj.values()) if isinstance(opp_team_obj, dict) else list(opp_team_obj)
    return "|".join(
        [
            f"side={side}",
            f"turn={turn_bucket(getattr(battle, 'turn', 0))}",
            f"field={weather}/{terrain}",
            f"own={','.join(own_active)}",
            f"opp={','.join(opp_active)}",
            f"alive={_alive_count_battle(own_team)}-{_alive_count_battle(opp_team)}",
        ]
    )
