"""Shared feature extraction for replay pretraining and live battle play.

The model is a candidate ranker: it scores the legal double-order candidates
for the current battle instead of predicting from a fixed action vocabulary.
That makes it usable with the fixed team in team.txt while still allowing
pretraining from replays played with other teams.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np


STATE_NUMERIC_SIZE = 96
STATE_HASH_SIZE = 256
ACTION_NUMERIC_SIZE = 40
ACTION_HASH_SIZE = 128

STATE_FEATURE_SIZE = STATE_NUMERIC_SIZE + STATE_HASH_SIZE
ACTION_FEATURE_SIZE = ACTION_NUMERIC_SIZE + ACTION_HASH_SIZE


def normalize_name(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = text.replace(" ", "-")
    text = re.sub(r"[^a-z0-9-]+", "", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _hash_index(namespace: str, value: Any, size: int) -> int:
    key = f"{namespace}:{normalize_name(value)}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little") % size


def _add_hash(vec: np.ndarray, namespace: str, value: Any, weight: float = 1.0) -> None:
    if value is None or value == "":
        return
    vec[_hash_index(namespace, value, vec.shape[0])] += float(weight)


def _finish_numeric(values: list[float], size: int) -> np.ndarray:
    if len(values) > size:
        values = values[:size]
    elif len(values) < size:
        values.extend([0.0] * (size - len(values)))
    return np.asarray(values, dtype=np.float32)


def _hp_fraction(pokemon: Any) -> float:
    if pokemon is None:
        return 0.0
    value = getattr(pokemon, "current_hp_fraction", None)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _flag(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


def _status_bucket(status: Any) -> float:
    if status is None:
        return 0.0
    name = normalize_name(getattr(status, "name", status))
    buckets = {
        "brn": 1,
        "burn": 1,
        "par": 2,
        "paralysis": 2,
        "slp": 3,
        "sleep": 3,
        "frz": 4,
        "freeze": 4,
        "psn": 5,
        "poison": 5,
        "tox": 6,
        "toxic": 6,
    }
    return buckets.get(name, 7) / 7.0


BOOST_STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
PROTECT_NAMES = {"protect", "detect", "kingsshield", "spikyshield", "banefulbunker"}
SPEED_SIDE_NAMES = {"tailwind", "trickroom"}
FIELD_SIDE_NAMES = {
    "auroraveil",
    "lightscreen",
    "reflect",
    "safeguard",
    "mist",
    "stealthrock",
    "spikes",
    "toxicspikes",
}


def _keys_or_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.keys())
    return _safe_list(value)


def _normalized_names(value: Any) -> list[str]:
    names = []
    for item in _keys_or_values(value):
        name = normalize_name(getattr(item, "name", item))
        if name:
            names.append(name)
    return names


def _boost_value(pokemon: Any, stat: str) -> float:
    if pokemon is None:
        return 0.0
    boosts = getattr(pokemon, "boosts", None) or {}
    value = 0.0
    if isinstance(boosts, dict):
        value = boosts.get(stat, 0.0)
    try:
        return max(-6.0, min(float(value), 6.0)) / 6.0
    except (TypeError, ValueError):
        return 0.0


def _has_any_name(value: Any, names: set[str]) -> float:
    found = set(_normalized_names(value))
    return 1.0 if found.intersection(names) else 0.0


def _side_conditions(battle: Any, attr: str) -> list[str]:
    return _normalized_names(getattr(battle, attr, None))


def _side_condition_count(battle: Any, attr: str) -> float:
    return min(len(_side_conditions(battle, attr)), 8) / 8.0


def _pokemon_species(pokemon: Any) -> str:
    return normalize_name(
        getattr(pokemon, "species", None)
        or getattr(pokemon, "base_species", None)
        or getattr(pokemon, "name", None)
    )


def _pokemon_types(pokemon: Any) -> list[str]:
    types = []
    for type_obj in _safe_list(getattr(pokemon, "types", [])):
        type_name = getattr(type_obj, "name", type_obj)
        if type_name:
            types.append(normalize_name(type_name))
    return types


def _pokemon_moves(pokemon: Any) -> list[Any]:
    moves = getattr(pokemon, "moves", None)
    if isinstance(moves, dict):
        return list(moves.values())
    return _safe_list(moves)


def _move_id(move: Any) -> str:
    return normalize_name(getattr(move, "id", None) or getattr(move, "name", None) or move)


def _move_type(move: Any) -> str:
    type_obj = getattr(move, "type", None)
    return normalize_name(getattr(type_obj, "name", type_obj))


def _move_category(move: Any) -> str:
    cat = getattr(move, "category", None)
    return normalize_name(getattr(cat, "name", cat))


def _move_power(move: Any) -> float:
    value = getattr(move, "base_power", None)
    if value is None:
        value = getattr(move, "basePower", None)
    if value is None:
        value = getattr(move, "power", None)
    try:
        return max(0.0, min(float(value or 0.0), 250.0)) / 250.0
    except (TypeError, ValueError):
        return 0.0


def _move_accuracy(move: Any) -> float:
    value = getattr(move, "accuracy", None)
    if value is True or value is None:
        return 1.0
    try:
        return max(0.0, min(float(value), 100.0)) / 100.0
    except (TypeError, ValueError):
        return 1.0


def _move_priority(move: Any) -> float:
    try:
        return max(-7.0, min(float(getattr(move, "priority", 0.0)), 7.0)) / 7.0
    except (TypeError, ValueError):
        return 0.0


def battle_state_features(battle: Any) -> np.ndarray:
    """Build fixed-size state features from a poke-env battle object."""
    numeric: list[float] = []
    hashed = np.zeros(STATE_HASH_SIZE, dtype=np.float32)

    numeric.extend(
        [
            min(float(getattr(battle, "turn", 0) or 0), 50.0) / 50.0,
            _flag(getattr(battle, "finished", False)),
            _flag(getattr(battle, "won", False)),
            _flag(getattr(battle, "lost", False)),
            _flag(getattr(battle, "force_switch", False)),
        ]
    )

    own_team = list(getattr(battle, "team", {}).values())
    opp_team = list(getattr(battle, "opponent_team", {}).values())
    own_fainted = sum(1 for p in own_team if getattr(p, "fainted", False))
    opp_fainted = sum(1 for p in opp_team if getattr(p, "fainted", False))
    own_hp = [_hp_fraction(p) for p in own_team if not getattr(p, "fainted", False)]
    opp_hp = [_hp_fraction(p) for p in opp_team if not getattr(p, "fainted", False)]
    numeric.extend(
        [
            own_fainted / 6.0,
            opp_fainted / 6.0,
            float(np.mean(own_hp)) if own_hp else 0.0,
            float(np.mean(opp_hp)) if opp_hp else 0.0,
        ]
    )

    active_groups = [
        ("own-active", _safe_list(getattr(battle, "active_pokemon", []))),
        ("opp-active", _safe_list(getattr(battle, "opponent_active_pokemon", []))),
    ]
    for namespace, active in active_groups:
        for slot in range(2):
            pokemon = active[slot] if slot < len(active) else None
            numeric.extend(
                [
                    _hp_fraction(pokemon),
                    _flag(getattr(pokemon, "fainted", False)) if pokemon else 0.0,
                    _status_bucket(getattr(pokemon, "status", None)) if pokemon else 0.0,
                ]
            )
            if pokemon is not None:
                numeric.extend(_boost_value(pokemon, stat) for stat in BOOST_STATS)
                numeric.extend(
                    [
                        _has_any_name(getattr(pokemon, "effects", None), PROTECT_NAMES),
                        _has_any_name(getattr(pokemon, "volatiles", None), PROTECT_NAMES),
                    ]
                )
                species = _pokemon_species(pokemon)
                _add_hash(hashed, f"{namespace}-species", species, 1.0)
                for type_name in _pokemon_types(pokemon):
                    _add_hash(hashed, f"{namespace}-type", type_name, 0.5)
                for move in _pokemon_moves(pokemon):
                    _add_hash(hashed, f"{namespace}-move", _move_id(move), 0.25)
                for effect in _normalized_names(getattr(pokemon, "effects", None)):
                    _add_hash(hashed, f"{namespace}-effect", effect, 0.25)
                for volatile in _normalized_names(getattr(pokemon, "volatiles", None)):
                    _add_hash(hashed, f"{namespace}-volatile", volatile, 0.25)
            else:
                numeric.extend([0.0] * (len(BOOST_STATS) + 2))

    for namespace, team in [("own-team", own_team), ("opp-team", opp_team)]:
        for pokemon in team:
            _add_hash(hashed, f"{namespace}-species", _pokemon_species(pokemon), 0.5)
            for type_name in _pokemon_types(pokemon):
                _add_hash(hashed, f"{namespace}-type", type_name, 0.25)

    for side, moves_by_slot in [("own-legal", getattr(battle, "available_moves", []))]:
        for slot, moves in enumerate(_safe_list(moves_by_slot)):
            numeric.append(min(len(_safe_list(moves)), 4) / 4.0)
            for move in _safe_list(moves):
                _add_hash(hashed, f"{side}-{slot}", _move_id(move), 0.25)

    weather_names = _normalized_names(getattr(battle, "weather", []))
    field_names = _normalized_names(getattr(battle, "fields", []))
    for weather in weather_names:
        _add_hash(hashed, "weather", weather, 1.0)
    for field in field_names:
        _add_hash(hashed, "field", field, 1.0)

    own_side = _side_conditions(battle, "side_conditions")
    opp_side = _side_conditions(battle, "opponent_side_conditions")
    for condition in own_side:
        _add_hash(hashed, "own-side-condition", condition, 0.75)
    for condition in opp_side:
        _add_hash(hashed, "opp-side-condition", condition, 0.75)
    numeric.extend(
        [
            _side_condition_count(battle, "side_conditions"),
            _side_condition_count(battle, "opponent_side_conditions"),
            1.0 if set(own_side).intersection(SPEED_SIDE_NAMES) else 0.0,
            1.0 if set(opp_side).intersection(SPEED_SIDE_NAMES) else 0.0,
            1.0 if set(own_side).intersection(FIELD_SIDE_NAMES) else 0.0,
            1.0 if set(opp_side).intersection(FIELD_SIDE_NAMES) else 0.0,
            min(len(weather_names), 4) / 4.0,
            min(len(field_names), 4) / 4.0,
            1.0 if "trickroom" in field_names else 0.0,
            _flag(getattr(battle, "can_tera", False)),
            _flag(getattr(battle, "opponent_can_tera", False)),
        ]
    )

    numeric_vec = _finish_numeric(numeric, STATE_NUMERIC_SIZE)
    hashed = np.clip(hashed, -4.0, 4.0) / 4.0
    return np.concatenate([numeric_vec, hashed]).astype(np.float32)


def _single_order_features(order: Any, slot: int) -> tuple[list[float], list[tuple[str, str, float]]]:
    class_name = order.__class__.__name__.lower() if order is not None else "none"
    raw_order = getattr(order, "order", None)
    numeric = [
        1.0 if "pass" in class_name else 0.0,
        1.0 if "forfeit" in class_name else 0.0,
        1.0 if raw_order is not None and raw_order.__class__.__name__.lower() == "pokemon" else 0.0,
        1.0 if raw_order is not None and raw_order.__class__.__name__.lower() == "move" else 0.0,
        (float(getattr(order, "move_target", 0) or 0) + 2.0) / 4.0,
        _flag(getattr(order, "mega", False)),
        _flag(getattr(order, "z_move", False)),
        _flag(getattr(order, "dynamax", False)),
        _flag(getattr(order, "terastallize", False)),
    ]
    hashes: list[tuple[str, str, float]] = []

    if raw_order is not None:
        raw_class = raw_order.__class__.__name__.lower()
        if raw_class == "move":
            numeric.extend([_move_power(raw_order), _move_accuracy(raw_order), _move_priority(raw_order)])
            hashes.append((f"slot{slot}-move", _move_id(raw_order), 1.0))
            hashes.append((f"slot{slot}-move-type", _move_type(raw_order), 0.5))
            hashes.append((f"slot{slot}-move-cat", _move_category(raw_order), 0.5))
        else:
            species = _pokemon_species(raw_order)
            hashes.append((f"slot{slot}-switch", species, 1.0))
            for type_name in _pokemon_types(raw_order):
                hashes.append((f"slot{slot}-switch-type", type_name, 0.5))
    return numeric, hashes


def order_action_features(order: Any) -> np.ndarray:
    """Build fixed-size features from a poke-env BattleOrder candidate."""
    numeric: list[float] = []
    hashed = np.zeros(ACTION_HASH_SIZE, dtype=np.float32)
    parts = [getattr(order, "first_order", order), getattr(order, "second_order", None)]
    for slot, single_order in enumerate(parts[:2]):
        nums, hashes = _single_order_features(single_order, slot)
        numeric.extend(nums)
        for namespace, value, weight in hashes:
            _add_hash(hashed, namespace, value, weight)

    message = getattr(order, "message", None) or str(order)
    _add_hash(hashed, "order-message", message, 0.25)
    numeric_vec = _finish_numeric(numeric, ACTION_NUMERIC_SIZE)
    hashed = np.clip(hashed, -4.0, 4.0) / 4.0
    return np.concatenate([numeric_vec, hashed]).astype(np.float32)


def replay_state_features(sample: dict[str, Any]) -> np.ndarray:
    """Build state features from a *_double_decisions.jsonl row."""
    numeric: list[float] = [
        min(float(sample.get("turn") or 0), 50.0) / 50.0,
        0.0,
        1.0 if sample.get("outcome") == "win" else 0.0,
        1.0 if sample.get("outcome") == "loss" else 0.0,
        0.0,
    ]
    hashed = np.zeros(STATE_HASH_SIZE, dtype=np.float32)

    decision_type = normalize_name(sample.get("decision_type"))
    _add_hash(hashed, "decision-type", decision_type, 1.0)

    preview = sample.get("team_preview") or {}
    for namespace, names in [("own-team", preview.get("own", [])), ("opp-team", preview.get("opp", []))]:
        for name in names or []:
            _add_hash(hashed, namespace, name, 0.75)

    revealed = sample.get("revealed_moves") or {}
    own_side = sample.get("player_side")
    opp_side = "p2" if own_side == "p1" else "p1"
    for side_label, side in [("own", own_side), ("opp", opp_side)]:
        for pokemon, moves in (revealed.get(side, {}) or {}).items():
            _add_hash(hashed, f"{side_label}-revealed-pokemon", pokemon, 0.5)
            for move in moves or []:
                _add_hash(hashed, f"{side_label}-revealed-move", move, 0.35)

    actions = sample.get("actions") or []
    numeric.extend([min(len(actions), 2) / 2.0])
    numeric_vec = _finish_numeric(numeric, STATE_NUMERIC_SIZE)
    hashed = np.clip(hashed, -4.0, 4.0) / 4.0
    return np.concatenate([numeric_vec, hashed]).astype(np.float32)


def replay_action_features(sample: dict[str, Any]) -> np.ndarray:
    """Build action features from a replay double-decision row."""
    numeric: list[float] = []
    hashed = np.zeros(ACTION_HASH_SIZE, dtype=np.float32)
    for slot, action in enumerate((sample.get("actions") or [])[:2]):
        action_type = normalize_name(action.get("type"))
        target_slot = action.get("target_slot") or action.get("target") or 0
        try:
            target_norm = (float(target_slot) + 2.0) / 4.0
        except (TypeError, ValueError):
            target_norm = 0.5
        numeric.extend(
            [
                0.0,
                0.0,
                1.0 if action_type == "switch" else 0.0,
                1.0 if action_type == "move" else 0.0,
                target_norm,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        if action_type == "move":
            _add_hash(hashed, f"slot{slot}-move", action.get("move"), 1.0)
            _add_hash(hashed, f"slot{slot}-target", action.get("target_slot") or action.get("target"), 0.25)
        elif action_type == "switch":
            _add_hash(hashed, f"slot{slot}-switch", action.get("details") or action.get("pokemon"), 1.0)

    _add_hash(hashed, "order-signature", sample.get("order_signature"), 0.25)
    numeric_vec = _finish_numeric(numeric, ACTION_NUMERIC_SIZE)
    hashed = np.clip(hashed, -4.0, 4.0) / 4.0
    return np.concatenate([numeric_vec, hashed]).astype(np.float32)


def _sim_hp_fraction(pokemon: dict[str, Any] | None) -> float:
    if not pokemon:
        return 0.0
    value = pokemon.get("hp_fraction")
    if value is None:
        hp = pokemon.get("hp", 0.0)
        maxhp = pokemon.get("maxhp", 0.0)
        try:
            return max(0.0, min(float(hp) / max(1.0, float(maxhp)), 1.0))
        except (TypeError, ValueError):
            return 0.0
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _sim_boost_value(pokemon: dict[str, Any] | None, stat: str) -> float:
    if not pokemon:
        return 0.0
    boosts = pokemon.get("boosts") or {}
    try:
        return max(-6.0, min(float(boosts.get(stat, 0.0)), 6.0)) / 6.0
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _sim_move_id(move: Any) -> str:
    if isinstance(move, dict):
        return normalize_name(move.get("id") or move.get("name"))
    return normalize_name(move)


def _sim_choice_parts(message: Any) -> list[str]:
    text = str(message or "").strip()
    if text.startswith("/choose "):
        text = text[len("/choose ") :]
    if text.startswith("/team "):
        text = "team " + text[len("/team ") :]
    if text == "/forfeit":
        text = "forfeit"
    return [part.strip() for part in text.split(",") if part.strip()]


def simulator_state_features(snapshot: dict[str, Any], side: str = "p1") -> np.ndarray:
    """Build state features from tools/showdown_sim_server.js offline state."""
    side = "p2" if side == "p2" else "p1"
    opp_side = "p1" if side == "p2" else "p2"
    own = (snapshot.get("sides") or {}).get(side) or {}
    opp = (snapshot.get("sides") or {}).get(opp_side) or {}
    field = snapshot.get("field") or {}
    legal = snapshot.get("legal") or {}
    winner_side = snapshot.get("winner_side") or ""

    numeric: list[float] = [
        min(float(snapshot.get("turn") or 0), 50.0) / 50.0,
        _flag(snapshot.get("ended")),
        1.0 if winner_side == side else 0.0,
        1.0 if winner_side == opp_side else 0.0,
        1.0 if snapshot.get("request_state") == "switch" else 0.0,
    ]
    hashed = np.zeros(STATE_HASH_SIZE, dtype=np.float32)

    own_team = list(own.get("team") or [])
    opp_team = list(opp.get("team") or [])
    own_fainted = sum(1 for pokemon in own_team if pokemon.get("fainted"))
    opp_fainted = sum(1 for pokemon in opp_team if pokemon.get("fainted"))
    own_hp = [_sim_hp_fraction(pokemon) for pokemon in own_team if not pokemon.get("fainted")]
    opp_hp = [_sim_hp_fraction(pokemon) for pokemon in opp_team if not pokemon.get("fainted")]
    numeric.extend(
        [
            own_fainted / 6.0,
            opp_fainted / 6.0,
            float(np.mean(own_hp)) if own_hp else 0.0,
            float(np.mean(opp_hp)) if opp_hp else 0.0,
        ]
    )

    active_groups = [
        ("own-active", list(own.get("active") or [])),
        ("opp-active", list(opp.get("active") or [])),
    ]
    for namespace, active in active_groups:
        for slot in range(2):
            pokemon = active[slot] if slot < len(active) else None
            numeric.extend(
                [
                    _sim_hp_fraction(pokemon),
                    _flag(pokemon.get("fainted")) if pokemon else 0.0,
                    _status_bucket(pokemon.get("status")) if pokemon else 0.0,
                ]
            )
            if pokemon:
                numeric.extend(_sim_boost_value(pokemon, stat) for stat in BOOST_STATS)
                volatiles = [normalize_name(item) for item in pokemon.get("volatiles") or []]
                numeric.extend(
                    [
                        1.0 if set(volatiles).intersection(PROTECT_NAMES) else 0.0,
                        0.0,
                    ]
                )
                species = normalize_name(pokemon.get("species") or pokemon.get("name"))
                _add_hash(hashed, f"{namespace}-species", species, 1.0)
                for type_name in pokemon.get("types") or []:
                    _add_hash(hashed, f"{namespace}-type", type_name, 0.5)
                for move in pokemon.get("moves") or []:
                    _add_hash(hashed, f"{namespace}-move", _sim_move_id(move), 0.25)
                for volatile in volatiles:
                    _add_hash(hashed, f"{namespace}-volatile", volatile, 0.25)
            else:
                numeric.extend([0.0] * (len(BOOST_STATS) + 2))

    for namespace, team in [("own-team", own_team), ("opp-team", opp_team)]:
        for pokemon in team:
            _add_hash(hashed, f"{namespace}-species", pokemon.get("species"), 0.5)
            for type_name in pokemon.get("types") or []:
                _add_hash(hashed, f"{namespace}-type", type_name, 0.25)

    own_legal = list(legal.get(side) or [])
    numeric.append(min(len(own_legal), 32) / 32.0)
    for choice in own_legal[:32]:
        for part in _sim_choice_parts(choice):
            tokens = part.split()
            if len(tokens) >= 2 and tokens[0] == "move":
                _add_hash(hashed, "own-legal-move", tokens[1], 0.15)

    weather = normalize_name(field.get("weather"))
    terrain = normalize_name(field.get("terrain"))
    pseudo = [normalize_name(item) for item in field.get("pseudo_weather") or []]
    _add_hash(hashed, "weather", weather, 1.0)
    _add_hash(hashed, "field", terrain, 1.0)
    for item in pseudo:
        _add_hash(hashed, "field", item, 1.0)

    own_side = [normalize_name(item) for item in own.get("side_conditions") or []]
    opp_conditions = [normalize_name(item) for item in opp.get("side_conditions") or []]
    for condition in own_side:
        _add_hash(hashed, "own-side-condition", condition, 0.75)
    for condition in opp_conditions:
        _add_hash(hashed, "opp-side-condition", condition, 0.75)
    field_names = [terrain, *pseudo]
    numeric.extend(
        [
            min(len(own_side), 8) / 8.0,
            min(len(opp_conditions), 8) / 8.0,
            1.0 if set(own_side).intersection(SPEED_SIDE_NAMES) else 0.0,
            1.0 if set(opp_conditions).intersection(SPEED_SIDE_NAMES) else 0.0,
            1.0 if set(own_side).intersection(FIELD_SIDE_NAMES) else 0.0,
            1.0 if set(opp_conditions).intersection(FIELD_SIDE_NAMES) else 0.0,
            1.0 if weather else 0.0,
            min(len(field_names), 4) / 4.0,
            1.0 if "trickroom" in field_names else 0.0,
            _flag(own.get("can_tera")),
            _flag(opp.get("can_tera")),
        ]
    )

    numeric_vec = _finish_numeric(numeric, STATE_NUMERIC_SIZE)
    hashed = np.clip(hashed, -4.0, 4.0) / 4.0
    return np.concatenate([numeric_vec, hashed]).astype(np.float32)


def simulator_action_features(message: Any) -> np.ndarray:
    """Build action features from a Showdown choice string."""
    numeric: list[float] = []
    hashed = np.zeros(ACTION_HASH_SIZE, dtype=np.float32)
    text = str(message or "").strip()
    parts = _sim_choice_parts(text)
    if text.startswith("/team ") or text.startswith("team "):
        parts = [text.replace("/team ", "team ", 1)]
    for slot in range(2):
        part = parts[slot] if slot < len(parts) else "pass"
        tokens = part.split()
        kind = tokens[0].lower() if tokens else "pass"
        target = 0.0
        for token in reversed(tokens):
            try:
                target = float(token)
                break
            except ValueError:
                continue
        numeric.extend(
            [
                1.0 if kind == "pass" else 0.0,
                1.0 if kind == "forfeit" else 0.0,
                1.0 if kind in {"switch", "team"} else 0.0,
                1.0 if kind == "move" else 0.0,
                (target + 2.0) / 4.0,
                0.0,
                0.0,
                0.0,
                1.0 if "terastallize" in part.lower() else 0.0,
            ]
        )
        if kind == "move" and len(tokens) >= 2:
            _add_hash(hashed, f"slot{slot}-move", tokens[1], 1.0)
        elif kind in {"switch", "team"} and len(tokens) >= 2:
            _add_hash(hashed, f"slot{slot}-switch", tokens[1], 1.0)

    _add_hash(hashed, "order-message", text, 0.25)
    numeric_vec = _finish_numeric(numeric, ACTION_NUMERIC_SIZE)
    hashed = np.clip(hashed, -4.0, 4.0) / 4.0
    return np.concatenate([numeric_vec, hashed]).astype(np.float32)


def outcome_to_value(outcome: Any) -> float:
    if outcome == "win":
        return 1.0
    if outcome == "loss":
        return -1.0
    return 0.0
