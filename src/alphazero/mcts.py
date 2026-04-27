"""MCTS over legal VGC double-order candidates.

This search is intentionally candidate-based. The expensive part for VGC is
generating and scoring legal double orders; the model supplies priors and a
state value, while the tree statistics decide how much each candidate is
explored. When a Showdown simulator client is configured, depth >= 2 evaluates
branches in Pokemon Showdown's real JS engine. In live local battles this can
use the server's internal serialized state directly; without that bridge it can
fall back to replaying an input log, and finally to the older tactical estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch

from src.alphazero.features import battle_state_features, order_action_features
from src.alphazero.network import AlphaZeroPolicyValueNet


@dataclass
class MCTSConfig:
    simulations: int = 64
    depth: int = 1
    cpuct: float = 1.5
    temperature: float = 0.0
    heuristic_weight: float = 0.75
    depth2_weight: float = 0.65
    device: str = "cpu"
    showdown_simulator: Any | None = None
    simulation_tracker: Any | None = None
    require_showdown_simulator: bool = False


@dataclass
class SearchResult:
    order: Any
    selected_index: int
    state_features: np.ndarray
    action_features: np.ndarray
    priors: np.ndarray
    visit_probs: np.ndarray
    value: float
    logprob: float
    candidate_values: np.ndarray
    simulator_used: bool = False
    simulator_repairs: int = 0
    simulator_errors: int = 0
    simulator_skipped_branches: int = 0
    simulator_error_details: list[dict[str, Any]] | None = None
    simulator_error_stage_counts: dict[str, int] | None = None


@dataclass
class _Child:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    total = exp.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full_like(logits, 1.0 / len(logits), dtype=np.float64)
    return exp / total


def _single_order_kind(order: Any) -> str:
    if order is None:
        return "none"
    class_name = order.__class__.__name__.lower()
    raw_order = getattr(order, "order", None)
    if "forfeit" in class_name:
        return "forfeit"
    if "pass" in class_name:
        return "pass"
    if raw_order is None:
        return "default"
    raw_class = raw_order.__class__.__name__.lower()
    if raw_class == "move":
        return "move"
    if raw_class == "pokemon":
        return "switch"
    return raw_class


def _move_power(move: Any) -> float:
    for attr in ("base_power", "basePower", "power"):
        value = getattr(move, attr, None)
        if value is not None:
            try:
                return max(0.0, min(float(value), 250.0)) / 250.0
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _move_accuracy(move: Any) -> float:
    value = getattr(move, "accuracy", None)
    if value is True or value is None:
        return 1.0
    try:
        return max(0.0, min(float(value), 100.0)) / 100.0
    except (TypeError, ValueError):
        return 1.0


def _hp_fraction(pokemon: Any) -> float:
    if pokemon is None:
        return 0.0
    try:
        return max(0.0, min(float(getattr(pokemon, "current_hp_fraction", 0.0) or 0.0), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _move_name(move: Any) -> str:
    value = getattr(move, "id", None) or getattr(move, "name", None) or ""
    return str(value).lower().replace(" ", "").replace("-", "")


def _move_type(move: Any) -> str:
    type_obj = getattr(move, "type", None)
    value = getattr(type_obj, "name", type_obj) or ""
    return str(value).lower()


def _move_target(move: Any) -> str:
    target = getattr(move, "target", None)
    value = getattr(target, "name", target) or ""
    return str(value).lower().replace("_", "-")


def _move_id(move: Any) -> str:
    return str(getattr(move, "id", None) or getattr(move, "name", None) or "").lower()


def _pokemon_types(pokemon: Any) -> set[str]:
    types = set()
    for type_obj in getattr(pokemon, "types", []) or []:
        value = getattr(type_obj, "name", type_obj)
        if value:
            types.add(str(value).lower())
    return types


@lru_cache(maxsize=1)
def _type_chart() -> dict[str, dict[str, float]]:
    try:
        from src.utils import load_type_chart

        return load_type_chart()
    except Exception:
        return {}


def _type_effectiveness(move_type: str, defender: Any) -> float:
    if not move_type or defender is None:
        return 1.0
    mult = 1.0
    chart = _type_chart().get(move_type, {})
    for defender_type in _pokemon_types(defender):
        mult *= float(chart.get(defender_type, 1.0))
    return max(0.0, min(mult, 4.0))


def _active_can_act(pokemon: Any) -> bool:
    if pokemon is None:
        return False
    if getattr(pokemon, "fainted", False):
        return False
    hp = getattr(pokemon, "current_hp_fraction", None)
    return hp is None or float(hp or 0.0) > 0.0


def _best_available_move_ids(battle: Any, slot: int) -> set[str]:
    available = getattr(battle, "available_moves", []) or []
    if slot >= len(available):
        return set()
    moves = list(available[slot] or [])
    damaging = [move for move in moves if _move_power(move) > 0]
    if not damaging:
        return set()

    def adjusted_power(move: Any) -> float:
        bonus = 1.15 if "all" in _move_target(move) else 1.0
        return _move_power(move) * _move_accuracy(move) * bonus

    best = max(adjusted_power(move) for move in damaging)
    return {_move_id(move) for move in damaging if adjusted_power(move) >= best - 1e-6}


def _status_move_score(move: Any, *, battle: Any, hp: float) -> float:
    name = _move_name(move)
    turn = int(getattr(battle, "turn", 0) or 0)

    if name in {"protect", "detect"}:
        if hp < 0.30:
            return 0.08
        return -0.18
    if name in {"recover", "slackoff", "roost", "moonlight", "synthesis"}:
        if hp < 0.45:
            return 0.18
        return -0.12
    if name == "tailwind":
        return 0.18 if turn <= 2 else -0.10
    if name in {"dragondance", "swordsdance", "nastyplot", "calmmind"}:
        return 0.14 if hp > 0.55 else -0.10
    if name in {"fakeout"}:
        return 0.18 if turn <= 2 else 0.04
    if name in {"partingshot"}:
        return -0.02 if hp < 0.50 else -0.12
    if name in {"charm", "willowisp", "thunderwave", "helpinghand"}:
        return -0.04
    return -0.08


def _move_score(move: Any, *, battle: Any, pokemon: Any, hp: float, terastallize: bool) -> float:
    power = _move_power(move)
    accuracy = _move_accuracy(move)
    if power <= 0:
        score = _status_move_score(move, battle=battle, hp=hp)
        if terastallize:
            score -= 0.10
        return score

    score = 0.20 + 1.35 * power * accuracy
    move_type = _move_type(move)
    if pokemon is not None and move_type in _pokemon_types(pokemon):
        score += 0.15
    target = _move_target(move)
    if "all" in target or "foe" in target or "opponent" in target:
        score += 0.08
    priority = getattr(move, "priority", 0) or 0
    try:
        if float(priority) > 0:
            score += 0.05
    except (TypeError, ValueError):
        pass
    if hp < 0.25:
        score += 0.04
    if terastallize:
        score += 0.08
    return score


def _target_slots(move: Any, move_target: int, defenders: list[Any]) -> list[int]:
    target = _move_target(move)
    alive_slots = [i for i, pokemon in enumerate(defenders[:2]) if _active_can_act(pokemon)]
    if not alive_slots:
        return []
    if "all" in target or "foe" in target or "opponent" in target:
        return alive_slots
    if move_target in {1, 2}:
        slot = move_target - 1
        return [slot] if slot in alive_slots else alive_slots[:1]
    return alive_slots[:1]


def _estimated_damage_fraction(move: Any, attacker: Any, defender: Any, *, spread: bool) -> float:
    power = _move_power(move)
    if power <= 0 or defender is None:
        return 0.0
    accuracy = _move_accuracy(move)
    move_type = _move_type(move)
    stab = 1.18 if attacker is not None and move_type in _pokemon_types(attacker) else 1.0
    effectiveness = _type_effectiveness(move_type, defender)
    spread_factor = 0.75 if spread else 1.0
    priority_bonus = 1.04 if float(getattr(move, "priority", 0) or 0) > 0 else 1.0
    estimate = 0.08 + 0.72 * power * accuracy * stab * np.sqrt(effectiveness) * spread_factor
    return float(max(0.0, min(estimate * priority_bonus, 1.25)))


def _apply_order_pressure(
    *,
    order: Any,
    attackers: list[Any],
    defenders: list[Any],
    defender_hp: np.ndarray,
) -> tuple[float, int]:
    """Project visible damage from one double order into defender HP fractions."""
    damage_total = 0.0
    ko_count = 0
    parts = [getattr(order, "first_order", order), getattr(order, "second_order", None)]
    for slot, single_order in enumerate(parts[:2]):
        if slot >= len(attackers) or not _active_can_act(attackers[slot]):
            continue
        if _single_order_kind(single_order) != "move":
            continue
        move = getattr(single_order, "order", None)
        target_slots = _target_slots(
            move,
            int(getattr(single_order, "move_target", 0) or 0),
            defenders,
        )
        spread = len(target_slots) > 1
        for target_slot in target_slots:
            if target_slot >= len(defenders) or not _active_can_act(defenders[target_slot]):
                continue
            before = float(defender_hp[target_slot])
            damage = min(
                before,
                _estimated_damage_fraction(
                    move,
                    attackers[slot],
                    defenders[target_slot],
                    spread=spread,
                ),
            )
            defender_hp[target_slot] = max(0.0, before - damage)
            damage_total += damage
            if before > 0.0 and defender_hp[target_slot] <= 0.02:
                ko_count += 1
    return damage_total, ko_count


def _known_moves(pokemon: Any) -> list[Any]:
    moves = getattr(pokemon, "moves", None)
    if isinstance(moves, dict):
        return list(moves.values())
    if moves is None:
        return []
    try:
        return list(moves)
    except TypeError:
        return []


def _best_response_threat(
    *,
    attackers: list[Any],
    attacker_hp: np.ndarray,
    defenders: list[Any],
    defender_hp: np.ndarray,
    battle: Any,
) -> tuple[float, int]:
    """Estimate the opponent's best public-information response after our action."""
    total_threat = 0.0
    ko_count = 0
    projected_hp = defender_hp.copy()
    for slot, attacker in enumerate(attackers[:2]):
        if not _active_can_act(attacker):
            continue
        if slot < len(attacker_hp) and attacker_hp[slot] <= 0.02:
            continue
        moves = [move for move in _known_moves(attacker) if _move_power(move) > 0]
        if not moves:
            total_threat += 0.16
            continue

        best_damage = 0.0
        best_target = None
        for move in moves:
            alive_targets = [
                i
                for i, defender in enumerate(defenders[:2])
                if _active_can_act(defender) and projected_hp[i] > 0.02
            ]
            if not alive_targets:
                break
            target_name = _move_target(move)
            spread = "all" in target_name or "foe" in target_name or "opponent" in target_name
            for target_slot in alive_targets:
                damage = _estimated_damage_fraction(
                    move,
                    attacker,
                    defenders[target_slot],
                    spread=spread,
                )
                if damage > best_damage:
                    best_damage = damage
                    best_target = target_slot
        if best_target is None:
            continue
        before = float(projected_hp[best_target])
        applied = min(before, best_damage)
        projected_hp[best_target] = max(0.0, before - applied)
        total_threat += applied
        if before > 0.0 and projected_hp[best_target] <= 0.02:
            ko_count += 1

    return total_threat, ko_count


def depth_two_candidate_value(battle: Any, order: Any) -> float:
    """One-turn minimax approximation: our order, then opponent response."""
    own_active = list(getattr(battle, "active_pokemon", []) or [])[:2]
    opp_active = list(getattr(battle, "opponent_active_pokemon", []) or [])[:2]
    while len(own_active) < 2:
        own_active.append(None)
    while len(opp_active) < 2:
        opp_active.append(None)

    own_hp = np.asarray([_hp_fraction(pokemon) for pokemon in own_active], dtype=np.float32)
    opp_hp = np.asarray([_hp_fraction(pokemon) for pokemon in opp_active], dtype=np.float32)
    opp_hp_after = opp_hp.copy()

    damage_dealt, opp_kos = _apply_order_pressure(
        order=order,
        attackers=own_active,
        defenders=opp_active,
        defender_hp=opp_hp_after,
    )
    response_damage, own_kos = _best_response_threat(
        attackers=opp_active,
        attacker_hp=opp_hp_after,
        defenders=own_active,
        defender_hp=own_hp,
        battle=battle,
    )

    tempo = 0.20 * opp_kos - 0.25 * own_kos
    hp_trade = damage_dealt - 0.85 * response_damage
    root = heuristic_candidate_value(battle, order)
    return float(np.tanh(0.70 * root + hp_trade + tempo))


def heuristic_candidate_value(battle: Any, order: Any) -> float:
    """Cheap tactical prior used before the neural net has enough training."""
    score = 0.0
    acting_slots = 0
    active = list(getattr(battle, "active_pokemon", []) or [])
    parts = [getattr(order, "first_order", order), getattr(order, "second_order", None)]
    for slot, single_order in enumerate(parts[:2]):
        kind = _single_order_kind(single_order)
        pokemon = active[slot] if slot < len(active) else None
        hp = float(getattr(pokemon, "current_hp_fraction", 1.0) or 1.0)
        can_act = _active_can_act(pokemon)
        if can_act:
            acting_slots += 1
        if kind == "move":
            move = getattr(single_order, "order", None)
            score += _move_score(
                move,
                battle=battle,
                pokemon=pokemon,
                hp=hp,
                terastallize=bool(getattr(single_order, "terastallize", False)),
            )
            if _move_id(move) in _best_available_move_ids(battle, slot):
                score += 0.45
        elif kind == "switch":
            if not can_act:
                score += 0.02
            elif hp < 0.25:
                score += 0.10
            elif hp < 0.45:
                score -= 0.02
            else:
                score -= 0.22
        elif kind == "pass":
            score += 0.0 if not can_act else -0.45
        elif kind == "forfeit":
            score -= 1.0
    if acting_slots > 0:
        score /= acting_slots
    return float(np.tanh(score))


class AlphaZeroMCTS:
    def __init__(self, model: AlphaZeroPolicyValueNet, config: MCTSConfig):
        self.model = model
        self.config = config
        self.rng = np.random.default_rng()
        self.model.to(config.device)
        self.model.eval()

    def _model_eval(
        self,
        state_features: np.ndarray,
        action_features: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        state = torch.as_tensor(state_features, dtype=torch.float32, device=self.config.device)
        actions = torch.as_tensor(action_features, dtype=torch.float32, device=self.config.device)
        with torch.no_grad():
            logits, values = self.model(state.unsqueeze(0), actions.unsqueeze(0))
        priors = _softmax(logits.squeeze(0).detach().cpu().numpy())
        value = float(values.squeeze(0).detach().cpu().item())
        return priors.astype(np.float32), value

    def search(self, battle: Any, candidates: list[Any]) -> SearchResult:
        if not candidates:
            raise ValueError("AlphaZeroMCTS.search requires at least one candidate")

        state_features = battle_state_features(battle)
        action_features = np.stack([order_action_features(order) for order in candidates]).astype(
            np.float32
        )
        priors, state_value = self._model_eval(state_features, action_features)
        heuristic_values = np.asarray(
            [heuristic_candidate_value(battle, order) for order in candidates],
            dtype=np.float32,
        )
        priors = _softmax(
            np.log(np.maximum(priors, 1e-8))
            + max(1.0, self.config.heuristic_weight * 3.0) * heuristic_values
        ).astype(np.float32)
        children = [_Child(float(prior)) for prior in priors]
        simulated_values = None
        simulator_repairs = 0
        simulator_errors = 0
        simulator_skipped_branches = 0
        simulator_error_details = []
        simulator_error_stage_counts = {}
        simulator_attempted = self.config.depth >= 2 and self.config.showdown_simulator is not None
        if simulator_attempted:
            # In simultaneous VGC, depth=2 means our choice plus the opponent's
            # simultaneous response for the current turn. The JS simulator's
            # recursive depth counts full future turns after that root choice,
            # so subtract one to avoid an unintended branch explosion.
            showdown_depth = max(1, int(self.config.depth) - 1)
            simulated_values = self.config.showdown_simulator.evaluate_candidates(
                tracker=self.config.simulation_tracker,
                battle=battle,
                candidates=candidates,
                depth=showdown_depth,
            )
            simulator_repairs = int(
                getattr(self.config.showdown_simulator, "last_repairs", 0) or 0
            )
            simulator_errors = int(
                getattr(self.config.showdown_simulator, "last_simulation_errors", 0) or 0
            )
            simulator_skipped_branches = int(
                getattr(self.config.showdown_simulator, "last_skipped_branches", 0) or 0
            )
            simulator_error_details = list(
                getattr(self.config.showdown_simulator, "last_error_details", []) or []
            )
            simulator_error_stage_counts = dict(
                getattr(self.config.showdown_simulator, "last_error_stage_counts", {}) or {}
            )

        search_values = heuristic_values
        using_real_simulator = False
        if simulated_values is not None:
            search_values = np.asarray(simulated_values, dtype=np.float32)
            using_real_simulator = True
        elif self.config.depth >= 2:
            if self.config.require_showdown_simulator:
                detail = getattr(self.config.showdown_simulator, "last_error", "") or "unknown error"
                raise RuntimeError(
                    "Showdown simulator required but no real simulation was returned: "
                    f"{detail}"
                )
            search_values = np.asarray(
                [depth_two_candidate_value(battle, order) for order in candidates],
                dtype=np.float32,
            )

        if using_real_simulator:
            candidate_values = np.asarray(
                [np.tanh(0.25 * state_value + search_value) for search_value in search_values],
                dtype=np.float32,
            )
        else:
            candidate_values = np.asarray(
                [
                    np.tanh(
                        state_value
                        + self.config.heuristic_weight
                        * heuristic_value
                        + self.config.depth2_weight
                        * (search_value - heuristic_value)
                    )
                    for heuristic_value, search_value in zip(heuristic_values, search_values)
                ],
                dtype=np.float32,
            )

        simulations = max(1, int(self.config.simulations))
        for _ in range(simulations):
            total_visits = sum(child.visits for child in children)
            scale = np.sqrt(total_visits + 1.0)
            scores = []
            for child in children:
                u = self.config.cpuct * child.prior * scale / (1.0 + child.visits)
                scores.append(child.q + u)
            index = int(np.argmax(scores))
            value = float(candidate_values[index])
            children[index].visits += 1
            children[index].value_sum += value

        visits = np.asarray([child.visits for child in children], dtype=np.float32)
        if visits.sum() <= 0:
            visit_probs = np.full(len(candidates), 1.0 / len(candidates), dtype=np.float32)
        else:
            visit_probs = visits / visits.sum()

        if self.config.temperature and self.config.temperature > 0:
            adjusted = np.power(visit_probs, 1.0 / self.config.temperature)
            adjusted = adjusted / adjusted.sum()
            selected_index = int(self.rng.choice(len(candidates), p=adjusted))
        else:
            selected_index = int(np.argmax(visit_probs))

        logprob = float(np.log(max(float(priors[selected_index]), 1e-8)))
        value = float(children[selected_index].q)
        return SearchResult(
            order=candidates[selected_index],
            selected_index=selected_index,
            state_features=state_features,
            action_features=action_features,
            priors=priors,
            visit_probs=visit_probs,
            value=value,
            logprob=logprob,
            candidate_values=candidate_values,
            simulator_used=using_real_simulator,
            simulator_repairs=simulator_repairs if simulator_attempted else 0,
            simulator_errors=simulator_errors if simulator_attempted else 0,
            simulator_skipped_branches=(
                simulator_skipped_branches if simulator_attempted else 0
            ),
            simulator_error_details=simulator_error_details if simulator_attempted else [],
            simulator_error_stage_counts=(
                simulator_error_stage_counts if simulator_attempted else {}
            ),
        )
