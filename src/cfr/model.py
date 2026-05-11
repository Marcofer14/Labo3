"""Tabular regret-matching model for approximate CFR."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def normalize_action(action: Any) -> str:
    text = str(action or "").strip()
    text = re.sub(r"^/choose\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text


def _uniform(actions: list[str]) -> dict[str, float]:
    if not actions:
        return {}
    probability = 1.0 / len(actions)
    return {action: probability for action in actions}


def entropy(strategy: dict[str, float]) -> float:
    total = 0.0
    for probability in strategy.values():
        if probability > 0:
            total -= probability * math.log(probability)
    return total


@dataclass
class CFRNode:
    regret_sum: dict[str, float] = field(default_factory=dict)
    strategy_sum: dict[str, float] = field(default_factory=dict)
    visits: int = 0

    def strategy(self, actions: list[str], *, average: bool = False) -> dict[str, float]:
        actions = [normalize_action(action) for action in actions]
        if not actions:
            return {}
        if average:
            weights = {action: max(0.0, float(self.strategy_sum.get(action, 0.0))) for action in actions}
        else:
            weights = {action: max(0.0, float(self.regret_sum.get(action, 0.0))) for action in actions}
        total = sum(weights.values())
        if total <= 1e-12:
            return _uniform(actions)
        return {action: weight / total for action, weight in weights.items()}

    def update(self, actions: list[str], strategy: dict[str, float], regrets: dict[str, float]) -> None:
        self.visits += 1
        for action in actions:
            action = normalize_action(action)
            self.regret_sum[action] = float(self.regret_sum.get(action, 0.0)) + float(regrets.get(action, 0.0))
            self.strategy_sum[action] = (
                float(self.strategy_sum.get(action, 0.0)) + float(strategy.get(action, 0.0))
            )

    def positive_regret(self, actions: list[str] | None = None) -> float:
        values = self.regret_sum
        if actions is not None:
            keys = [normalize_action(action) for action in actions]
            values = {key: values.get(key, 0.0) for key in keys}
        return sum(max(0.0, float(value)) for value in values.values())


class CFRModel:
    def __init__(self, *, metadata: dict[str, Any] | None = None):
        self.nodes: dict[str, CFRNode] = {}
        self.metadata = dict(metadata or {})

    def node(self, key: str) -> CFRNode:
        if key not in self.nodes:
            self.nodes[key] = CFRNode()
        return self.nodes[key]

    def strategy(self, key: str, actions: list[str], *, average: bool = False) -> dict[str, float]:
        return self.node(key).strategy(actions, average=average)

    def update(self, key: str, actions: list[str], strategy: dict[str, float], regrets: dict[str, float]) -> None:
        self.node(key).update(actions, strategy, regrets)

    def select_action(
        self,
        key: str,
        actions: list[str],
        *,
        average: bool = False,
        temperature: float = 1.0,
        rng: random.Random | None = None,
    ) -> str:
        actions = [normalize_action(action) for action in actions]
        if not actions:
            return ""
        strategy = self.strategy(key, actions, average=average)
        if temperature <= 0:
            return max(actions, key=lambda action: strategy.get(action, 0.0))
        if temperature != 1.0:
            weights = [strategy.get(action, 0.0) ** (1.0 / max(temperature, 1e-6)) for action in actions]
            total = sum(weights)
            if total > 0:
                strategy = {action: weight / total for action, weight in zip(actions, weights)}
        rng = rng or random
        threshold = rng.random()
        cumulative = 0.0
        for action in actions:
            cumulative += strategy.get(action, 0.0)
            if threshold <= cumulative:
                return action
        return actions[-1]

    def stats(self) -> dict[str, Any]:
        visits = [node.visits for node in self.nodes.values()]
        regrets = [node.positive_regret() for node in self.nodes.values()]
        return {
            "information_sets": len(self.nodes),
            "visited_information_sets": sum(1 for value in visits if value > 0),
            "total_visits": sum(visits),
            "avg_positive_regret": sum(regrets) / len(regrets) if regrets else 0.0,
            "max_positive_regret": max(regrets) if regrets else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cfr-tabular-v1",
            "metadata": self.metadata,
            "nodes": {
                key: {
                    "regret_sum": node.regret_sum,
                    "strategy_sum": node.strategy_sum,
                    "visits": node.visits,
                }
                for key, node in self.nodes.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CFRModel":
        model = cls(metadata=payload.get("metadata") or {})
        for key, item in (payload.get("nodes") or {}).items():
            model.nodes[str(key)] = CFRNode(
                regret_sum={str(k): float(v) for k, v in (item.get("regret_sum") or {}).items()},
                strategy_sum={str(k): float(v) for k, v in (item.get("strategy_sum") or {}).items()},
                visits=int(item.get("visits") or 0),
            )
        return model

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CFRModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
