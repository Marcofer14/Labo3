"""Neural prior for approximate CFR.

The tabular CFR table is still the source of regret updates. This network learns
to generalize simulator utility matrices across similar states and actions, then
acts as a prior/fallback when a table entry has too few visits.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.alphazero.features import (
    ACTION_FEATURE_SIZE,
    STATE_FEATURE_SIZE,
    battle_state_features,
    order_action_features,
    simulator_action_features,
    simulator_state_features,
)
from src.alphazero.network import AlphaZeroPolicyValueNet, build_model
from src.cfr.model import normalize_action


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    temp = max(float(temperature), 1e-4)
    logits = values.astype(np.float64) / temp
    logits -= np.max(logits)
    exp = np.exp(logits)
    total = float(exp.sum())
    if total <= 0.0 or not math.isfinite(total):
        return np.full(values.shape, 1.0 / max(1, values.size), dtype=np.float32)
    return (exp / total).astype(np.float32)


def normalize_strategy(actions: list[str], weights: dict[str, float]) -> dict[str, float]:
    actions = [normalize_action(action) for action in actions]
    values = [max(0.0, float(weights.get(action, 0.0))) for action in actions]
    total = sum(values)
    if total <= 1e-12:
        probability = 1.0 / max(1, len(actions))
        return {action: probability for action in actions}
    return {action: value / total for action, value in zip(actions, values)}


def mix_strategies(
    actions: list[str],
    primary: dict[str, float],
    prior: dict[str, float],
    prior_weight: float,
) -> dict[str, float]:
    actions = [normalize_action(action) for action in actions]
    weight = min(1.0, max(0.0, float(prior_weight)))
    mixed = {
        action: (1.0 - weight) * float(primary.get(action, 0.0))
        + weight * float(prior.get(action, 0.0))
        for action in actions
    }
    return normalize_strategy(actions, mixed)


class ResidualBlock(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(size),
            nn.Linear(size, size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(size * 2, size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CFRPriorNet(nn.Module):
    """Residual candidate ranker used as the neural prior for CFR.

    It keeps the same variable-candidate interface as the AlphaZero ranker, but
    uses separate residual state/action encoders and a wider pair head. This is
    still intentionally lightweight: the tabular CFR regrets remain the main
    learner, while this network generalizes action preferences across similar
    public states.
    """

    model_type = "cfr_residual_prior"

    def __init__(
        self,
        state_size: int = STATE_FEATURE_SIZE,
        action_size: int = ACTION_FEATURE_SIZE,
        hidden_size: int = 256,
        embedding_size: int = 192,
        layers: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.layers = layers
        self.dropout = float(dropout)

        self.state_input = nn.Sequential(
            nn.Linear(state_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.action_input = nn.Sequential(
            nn.Linear(action_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.state_blocks = nn.ModuleList(
            [ResidualBlock(embedding_size, self.dropout) for _ in range(max(1, layers))]
        )
        self.action_blocks = nn.ModuleList(
            [ResidualBlock(embedding_size, self.dropout) for _ in range(max(1, layers))]
        )
        pair_size = embedding_size * 4
        self.policy_head = nn.Sequential(
            nn.Linear(pair_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            ResidualBlock(hidden_size, self.dropout),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            ResidualBlock(hidden_size, self.dropout),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def _encode_state(self, states: torch.Tensor) -> torch.Tensor:
        x = self.state_input(states)
        for block in self.state_blocks:
            x = block(x)
        return x

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        batch_size, candidate_count, _ = actions.shape
        x = self.action_input(actions.reshape(batch_size * candidate_count, -1))
        for block in self.action_blocks:
            x = block(x)
        return x.reshape(batch_size, candidate_count, -1)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states.dim() != 2:
            raise ValueError(f"states must be [B, S], got {tuple(states.shape)}")
        if actions.dim() != 3:
            raise ValueError(f"actions must be [B, C, A], got {tuple(actions.shape)}")

        state_emb = self._encode_state(states)
        action_emb = self._encode_actions(actions)
        expanded_state = state_emb.unsqueeze(1).expand_as(action_emb)
        pair = torch.cat(
            [
                expanded_state,
                action_emb,
                expanded_state * action_emb,
                torch.abs(expanded_state - action_emb),
            ],
            dim=-1,
        )
        logits = self.policy_head(pair).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
        values = self.value_head(state_emb).squeeze(-1)
        return logits, values

    def config(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
            "embedding_size": self.embedding_size,
            "layers": self.layers,
            "dropout": self.dropout,
        }


def build_cfr_prior_model(config: dict[str, Any] | None = None) -> nn.Module:
    config = dict(config or {})
    config.pop("model_type", None)
    return CFRPriorNet(**config)


class CFRNeuralPrior:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        lr: float = 3e-4,
        hidden_size: int = 256,
        embedding_size: int = 192,
        layers: int = 3,
        dropout: float = 0.05,
        policy_smoothing: float = 0.02,
    ):
        self.device = torch.device(device)
        self.metadata: dict[str, Any] = {"examples_seen": 0, "updates": 0}
        self.policy_smoothing = float(policy_smoothing)
        if checkpoint_path and Path(checkpoint_path).exists():
            payload = torch.load(checkpoint_path, map_location=self.device)
            model_config = payload.get("model_config") or {}
            model_type = payload.get("model_type") or model_config.get("model_type")
            self.model = (
                build_cfr_prior_model(model_config)
                if model_type == CFRPriorNet.model_type
                else build_model(model_config)
            )
            self.model.load_state_dict(payload["model_state_dict"])
            self.metadata.update(payload.get("metadata") or {})
        else:
            self.model = CFRPriorNet(
                hidden_size=hidden_size,
                embedding_size=embedding_size,
                layers=layers,
                dropout=dropout,
            )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    @property
    def ready(self) -> bool:
        return int(self.metadata.get("examples_seen") or 0) > 0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "cfr-neural-prior-v1",
                "model_type": getattr(self.model, "model_type", "alphazero_ranker"),
                "model_config": self.model.config(),
                "model_state_dict": self.model.state_dict(),
                "metadata": self.metadata,
            },
            path,
        )

    def example_from_utilities(
        self,
        snapshot: dict[str, Any],
        side: str,
        actions: list[str],
        utilities: np.ndarray,
        value_target: float,
        *,
        target_temperature: float,
    ) -> dict[str, Any]:
        actions = [normalize_action(action) for action in actions]
        utilities = np.asarray(utilities, dtype=np.float32)
        return {
            "state_features": simulator_state_features(snapshot, side).astype(np.float32),
            "action_features": np.stack(
                [simulator_action_features(action) for action in actions]
            ).astype(np.float32),
            "target_probs": _softmax(utilities, target_temperature),
            "value_target": float(np.clip(value_target, -1.0, 1.0)),
            "actions": actions,
        }

    def predict_features(
        self,
        state_features: np.ndarray,
        action_features: np.ndarray,
        actions: list[str],
    ) -> tuple[dict[str, float], float]:
        actions = [normalize_action(action) for action in actions]
        if not actions:
            return {}, 0.0
        self.model.eval()
        state = torch.as_tensor(state_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_tensor = torch.as_tensor(action_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(state, action_tensor)
            probs = torch.softmax(logits.squeeze(0), dim=0).detach().cpu().numpy()
        strategy = {action: float(prob) for action, prob in zip(actions, probs)}
        return normalize_strategy(actions, strategy), float(value.squeeze(0).detach().cpu().item())

    def predict_offline(
        self,
        snapshot: dict[str, Any],
        side: str,
        actions: list[str],
    ) -> tuple[dict[str, float], float]:
        actions = [normalize_action(action) for action in actions]
        if not actions:
            return {}, 0.0
        action_features = np.stack([simulator_action_features(action) for action in actions]).astype(np.float32)
        return self.predict_features(simulator_state_features(snapshot, side), action_features, actions)

    def predict_battle(
        self,
        battle: Any,
        actions: list[str],
        orders: list[Any],
    ) -> tuple[dict[str, float], float]:
        actions = [normalize_action(action) for action in actions]
        if not actions:
            return {}, 0.0
        action_features = np.stack([order_action_features(order) for order in orders]).astype(np.float32)
        return self.predict_features(battle_state_features(battle), action_features, actions)

    def fit(
        self,
        examples: list[dict[str, Any]],
        *,
        epochs: int = 1,
        batch_size: int = 64,
    ) -> dict[str, float]:
        if not examples:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        self.model.train()
        batch_size = max(1, int(batch_size))
        epochs = max(1, int(epochs))
        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []

        for _ in range(epochs):
            order = torch.randperm(len(examples)).tolist()
            for start in range(0, len(order), batch_size):
                batch = [examples[index] for index in order[start : start + batch_size]]
                max_actions = max(len(item["target_probs"]) for item in batch)
                states = torch.zeros((len(batch), STATE_FEATURE_SIZE), dtype=torch.float32)
                actions = torch.zeros((len(batch), max_actions, ACTION_FEATURE_SIZE), dtype=torch.float32)
                targets = torch.zeros((len(batch), max_actions), dtype=torch.float32)
                mask = torch.zeros((len(batch), max_actions), dtype=torch.bool)
                values = torch.zeros((len(batch),), dtype=torch.float32)

                for row, item in enumerate(batch):
                    count = len(item["target_probs"])
                    states[row] = torch.as_tensor(item["state_features"], dtype=torch.float32)
                    actions[row, :count] = torch.as_tensor(item["action_features"], dtype=torch.float32)
                    targets[row, :count] = torch.as_tensor(item["target_probs"], dtype=torch.float32)
                    mask[row, :count] = True
                    values[row] = float(item["value_target"])

                states = states.to(self.device)
                actions = actions.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)
                values = values.to(self.device)

                logits, predicted_values = self.model(states, actions, mask)
                log_probs = torch.log_softmax(logits, dim=1)
                if self.policy_smoothing > 0:
                    valid = mask.float()
                    valid_count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                    targets = (1.0 - self.policy_smoothing) * targets
                    targets = targets + self.policy_smoothing * valid / valid_count
                policy_loss = -(targets * log_probs).sum(dim=1).mean()
                value_loss = nn.functional.mse_loss(predicted_values, values)
                loss = policy_loss + 0.25 * value_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                losses.append(float(loss.detach().cpu().item()))
                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))

        self.metadata["examples_seen"] = int(self.metadata.get("examples_seen") or 0) + len(examples)
        self.metadata["updates"] = int(self.metadata.get("updates") or 0) + len(losses)
        self.model.eval()
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        }
