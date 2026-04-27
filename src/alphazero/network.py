"""Policy/value network for the AlphaZero-style candidate ranker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.alphazero.features import ACTION_FEATURE_SIZE, STATE_FEATURE_SIZE


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, layers: int, output_size: int):
        super().__init__()
        blocks: list[nn.Module] = []
        last_size = input_size
        for _ in range(max(layers, 1)):
            blocks.extend(
                [
                    nn.Linear(last_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                ]
            )
            last_size = hidden_size
        blocks.append(nn.Linear(last_size, output_size))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AlphaZeroPolicyValueNet(nn.Module):
    """Scores legal action candidates and estimates the current state's value."""

    def __init__(
        self,
        state_size: int = STATE_FEATURE_SIZE,
        action_size: int = ACTION_FEATURE_SIZE,
        hidden_size: int = 256,
        embedding_size: int = 192,
        layers: int = 2,
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.layers = layers

        self.state_encoder = MLP(state_size, hidden_size, layers, embedding_size)
        self.action_encoder = MLP(action_size, hidden_size, layers, embedding_size)
        self.policy_head = MLP(embedding_size * 3, hidden_size, 1, 1)
        self.value_head = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return candidate logits [B, C] and value [B]."""
        if states.dim() != 2:
            raise ValueError(f"states must be [B, S], got {tuple(states.shape)}")
        if actions.dim() != 3:
            raise ValueError(f"actions must be [B, C, A], got {tuple(actions.shape)}")

        state_emb = self.state_encoder(states)
        batch_size, candidate_count, _ = actions.shape
        flat_actions = actions.reshape(batch_size * candidate_count, -1)
        action_emb = self.action_encoder(flat_actions).reshape(
            batch_size, candidate_count, -1
        )
        expanded_state = state_emb.unsqueeze(1).expand(-1, candidate_count, -1)
        pair = torch.cat(
            [expanded_state, action_emb, expanded_state * action_emb],
            dim=-1,
        )
        logits = self.policy_head(pair).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
        values = self.value_head(state_emb).squeeze(-1)
        return logits, values

    def config(self) -> dict[str, Any]:
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
            "embedding_size": self.embedding_size,
            "layers": self.layers,
        }


def build_model(config: dict[str, Any] | None = None) -> AlphaZeroPolicyValueNet:
    config = dict(config or {})
    return AlphaZeroPolicyValueNet(**config)


def save_checkpoint(
    path: str | Path,
    model: AlphaZeroPolicyValueNet,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": model.config(),
        "model_state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> AlphaZeroPolicyValueNet:
    payload = torch.load(path, map_location=device)
    model = build_model(payload.get("model_config"))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model
