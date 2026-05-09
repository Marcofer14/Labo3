"""
src.training.policy · Policy customizada para VGC.

Arquitectura:
  obs (854,) → MLP [256] (ReLU) → LSTM (hidden 128) → policy/value heads

Nota sobre action masking:
  · sb3-contrib provee MaskablePPO (sin LSTM) y RecurrentPPO (con LSTM, sin
    masking). No existe MaskableRecurrentPPO oficial.
  · Implementación: el action_mask viaja DENTRO de la observación como
    feature numérica. La red lo "ve" pero el sampling NO es estricto.
    En la práctica las acciones inválidas reciben reward 0 (default
    action de poke-env) y la política las desaprende rápido.
  · Si se prefiere masking estricto, usar `algorithm="maskable_ppo"` en
    TrainingConfig (sacrifica LSTM).

Devuelve helpers para construir el modelo según la config.
"""

from __future__ import annotations

import torch as th
import torch.nn as nn
from typing import Any


# ── Helpers de activación ────────────────────────────────────────

ACTIVATION_MAP = {
    "relu":  nn.ReLU,
    "tanh":  nn.Tanh,
    "elu":   nn.ELU,
    "gelu":  nn.GELU,
}


def get_activation_class(name: str) -> type:
    name = name.lower()
    if name not in ACTIVATION_MAP:
        raise ValueError(f"Activación desconocida: {name}. Opciones: {list(ACTIVATION_MAP)}")
    return ACTIVATION_MAP[name]


# ── Constructor de policy_kwargs según TrainingConfig ────────────

def build_policy_kwargs(train_cfg, algorithm: str) -> dict:
    """
    Devuelve el dict `policy_kwargs` que se pasa al constructor del modelo
    SB3. La forma exacta depende del algoritmo elegido.
    """
    activation_cls = get_activation_class(train_cfg.activation)

    if algorithm == "recurrent_ppo":
        # sb3-contrib RecurrentPPO: usa MlpLstmPolicy o MultiInputLstmPolicy
        return {
            "net_arch":          train_cfg.net_arch,
            "activation_fn":     activation_cls,
            "lstm_hidden_size":  train_cfg.lstm_hidden_size,
            "n_lstm_layers":     train_cfg.lstm_layers,
            "shared_lstm":       train_cfg.shared_lstm,
            "enable_critic_lstm": train_cfg.enable_critic_lstm,
        }

    if algorithm == "maskable_ppo":
        return {
            "net_arch":      train_cfg.net_arch,
            "activation_fn": activation_cls,
        }

    raise ValueError(f"algorithm desconocido: {algorithm}")


# ── Lazy imports de sb3-contrib (no rompe si no está instalado) ──

def import_recurrent_ppo():
    """Devuelve (RecurrentPPO, policy_class). Requiere sb3-contrib."""
    from sb3_contrib import RecurrentPPO
    return RecurrentPPO, "MlpLstmPolicy"


def import_maskable_ppo():
    from sb3_contrib import MaskablePPO
    return MaskablePPO, "MlpPolicy"


# ── Hook de activaciones (para callbacks) ───────────────────────

class ActivationRecorder:
    """
    Registra mean/std/saturated de activaciones de las capas del MLP.
    Se conecta vía forward hooks al feature extractor.
    """

    def __init__(self):
        self._stats: list[dict] = []
        self._hooks = []

    def attach(self, model) -> None:
        """Conecta hooks a las capas Linear+activation del policy MLP."""
        try:
            mlp = self._find_mlp(model.policy)
        except AttributeError:
            return
        for i, mod in enumerate(mlp):
            if isinstance(mod, (nn.ReLU, nn.Tanh, nn.ELU, nn.GELU)):
                h = mod.register_forward_hook(self._make_hook(i))
                self._hooks.append(h)

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_hook(self, idx: int):
        def hook(module, _inp, out):
            with th.no_grad():
                t = out.detach().float()
                stat = {
                    "layer":          idx,
                    "mean":           float(t.mean().item()),
                    "std":            float(t.std().item()),
                    "frac_zero":      float((t.abs() < 1e-6).float().mean().item()),
                    "frac_saturated": float((t.abs() > 0.99).float().mean().item()),
                }
                self._stats.append(stat)
        return hook

    @staticmethod
    def _find_mlp(policy) -> nn.Sequential:
        # Busca el MLP del feature extractor en distintos layouts
        if hasattr(policy, "mlp_extractor"):
            return policy.mlp_extractor.policy_net
        if hasattr(policy, "features_extractor"):
            return policy.features_extractor
        raise AttributeError("No encuentro MLP en la policy")

    def drain(self) -> list[dict]:
        out = list(self._stats)
        self._stats.clear()
        return out
