"""
src.rewards.config
─────────────────────────────────────────────────────────────────
RewardConfig: pesos y flags on/off de los 4 layers.

El curriculum scheduler muta esta config por stage. Por defecto:
  Stage 1: A + B básico (dmg/KO/win)        → aprende qué acciones son válidas
  Stage 2: A + B completo                    → aprende táctica de turno
  Stage 3: A + B + C                         → aprende estrategia de campo
  Stage 4: A + B + C + D                     → aprende meta-game
  Stage 5: full + self-play league           → robustez
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass
class RewardConfig:
    # ── meta ──
    stage:        int  = 1
    debug:        bool = False     # devolver breakdown por componente

    # ── Layer A · terminal ──────────────────────────────────────
    enable_layer_a: bool  = True
    w_win:          float = 15.0

    # ── Layer B · táctico (siempre activo) ──────────────────────
    enable_layer_b: bool  = True

    # B-existentes
    w_dmg_dealt:    float =  1.0      # por cada 1% HP infligido
    w_dmg_taken:    float = -0.8      # por cada 1% HP recibido
    w_ko:           float =  3.0      # KO infligido
    w_ko_taken:     float = -2.5      # KO recibido

    # B-nuevos (controlados por flag separado para portear primero lo viejo)
    enable_b_extra: bool  = False

    # B5 — Protect inteligente
    w_protect_blocked_per_pct: float =  0.08    # por 1% de HP bloqueado
    w_hit_protect:             float = -0.10    # gastamos turno en rival protegido

    # B6 — Super efectivo
    w_super_effective_per_step: float = 0.06    # por cada step ×2 sobre 1×

    # B7 — KO antes que el rival se mueva
    w_ko_outspeed:  float = 0.12

    # B8 — Status infligido
    w_status_burn_phys:    float = 0.10   # burn en atacante físico (atk base > 90)
    w_status_burn_other:   float = 0.04
    w_status_paralysis:    float = 0.08
    w_status_sleep:        float = 0.12
    w_status_freeze:       float = 0.10
    w_status_poison_wall:  float = 0.06   # poison/toxic en defensor (def base > 90)
    w_status_poison_other: float = 0.02

    # B9 — Focus fire (ambos al mismo target)
    w_focus_fire_same_target: float = 0.05
    w_focus_fire_combo_ko:    float = 0.08   # extra si cae ese rival

    # B10 — Spread doble hit
    w_spread_double_hit: float = 0.04

    # B11 — PP eficiencia
    w_pp_waste:  float = -0.01   # usar move con PP < 20% restante teniendo opciones

    # B12 — Sash / focus survival
    w_sash_survive: float = 0.06

    # ── Layer C · estratégico ───────────────────────────────────
    enable_layer_c: bool = False
    # Multiplicador 0.0 → 1.0 que sube linealmente al transicionar a stage 3
    # (lo controla el curriculum scheduler para introducir C gradualmente)
    layer_c_scale:  float = 1.0

    # C1 — Intimidate
    w_intimidate_attacker: float = 0.12   # a Atk base > 90
    w_intimidate_other:    float = 0.02

    # C2 — Climate override
    w_climate_override:    float = 0.15

    # C3 — Switch evita SE
    w_switch_smart:        float = 0.10

    # C4 — Tailwind
    w_tailwind_slow_side: float = 0.15
    w_tailwind_early:     float = 0.05    # turn ≤ 2
    w_tailwind_wasted:    float = -0.05

    # C5 — Trick Room
    w_tr_slow_team:    float =  0.12
    w_tr_remove_fast:  float =  0.10
    w_tr_wasted:       float = -0.08

    # C6 — Fake Out
    w_fakeout_priority:   float = 0.10
    w_fakeout_combo_ko:   float = 0.15
    w_fakeout_boosted:    float = 0.08

    # C7 — Helping Hand
    w_hh_dmg_thresh:    float = 0.08    # partner dealt > 50% HP
    w_hh_ko_unlock:     float = 0.15    # partner KO con HH (no llegaba sin)
    w_hh_wasted:        float = -0.04

    # C8 — Tera
    w_tera_remove_weakness: float =  0.20
    w_tera_stab:            float =  0.12
    w_tera_wasted:          float = -0.05

    # C9 — Pivot (Parting Shot / U-turn)
    w_parting_shot_offensive: float = 0.08    # bajó stats a atacante
    w_uturn_better_matchup:   float = 0.06

    # C10 — Redirection
    w_redirect_lethal: float = 0.15
    w_redirect_strong: float = 0.08

    # C11 — Double threat trap
    w_double_threat_match:  float = 0.10
    w_spread_covers_both:   float = 0.08

    # C12 — HP advantage late
    w_hp_advantage_late: float = 0.05

    # ── Layer D · meta / endgame ────────────────────────────────
    enable_layer_d: bool = False

    w_lead_matchup_advantage: float = 0.20    # D1 turn 1 only
    w_trade_efficient_ko:     float = 0.15    # D2
    w_setup_window:           float = 0.18    # D3
    w_endgame_closer:         float = 0.25    # D4 (last alive vs last alive)
    w_info_denial_tera:       float = 0.10    # D5 hold tera until valuable
    w_bringback_save:         float = 0.20    # D6 saved key Pokemon
    w_short_match_bonus:      float = 0.10    # D7 win turn ≤ 12
    w_speed_control_proper:   float = 0.08    # D8

    # ── Construcción por stage ──────────────────────────────────

    @classmethod
    def stage_1(cls, **kw) -> "RewardConfig":
        return cls(stage=1, enable_layer_a=True, enable_layer_b=True,
                   enable_b_extra=False, enable_layer_c=False, enable_layer_d=False, **kw)

    @classmethod
    def stage_2(cls, **kw) -> "RewardConfig":
        return cls(stage=2, enable_layer_a=True, enable_layer_b=True,
                   enable_b_extra=True, enable_layer_c=False, enable_layer_d=False, **kw)

    @classmethod
    def stage_3(cls, layer_c_scale: float = 1.0, **kw) -> "RewardConfig":
        return cls(stage=3, enable_layer_a=True, enable_layer_b=True,
                   enable_b_extra=True, enable_layer_c=True, enable_layer_d=False,
                   layer_c_scale=layer_c_scale, **kw)

    @classmethod
    def stage_4(cls, **kw) -> "RewardConfig":
        return cls(stage=4, enable_layer_a=True, enable_layer_b=True,
                   enable_b_extra=True, enable_layer_c=True, enable_layer_d=True, **kw)

    @classmethod
    def stage_5(cls, **kw) -> "RewardConfig":
        # Stage 5: igual a 4 pero pensado para self-play league
        return cls(stage=5, enable_layer_a=True, enable_layer_b=True,
                   enable_b_extra=True, enable_layer_c=True, enable_layer_d=True, **kw)

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
