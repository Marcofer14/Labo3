"""
src.rewards.layer_d · Recompensas meta / endgame (Stage 3+).

Componentes:
  D1 Lead matchup advantage    (solo turno 1)
  D2 Trade KO eficiente        (rival cae con bajo costo de HP propio)
  D3 Setup window correcto     (TR/Tailwind/screens en momento útil)
  D4 Endgame closer            (last 2v2 / cierre con último vivo)
  D5 Info denial / Tera economy (no revelamos Tera temprano)
  D6 Bringback save            (preservamos key Pokemon al switch)
  D7 Match length efficiency   (victoria en ≤ 12 turnos)
  D8 Speed control proper      (TR con team slow / TW con team fast)

Estas heurísticas miran "metainfo" del partido más que efectos atómicos
de un turno. Pesos altos para que tengan voz frente a B/C ya entrenados.
"""

from __future__ import annotations
from typing import Optional

from src.rewards.state          import TurnSnapshot, PokemonSnapshot
from src.rewards.config         import RewardConfig
from src.rewards.action_decoder import DecodedAction
from src.utils                  import get_effectiveness


# ── Helpers ──────────────────────────────────────────────────────

def _move_name_used(p: PokemonSnapshot, dec: DecodedAction) -> str:
    if not dec.is_move() or not p.species:
        return ""
    try:
        return p.moves_revealed[dec.move_idx]
    except (IndexError, AttributeError):
        return ""


def _move_obj(name: str, moves_data: dict) -> Optional[dict]:
    return moves_data.get(name) if name else None


def _avg_speed(actives: list[PokemonSnapshot]) -> float:
    speeds = [p.speed_stat for p in actives if p.species and not p.fainted and p.speed_stat]
    return sum(speeds) / len(speeds) if speeds else 0.0


# ── D1 — Lead matchup advantage ──────────────────────────────────

def _d_lead_matchup(curr, prev, cfg, type_chart, moves_data) -> tuple[float, dict]:
    """Solo en el primer turno: ¿nuestros activos tienen coverage SE
    contra los rivales? Si sí, +reward."""
    if curr.turn != 1 or prev is not None:
        return 0.0, {}
    se_count = 0
    for own_p in curr.own_active:
        if not own_p.species:
            continue
        for mname in own_p.moves_revealed:
            m = _move_obj(mname, moves_data)
            if not m or m.get("category") == "status":
                continue
            mtype = m.get("type", "")
            for rival_p in curr.rival_active:
                if rival_p.types:
                    eff = get_effectiveness(mtype, rival_p.types, type_chart)
                    if eff >= 2.0:
                        se_count += 1
                        break  # uno por movimiento
    # Threshold: al menos 2 hits SE disponibles entre nuestros 2 activos
    if se_count >= 2:
        return cfg.w_lead_matchup_advantage, {"d.lead_matchup": cfg.w_lead_matchup_advantage}
    return 0.0, {}


# ── D2 — Trade KO eficiente ──────────────────────────────────────

def _d_trade_efficient(curr, prev, cfg) -> tuple[float, dict]:
    """KO infligido + costo de HP propio < 60%."""
    if prev is None:
        return 0.0, {}
    new_kos = curr.rival_ko_count - prev.rival_ko_count
    if new_kos < 1:
        return 0.0, {}
    own_dmg = 0.0
    for i in range(2):
        prev_p = prev.own_active[i]
        curr_p = curr.own_active[i]
        if prev_p.species and curr_p.species == prev_p.species:
            own_dmg += max(0.0, prev_p.hp_pct - curr_p.hp_pct)
    if own_dmg < 0.60:
        v = cfg.w_trade_efficient_ko * new_kos
        return v, {"d.trade_efficient_ko": v}
    return 0.0, {}


# ── D3 — Setup window correcto ───────────────────────────────────

SETUP_MOVES = {"trick-room", "tailwind", "reflect", "light-screen", "aurora-veil"}

def _d_setup_window(curr, prev, last_actions, cfg) -> tuple[float, dict]:
    """Setup move usado con suficiente equipo vivo para aprovecharlo."""
    if prev is None or last_actions is None:
        return 0.0, {}
    setup_used = False
    for slot, dec in enumerate(last_actions):
        if slot >= len(prev.own_active):
            break
        prev_p = prev.own_active[slot]
        if _move_name_used(prev_p, dec) in SETUP_MOVES:
            setup_used = True
            break
    if not setup_used:
        return 0.0, {}
    # Útil si tenemos al menos 4 vivos (≥3 turnos para aprovechar)
    if curr.own_alive >= 4:
        return cfg.w_setup_window, {"d.setup_window": cfg.w_setup_window}
    return 0.0, {}


# ── D4 — Endgame closer ──────────────────────────────────────────

def _d_endgame_closer(curr, prev, cfg) -> tuple[float, dict]:
    """KO en fase 2v2 final."""
    if prev is None:
        return 0.0, {}
    new_kos = curr.rival_ko_count - prev.rival_ko_count
    if new_kos < 1:
        return 0.0, {}
    if curr.own_alive <= 2 and curr.rival_alive <= 2:
        return cfg.w_endgame_closer, {"d.endgame_closer": cfg.w_endgame_closer}
    return 0.0, {}


# ── D5 — Info denial / Tera economy ──────────────────────────────

def _d_info_denial_tera(curr, prev, last_actions, cfg) -> tuple[float, dict]:
    """
    Tera activado tarde (turn ≥ 3) → denegamos info + preservamos opción.
    Si ya activaste Tera en turno 1-2 sin razón, esto no dispara
    (y C8 puede penalizar si fue desperdicio).
    """
    if prev is None or last_actions is None or curr.turn < 3:
        return 0.0, {}
    for slot, dec in enumerate(last_actions):
        if not (dec.is_move() and dec.tera):
            continue
        if slot >= len(prev.own_active) or slot >= len(curr.own_active):
            continue
        prev_p = prev.own_active[slot]
        curr_p = curr.own_active[slot]
        if curr_p.terastallized and not prev_p.terastallized:
            return cfg.w_info_denial_tera, {"d.info_denial_tera": cfg.w_info_denial_tera}
    return 0.0, {}


# ── D6 — Bringback save ──────────────────────────────────────────

def _d_bringback_save(curr, prev, last_actions, cfg) -> tuple[float, dict]:
    """Switch out de key Pokemon a HP bajo para preservarlo."""
    if prev is None or last_actions is None:
        return 0.0, {}
    out = 0.0
    debug = {}
    for slot, dec in enumerate(last_actions):
        if not dec.is_switch():
            continue
        if slot >= len(prev.own_active) or slot >= len(curr.own_active):
            continue
        prev_p = prev.own_active[slot]
        curr_p = curr.own_active[slot]
        if not prev_p.species or prev_p.species == curr_p.species:
            continue
        # Salvamos atacante ofensivo a HP bajo (≤ 30%)
        if prev_p.hp_pct <= 0.30 and prev_p.is_offensive_attacker:
            out += cfg.w_bringback_save
            debug["d.bringback_save"] = debug.get("d.bringback_save", 0.0) + cfg.w_bringback_save
    return out, debug


# ── D7 — Match length efficiency ─────────────────────────────────

def _d_short_match(curr, prev, cfg) -> tuple[float, dict]:
    """Bonus si ganamos en ≤ 12 turnos."""
    if not curr.finished or not curr.won:
        return 0.0, {}
    if curr.turn <= 12:
        return cfg.w_short_match_bonus, {"d.short_match": cfg.w_short_match_bonus}
    return 0.0, {}


# ── D8 — Speed control proper ────────────────────────────────────

def _d_speed_control_proper(curr, prev, last_actions, cfg) -> tuple[float, dict]:
    """TR con team slow O Tailwind con team fast (decisión coherente)."""
    if prev is None or last_actions is None:
        return 0.0, {}
    tr_set = (not prev.field.trick_room) and curr.field.trick_room
    tw_set = (not prev.field.own_tailwind) and curr.field.own_tailwind
    if not (tr_set or tw_set):
        return 0.0, {}
    target_move = "trick-room" if tr_set else "tailwind"
    we_did = False
    for slot, dec in enumerate(last_actions):
        if slot >= len(prev.own_active):
            break
        if _move_name_used(prev.own_active[slot], dec) == target_move:
            we_did = True
            break
    if not we_did:
        return 0.0, {}
    own_avg   = _avg_speed(prev.own_active)
    rival_avg = _avg_speed(prev.rival_active)
    if own_avg == 0 or rival_avg == 0:
        return 0.0, {}
    if tr_set and own_avg < rival_avg * 0.8:
        return cfg.w_speed_control_proper, {"d.speed_control_tr_proper": cfg.w_speed_control_proper}
    if tw_set and own_avg > rival_avg:
        return cfg.w_speed_control_proper, {"d.speed_control_tw_proper": cfg.w_speed_control_proper}
    return 0.0, {}


# ── Compute principal ────────────────────────────────────────────

def compute(
    curr: TurnSnapshot,
    prev: Optional[TurnSnapshot],
    last_actions,
    data: dict,
    cfg:  RewardConfig,
) -> tuple[float, dict[str, float]]:
    if not cfg.enable_layer_d:
        return 0.0, {}

    type_chart = data.get("type_chart", {})
    moves      = data.get("moves", {})

    total = 0.0
    breakdown: dict[str, float] = {}

    for fn, args in [
        (_d_lead_matchup,         (curr, prev, cfg, type_chart, moves)),
        (_d_trade_efficient,      (curr, prev, cfg)),
        (_d_setup_window,         (curr, prev, last_actions, cfg)),
        (_d_endgame_closer,       (curr, prev, cfg)),
        (_d_info_denial_tera,     (curr, prev, last_actions, cfg)),
        (_d_bringback_save,       (curr, prev, last_actions, cfg)),
        (_d_short_match,          (curr, prev, cfg)),
        (_d_speed_control_proper, (curr, prev, last_actions, cfg)),
    ]:
        v, b = fn(*args)
        total += v
        breakdown.update(b)

    return total, breakdown
