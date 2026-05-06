"""
src.rewards.layer_c · Recompensas estratégicas (Stage 2+).

Componentes:
  C1  Intimidate                (penaliza atacantes físicos)
  C2  Climate override          (cambiamos el weather del rival)
  C3  Switch smart              (switch evita SE)
  C4  Tailwind timing           (set en lado lento / temprano)
  C5  Trick Room                (set con team slow)
  C6  Fake Out                  (priority + combo + boosted target)
  C7  Helping Hand              (partner deal big dmg / KO unlock)
  C8  Tera                      (remove weakness / STAB / no wasted)
  C9  Pivot (U-turn / Parting Shot / Volt Switch)
  C10 Redirection (Rage Powder / Follow Me)
  C11 Double threat             (ambos rivales weak al mismo tipo)
  C12 HP advantage late game

Las heurísticas son aproximadas: poke-env no expone log atómico de
cada evento, así que inferimos efectos a partir del diff prev→curr.
"""

from __future__ import annotations
from typing import Optional

from src.rewards.state          import TurnSnapshot, PokemonSnapshot
from src.rewards.config         import RewardConfig
from src.rewards.action_decoder import DecodedAction
from src.utils                  import get_effectiveness


# ── Helper común ──────────────────────────────────────────────────

def _move_name_used(p: PokemonSnapshot, dec: DecodedAction) -> str:
    """Resuelve el move name a partir del move_idx (0-3) si existe."""
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


# ── C1 — Intimidate ───────────────────────────────────────────────

def _c_intimidate(curr, prev, last_actions, cfg) -> tuple[float, dict]:
    if prev is None or last_actions is None:
        return 0.0, {}
    # Detectar nuestro switch este turno
    our_switch = any(d.is_switch() for d in last_actions)
    if not our_switch:
        return 0.0, {}
    out = 0.0
    debug = {}
    for i in range(2):
        prev_r = prev.rival_active[i]
        curr_r = curr.rival_active[i]
        if not curr_r.species or curr_r.species != prev_r.species:
            continue
        if curr_r.boosts.get("atk", 0) < prev_r.boosts.get("atk", 0):
            w = cfg.w_intimidate_attacker if curr_r.is_physical_attacker else cfg.w_intimidate_other
            out += w
            key = "c.intimidate_atk" if curr_r.is_physical_attacker else "c.intimidate_other"
            debug[key] = debug.get(key, 0.0) + w
    return out, debug


# ── C2 — Climate override ─────────────────────────────────────────

def _c_climate_override(curr, prev, cfg) -> tuple[float, dict]:
    if prev is None or prev.field.weather == curr.field.weather:
        return 0.0, {}
    rival_typical = {"rain", "sun"}
    own_typical   = {"sandstorm", "snow"}
    if prev.field.weather in rival_typical and curr.field.weather in own_typical:
        return cfg.w_climate_override, {"c.climate_override": cfg.w_climate_override}
    return 0.0, {}


# ── C3 — Switch evita SE ──────────────────────────────────────────

def _c_switch_smart(curr, prev, last_actions, cfg, type_chart, moves_data) -> tuple[float, dict]:
    if prev is None or last_actions is None:
        return 0.0, {}
    out = 0.0
    debug = {}
    for slot, dec in enumerate(last_actions):
        if not dec.is_switch():
            continue
        new_p = curr.own_active[slot]
        old_p = prev.own_active[slot]
        if not new_p.species or not old_p.types:
            continue
        for rival_p in curr.rival_active:
            if not rival_p.species:
                continue
            for mname in rival_p.moves_revealed:
                m = _move_obj(mname, moves_data)
                if not m or m.get("category") == "status":
                    continue
                mtype = m.get("type", "")
                old_eff = get_effectiveness(mtype, old_p.types, type_chart)
                new_eff = get_effectiveness(mtype, new_p.types, type_chart)
                if old_eff >= 2.0 and new_eff <= 1.0:
                    out += cfg.w_switch_smart
                    debug["c.switch_smart"] = debug.get("c.switch_smart", 0.0) + cfg.w_switch_smart
                    break
    return out, debug


# ── C4 — Tailwind timing ──────────────────────────────────────────

def _c_tailwind(curr, prev, last_actions, cfg) -> tuple[float, dict]:
    if prev is None or prev.field.own_tailwind or not curr.field.own_tailwind:
        return 0.0, {}
    own_avg   = _avg_speed(prev.own_active)
    rival_avg = _avg_speed(prev.rival_active)
    if own_avg == 0 or rival_avg == 0:
        return 0.0, {}
    out = 0.0
    debug = {}
    if own_avg < rival_avg:
        out += cfg.w_tailwind_slow_side
        debug["c.tailwind_slow"] = cfg.w_tailwind_slow_side
    elif own_avg > rival_avg * 1.2:
        out += cfg.w_tailwind_wasted
        debug["c.tailwind_wasted"] = cfg.w_tailwind_wasted
    if curr.turn <= 2:
        out += cfg.w_tailwind_early
        debug["c.tailwind_early"] = cfg.w_tailwind_early
    return out, debug


# ── C5 — Trick Room timing ────────────────────────────────────────

def _c_trick_room(curr, prev, last_actions, cfg) -> tuple[float, dict]:
    if prev is None or last_actions is None:
        return 0.0, {}
    set_tr = (not prev.field.trick_room) and curr.field.trick_room
    if not set_tr:
        return 0.0, {}
    # ¿Lo seteamos nosotros? (action = move "trick-room")
    we_used_tr = False
    for slot, dec in enumerate(last_actions):
        prev_p = prev.own_active[slot]
        if _move_name_used(prev_p, dec) == "trick-room":
            we_used_tr = True
            break
    if not we_used_tr:
        return 0.0, {}
    own_avg   = _avg_speed(prev.own_active)
    rival_avg = _avg_speed(prev.rival_active)
    if own_avg == 0 or rival_avg == 0:
        return 0.0, {}
    out = 0.0
    debug = {}
    if own_avg < rival_avg:
        out += cfg.w_tr_slow_team
        debug["c.tr_slow_team"] = cfg.w_tr_slow_team
    elif own_avg > rival_avg * 1.2:
        out += cfg.w_tr_wasted
        debug["c.tr_wasted"] = cfg.w_tr_wasted
    return out, debug


# ── C6 — Fake Out ─────────────────────────────────────────────────

def _c_fake_out(curr, prev, last_actions, cfg) -> tuple[float, dict]:
    if prev is None or last_actions is None:
        return 0.0, {}
    out = 0.0
    debug = {}
    for slot, dec in enumerate(last_actions):
        if not dec.is_move() or dec.target >= 0:
            continue
        own_p = curr.own_active[slot]
        if _move_name_used(own_p, dec) != "fake-out":
            continue
        rival_idx = abs(dec.target) - 1
        if not (0 <= rival_idx <= 1):
            continue
        prev_r = prev.rival_active[rival_idx]
        curr_r = curr.rival_active[rival_idx]
        if not prev_r.species or curr_r.species != prev_r.species:
            continue
        if curr_r.hp_pct >= prev_r.hp_pct:
            continue   # no conectó
        out += cfg.w_fakeout_priority
        debug["c.fakeout"] = debug.get("c.fakeout", 0.0) + cfg.w_fakeout_priority
        # Target con boosts +stat
        if any(v > 0 for v in prev_r.boosts.values()):
            out += cfg.w_fakeout_boosted
            debug["c.fakeout_boosted"] = debug.get("c.fakeout_boosted", 0.0) + cfg.w_fakeout_boosted
        # Combo KO ese turno
        if curr_r.hp_pct == 0:
            out += cfg.w_fakeout_combo_ko
            debug["c.fakeout_combo_ko"] = debug.get("c.fakeout_combo_ko", 0.0) + cfg.w_fakeout_combo_ko
    return out, debug


# ── C7 — Helping Hand ─────────────────────────────────────────────

def _c_helping_hand(curr, prev, last_actions, cfg) -> tuple[float, dict]:
    if prev is None or last_actions is None:
        return 0.0, {}
    hh_used = False
    for slot, dec in enumerate(last_actions):
        if _move_name_used(curr.own_active[slot], dec) == "helping-hand":
            hh_used = True
            break
    if not hh_used:
        return 0.0, {}
    # Daño total infligido por partner ese turno
    partner_dmg = 0.0
    for prev_r, curr_r in zip(prev.rival_active, curr.rival_active):
        if prev_r.species and prev_r.species == curr_r.species:
            partner_dmg += max(0.0, prev_r.hp_pct - curr_r.hp_pct)
    out = 0.0
    debug = {}
    if partner_dmg > 0.5:
        out += cfg.w_hh_dmg_thresh
        debug["c.hh_big_dmg"] = cfg.w_hh_dmg_thresh
    elif partner_dmg < 0.05:
        out += cfg.w_hh_wasted
        debug["c.hh_wasted"] = cfg.w_hh_wasted
    new_kos = curr.rival_ko_count - prev.rival_ko_count
    if new_kos >= 1 and partner_dmg > 0.3:
        out += cfg.w_hh_ko_unlock
        debug["c.hh_ko_unlock"] = cfg.w_hh_ko_unlock
    return out, debug


# ── C8 — Tera activation smart ────────────────────────────────────

def _c_tera(curr, prev, last_actions, cfg, moves_data, type_chart) -> tuple[float, dict]:
    if prev is None or last_actions is None:
        return 0.0, {}
    out = 0.0
    debug = {}
    for slot, dec in enumerate(last_actions):
        if not (dec.is_move() and dec.tera):
            continue
        own_p  = curr.own_active[slot]
        prev_p = prev.own_active[slot]
        if not own_p.species or not own_p.terastallized or prev_p.terastallized:
            continue
        # Tera removió debilidad?
        new_types = [own_p.tera_type] if own_p.tera_type else own_p.types
        old_types = prev_p.types
        for rival_p in curr.rival_active:
            if not rival_p.species:
                continue
            improved = False
            for mname in rival_p.moves_revealed:
                m = _move_obj(mname, moves_data)
                if not m or m.get("category") == "status":
                    continue
                mtype = m.get("type", "")
                old_eff = get_effectiveness(mtype, old_types, type_chart)
                new_eff = get_effectiveness(mtype, new_types, type_chart)
                if old_eff >= 2.0 and new_eff <= 1.0:
                    improved = True
                    break
            if improved:
                out += cfg.w_tera_remove_weakness
                debug["c.tera_remove_weakness"] = debug.get("c.tera_remove_weakness", 0.0) + cfg.w_tera_remove_weakness
                break
        # STAB en el move usado
        m = _move_obj(_move_name_used(own_p, dec), moves_data)
        if m and own_p.tera_type and m.get("type") == own_p.tera_type:
            out += cfg.w_tera_stab
            debug["c.tera_stab"] = debug.get("c.tera_stab", 0.0) + cfg.w_tera_stab
        # Tera con HP bajo (desperdicio)
        if 0 < own_p.hp_pct < 0.20:
            out += cfg.w_tera_wasted
            debug["c.tera_wasted"] = debug.get("c.tera_wasted", 0.0) + cfg.w_tera_wasted
    return out, debug


# ── C9 — Pivot (U-turn / Parting Shot / Volt Switch) ─────────────

PIVOT_MOVES = {"u-turn", "volt-switch", "flip-turn", "parting-shot", "teleport"}

def _c_pivot(curr, prev, last_actions, cfg, moves_data, type_chart) -> tuple[float, dict]:
    if prev is None or last_actions is None:
        return 0.0, {}
    out = 0.0
    debug = {}
    for slot, dec in enumerate(last_actions):
        prev_p = prev.own_active[slot]
        new_p  = curr.own_active[slot]
        mname  = _move_name_used(prev_p, dec)
        if mname not in PIVOT_MOVES:
            continue
        # Mejor matchup post-pivot?
        if new_p.species and new_p.species != prev_p.species and new_p.types:
            best = False
            for rival_p in curr.rival_active:
                if not rival_p.species:
                    continue
                for rmname in rival_p.moves_revealed:
                    rm = _move_obj(rmname, moves_data)
                    if not rm or rm.get("category") == "status":
                        continue
                    rt = rm.get("type", "")
                    old_eff = get_effectiveness(rt, prev_p.types, type_chart)
                    new_eff = get_effectiveness(rt, new_p.types, type_chart)
                    if old_eff > 1.0 and new_eff <= 1.0:
                        best = True
                        break
                if best:
                    break
            if best:
                out += cfg.w_uturn_better_matchup
                debug["c.pivot_better_matchup"] = debug.get("c.pivot_better_matchup", 0.0) + cfg.w_uturn_better_matchup
        # Parting Shot bajó stats a un atacante?
        if mname == "parting-shot":
            for i in range(2):
                prev_r = prev.rival_active[i]
                curr_r = curr.rival_active[i]
                if not prev_r.species or curr_r.species != prev_r.species:
                    continue
                drop_atk = curr_r.boosts.get("atk", 0) < prev_r.boosts.get("atk", 0)
                drop_spa = curr_r.boosts.get("spa", 0) < prev_r.boosts.get("spa", 0)
                if (drop_atk or drop_spa) and prev_r.is_offensive_attacker:
                    out += cfg.w_parting_shot_offensive
                    debug["c.parting_shot_off"] = debug.get("c.parting_shot_off", 0.0) + cfg.w_parting_shot_offensive
                    break
    return out, debug


# ── C10 — Redirection ─────────────────────────────────────────────

REDIRECT_MOVES = {"rage-powder", "follow-me"}

def _c_redirect(curr, prev, last_actions, cfg) -> tuple[float, dict]:
    if prev is None or last_actions is None:
        return 0.0, {}
    out = 0.0
    debug = {}
    for slot, dec in enumerate(last_actions):
        own_p = curr.own_active[slot]
        if _move_name_used(own_p, dec) not in REDIRECT_MOVES:
            continue
        # Tomamos daño Y partner intacto → redirigimos
        prev_hp_self    = prev.own_active[slot].hp_pct
        curr_hp_self    = own_p.hp_pct
        damage_self     = max(0.0, prev_hp_self - curr_hp_self)

        partner_idx     = 1 - slot
        prev_partner    = prev.own_active[partner_idx]
        curr_partner    = curr.own_active[partner_idx]
        partner_dmg     = (max(0.0, prev_partner.hp_pct - curr_partner.hp_pct)
                           if prev_partner.species == curr_partner.species else 0.0)

        if damage_self > 0.30 and partner_dmg < 0.05 and prev_partner.species:
            if damage_self > 0.60 or curr_hp_self == 0:
                out += cfg.w_redirect_lethal
                debug["c.redirect_lethal"] = debug.get("c.redirect_lethal", 0.0) + cfg.w_redirect_lethal
            else:
                out += cfg.w_redirect_strong
                debug["c.redirect_strong"] = debug.get("c.redirect_strong", 0.0) + cfg.w_redirect_strong
    return out, debug


# ── C11 — Double threat ───────────────────────────────────────────

def _c_double_threat(curr, prev, cfg, moves_data, type_chart) -> tuple[float, dict]:
    rival_a = curr.rival_active[0]
    rival_b = curr.rival_active[1]
    if not (rival_a.species and rival_b.species and not rival_a.fainted and not rival_b.fainted):
        return 0.0, {}
    out = 0.0
    debug = {}
    for own_p in curr.own_active:
        if not own_p.species or own_p.fainted:
            continue
        for mname in own_p.moves_revealed:
            m = _move_obj(mname, moves_data)
            if not m or m.get("category") == "status":
                continue
            mtype = m.get("type", "")
            eff_a = get_effectiveness(mtype, rival_a.types, type_chart) if rival_a.types else 1.0
            eff_b = get_effectiveness(mtype, rival_b.types, type_chart) if rival_b.types else 1.0
            if eff_a >= 2.0 and eff_b >= 2.0:
                tk = m.get("target", "") or ""
                is_spread = "all-" in tk or "opponent" in tk
                if is_spread:
                    out += cfg.w_spread_covers_both
                    debug["c.spread_covers_both"] = cfg.w_spread_covers_both
                else:
                    out += cfg.w_double_threat_match
                    debug["c.double_threat_match"] = cfg.w_double_threat_match
                return out, debug    # uno por turno
    return out, debug


# ── C12 — HP advantage late game ─────────────────────────────────

def _c_hp_advantage(curr, prev, cfg) -> tuple[float, dict]:
    if curr.turn < 5:
        return 0.0, {}
    if curr.own_team_hp <= curr.rival_team_hp + 0.10:
        return 0.0, {}
    scale = min(1.0, curr.turn / 15.0)
    val   = cfg.w_hp_advantage_late * scale
    return val, {"c.hp_advantage_late": val}


# ── Compute principal ────────────────────────────────────────────

def compute(
    curr: TurnSnapshot,
    prev: Optional[TurnSnapshot],
    last_actions,
    data: dict,
    cfg:  RewardConfig,
) -> tuple[float, dict[str, float]]:
    if not cfg.enable_layer_c:
        return 0.0, {}

    type_chart = data.get("type_chart", {})
    moves      = data.get("moves", {})

    total = 0.0
    breakdown: dict[str, float] = {}

    for fn, args in [
        (_c_intimidate,        (curr, prev, last_actions, cfg)),
        (_c_climate_override,  (curr, prev, cfg)),
        (_c_switch_smart,      (curr, prev, last_actions, cfg, type_chart, moves)),
        (_c_tailwind,          (curr, prev, last_actions, cfg)),
        (_c_trick_room,        (curr, prev, last_actions, cfg)),
        (_c_fake_out,          (curr, prev, last_actions, cfg)),
        (_c_helping_hand,      (curr, prev, last_actions, cfg)),
        (_c_tera,              (curr, prev, last_actions, cfg, moves, type_chart)),
        (_c_pivot,             (curr, prev, last_actions, cfg, moves, type_chart)),
        (_c_redirect,          (curr, prev, last_actions, cfg)),
        (_c_double_threat,     (curr, prev, cfg, moves, type_chart)),
        (_c_hp_advantage,      (curr, prev, cfg)),
    ]:
        v, b = fn(*args)
        total += v
        breakdown.update(b)

    return total, breakdown
