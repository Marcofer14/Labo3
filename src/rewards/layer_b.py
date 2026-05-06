"""
src.rewards.layer_b · Recompensas tácticas (cada turno).

Activado en todos los stages.

Componentes:
  B1  Daño infligido           (siempre)
  B2  Daño recibido            (siempre)
  B3  KO infligido             (siempre)
  B4  KO recibido              (siempre)
  ── B-extra (enable_b_extra) ─
  B5  Protect inteligente / hit Protect
  B6  Super efectivo
  B7  KO antes que el rival se mueva (outspeed)
  B8  Status infligido
  B9  Focus fire (ambos al mismo target)
  B10 Spread doble hit
  B11 PP eficiencia
  B12 Sash / Sturdy survive
"""

from __future__ import annotations
from typing import Optional

from src.rewards.state          import TurnSnapshot, PokemonSnapshot
from src.rewards.config         import RewardConfig
from src.rewards.action_decoder import DecodedAction
from src.utils                  import get_effectiveness


# ── Helpers ───────────────────────────────────────────────────────

def _damage_dealt(curr: TurnSnapshot, prev: TurnSnapshot) -> list[float]:
    """% HP que perdieron los rivales activos este turno (sin contar reentradas)."""
    out = []
    for i in range(2):
        prev_hp = prev.rival_active[i].hp_pct if prev else 1.0
        curr_hp = curr.rival_active[i].hp_pct
        # Si el slot cambió de Pokemon (switch del rival), el delta no es daño nuestro
        same = (prev and prev.rival_active[i].species == curr.rival_active[i].species
                and curr.rival_active[i].species != "")
        out.append(max(0.0, prev_hp - curr_hp) if same else 0.0)
    return out


def _damage_taken(curr: TurnSnapshot, prev: TurnSnapshot) -> list[float]:
    out = []
    for i in range(2):
        prev_hp = prev.own_active[i].hp_pct if prev else 1.0
        curr_hp = curr.own_active[i].hp_pct
        same = (prev and prev.own_active[i].species == curr.own_active[i].species
                and curr.own_active[i].species != "")
        out.append(max(0.0, prev_hp - curr_hp) if same else 0.0)
    return out


# ── Componentes B1–B4 (siempre on) ────────────────────────────────

def _b_damage(curr, prev, cfg) -> tuple[float, dict]:
    if prev is None:
        return 0.0, {}
    dealt = _damage_dealt(curr, prev)
    taken = _damage_taken(curr, prev)
    r_dealt = sum(dealt) * 100.0 * cfg.w_dmg_dealt
    r_taken = sum(taken) * 100.0 * cfg.w_dmg_taken     # negativo
    return r_dealt + r_taken, {
        "b.dmg_dealt": r_dealt,
        "b.dmg_taken": r_taken,
    }


def _b_ko(curr, prev, cfg) -> tuple[float, dict]:
    if prev is None:
        return 0.0, {}
    new_rival_ko = curr.rival_ko_count - prev.rival_ko_count
    new_own_ko   = curr.own_ko_count   - prev.own_ko_count
    r_kos      = new_rival_ko * cfg.w_ko
    r_ko_taken = new_own_ko   * cfg.w_ko_taken
    return r_kos + r_ko_taken, {
        "b.ko":       r_kos,
        "b.ko_taken": r_ko_taken,
    }


# ── Componentes B-extra ───────────────────────────────────────────

def _b_protect(curr, prev, last_actions, cfg, type_chart) -> tuple[float, dict]:
    """
    B5a — Protect bloqueó: nuestro Pokemon usó Protect Y el rival lo apuntó Y no perdimos HP.
    B5b — Hit Protect: usamos move ofensivo, target estaba en Protect (rival HP no cambió).
    Heurística: comparamos last_actions con cambios de HP.
    """
    if prev is None or last_actions is None:
        return 0.0, {}
    own_a, own_b = last_actions  # DecodedAction × 2

    out = 0.0
    debug = {}

    # B5a — propios usando Protect (move name lo decodificamos via species moves)
    for slot, dec in enumerate((own_a, own_b)):
        if not dec.is_move():
            continue
        own_p = curr.own_active[slot]
        if not own_p.species:
            continue
        # Si moves_revealed[move_idx] tiene "protect" en el nombre o coincide
        try:
            move_name = own_p.moves_revealed[dec.move_idx]
        except (IndexError, AttributeError):
            move_name = ""

        if move_name and "protect" in move_name.lower():
            # ¿algún rival le apuntó? se infiere por: rival usó move ofensivo + nuestro HP intacto
            prev_hp = prev.own_active[slot].hp_pct
            curr_hp = own_p.hp_pct
            blocked = max(0.0, 1.0 - (1.0 if prev_hp == 0 else curr_hp/prev_hp)) if prev_hp > 0 else 0.0
            # Si propio HP no cambió, asumimos que protect bloqueó algo apuntado
            if curr_hp >= prev_hp - 0.01:
                # Recompensa pequeña fija; daño bloqueado preciso requiere log parsing
                inc = cfg.w_protect_blocked_per_pct * 10.0   # ~+0.8 si bloqueó algo
                out += inc
                debug.setdefault("b.protect_blocked", 0.0)
                debug["b.protect_blocked"] += inc

    # B5b — atacamos rival que estaba en Protect (rival HP intacto pese a move ofensivo)
    for slot, dec in enumerate((own_a, own_b)):
        if not dec.is_move() or dec.target >= 0:
            continue   # no atacamos a un rival
        rival_idx = abs(dec.target) - 1  # -1 → 0, -2 → 1
        if rival_idx < 0 or rival_idx > 1:
            continue
        own_p = curr.own_active[slot]
        try:
            move_name = own_p.moves_revealed[dec.move_idx] if own_p.species else ""
        except (IndexError, AttributeError):
            move_name = ""
        if not move_name or "protect" in move_name.lower():
            continue  # nosotros usamos status/protect, no aplica
        # Rival HP no cambió pese a que apuntamos → o protect o miss
        prev_rhp = prev.rival_active[rival_idx].hp_pct if prev else 1.0
        curr_rhp = curr.rival_active[rival_idx].hp_pct
        if abs(prev_rhp - curr_rhp) < 0.005 and prev_rhp > 0:
            out += cfg.w_hit_protect
            debug["b.hit_protect"] = debug.get("b.hit_protect", 0.0) + cfg.w_hit_protect

    return out, debug


def _b_super_effective(curr, prev, last_actions, cfg, type_chart, moves_data) -> tuple[float, dict]:
    """B6 — bonus por hits SE (×2 → +1 step, ×4 → +2 steps)."""
    if prev is None or last_actions is None:
        return 0.0, {}
    own_a, own_b = last_actions

    out = 0.0
    debug = {}
    for slot, dec in enumerate((own_a, own_b)):
        if not dec.is_move() or dec.target >= 0:
            continue
        own_p = curr.own_active[slot]
        try:
            move_name = own_p.moves_revealed[dec.move_idx] if own_p.species else ""
        except (IndexError, AttributeError):
            continue
        move = moves_data.get(move_name)
        if not move:
            continue
        if move.get("category") == "status":
            continue
        rival_idx = abs(dec.target) - 1
        if not (0 <= rival_idx <= 1):
            continue
        rival_types = curr.rival_active[rival_idx].types
        if not rival_types:
            continue
        mult = get_effectiveness(move.get("type", ""), rival_types, type_chart)
        if mult <= 1.0:
            continue
        # 2× → 1 step, 4× → 2 steps
        steps = 1 if mult <= 2.0 else 2
        inc = steps * cfg.w_super_effective_per_step
        out += inc
        debug["b.super_effective"] = debug.get("b.super_effective", 0.0) + inc
    return out, debug


def _b_ko_outspeed(curr, prev, cfg) -> tuple[float, dict]:
    """B7 — KO al rival cuando él era más rápido (le ganamos sin permitirle moverse)."""
    if prev is None:
        return 0.0, {}
    out = 0.0
    debug = {}
    for i in range(2):
        prev_r = prev.rival_active[i]
        curr_r = curr.rival_active[i]
        if not prev_r.species or curr_r.species != prev_r.species:
            continue
        # Cayó este turno?
        cayo = curr_r.hp_pct == 0 and prev_r.hp_pct > 0
        if not cayo:
            continue
        # Quién era más rápido? Comparamos contra cualquier propio activo más lento
        for own in curr.own_active:
            if not own.species or own.fainted:
                continue
            if prev_r.speed_stat > own.speed_stat > 0:
                out += cfg.w_ko_outspeed
                debug["b.ko_outspeed"] = debug.get("b.ko_outspeed", 0.0) + cfg.w_ko_outspeed
                break
    return out, debug


def _b_status(curr, prev, cfg) -> tuple[float, dict]:
    """B8 — status infligido al rival (que no tenía antes)."""
    if prev is None:
        return 0.0, {}
    out = 0.0
    debug = {}
    for i in range(2):
        prev_r = prev.rival_active[i]
        curr_r = curr.rival_active[i]
        if not curr_r.species or curr_r.species != prev_r.species:
            continue
        if prev_r.status or not curr_r.status:
            continue
        s = curr_r.status
        # Pesos según tipo de status y rol del rival
        if s == "burn":
            inc = cfg.w_status_burn_phys if curr_r.is_physical_attacker else cfg.w_status_burn_other
        elif s == "paralysis":
            inc = cfg.w_status_paralysis
        elif s == "sleep":
            inc = cfg.w_status_sleep
        elif s == "freeze":
            inc = cfg.w_status_freeze
        elif s in ("poison", "toxic"):
            inc = cfg.w_status_poison_wall if curr_r.is_wall else cfg.w_status_poison_other
        else:
            continue
        out += inc
        debug[f"b.status_{s}"] = debug.get(f"b.status_{s}", 0.0) + inc
    return out, debug


def _b_focus_fire(curr, prev, last_actions, cfg) -> tuple[float, dict]:
    """B9 — ambos propios apuntan al mismo rival (+ extra si cae)."""
    if prev is None or last_actions is None:
        return 0.0, {}
    a, b = last_actions
    if not (a.is_move() and b.is_move()):
        return 0.0, {}
    if a.target >= 0 or b.target >= 0:
        return 0.0, {}
    if a.target != b.target:
        return 0.0, {}
    out = cfg.w_focus_fire_same_target
    debug = {"b.focus_fire": out}

    rival_idx = abs(a.target) - 1
    if 0 <= rival_idx <= 1:
        prev_r = prev.rival_active[rival_idx]
        curr_r = curr.rival_active[rival_idx]
        if (prev_r.species and prev_r.species == curr_r.species
                and prev_r.hp_pct > 0 and curr_r.hp_pct == 0):
            out += cfg.w_focus_fire_combo_ko
            debug["b.focus_fire_combo_ko"] = cfg.w_focus_fire_combo_ko
    return out, debug


def _b_spread(curr, prev, last_actions, cfg, moves_data) -> tuple[float, dict]:
    """B10 — spread move conectó en ambos rivales vivos."""
    if prev is None or last_actions is None:
        return 0.0, {}
    out = 0.0
    debug = {}
    for slot, dec in enumerate(last_actions):
        if not dec.is_move():
            continue
        own_p = curr.own_active[slot]
        try:
            mname = own_p.moves_revealed[dec.move_idx] if own_p.species else ""
        except (IndexError, AttributeError):
            continue
        move = moves_data.get(mname)
        if not move:
            continue
        target_kind = move.get("target", "")
        # spread targets en PokeAPI: "all-opponents", "all-other-pokemon"
        if "all-" not in target_kind and "opponent" not in target_kind:
            continue
        # ¿ambos rivales tenían HP > 0 antes y bajaron HP?
        hits = 0
        for i in range(2):
            prev_r = prev.rival_active[i]
            curr_r = curr.rival_active[i]
            if (prev_r.species and prev_r.species == curr_r.species
                    and prev_r.hp_pct > 0 and curr_r.hp_pct < prev_r.hp_pct):
                hits += 1
        if hits >= 2:
            out += cfg.w_spread_double_hit
            debug["b.spread_both_hit"] = debug.get("b.spread_both_hit", 0.0) + cfg.w_spread_double_hit
    return out, debug


def _b_pp_waste(curr, prev, last_actions, cfg) -> tuple[float, dict]:
    """B11 — penalidad chica por usar move con PP <20% restantes habiendo alternativas."""
    # poke-env tracking de PP es por slot. Sin acceso preciso al PP previo,
    # esta heurística queda como placeholder; la dejo no implementada para no
    # introducir señal ruidosa.
    return 0.0, {}


def _b_sash_survive(curr, prev, cfg) -> tuple[float, dict]:
    """B12 — sobrevivimos con HP exacto = 1 (Focus Sash / Sturdy / pin-point survive)."""
    if prev is None:
        return 0.0, {}
    out = 0.0
    debug = {}
    for i in range(2):
        prev_p = prev.own_active[i]
        curr_p = curr.own_active[i]
        if not curr_p.species or curr_p.species != prev_p.species:
            continue
        # Sobrevivimos con HP > 0 cuando el daño previsto era > el HP previo
        # Heurística simple: HP actual ≤ 1% pero no fainted Y prev_hp > 30%
        if 0 < curr_p.hp_pct <= 0.02 and prev_p.hp_pct >= 0.30:
            out += cfg.w_sash_survive
            debug["b.sash_survive"] = debug.get("b.sash_survive", 0.0) + cfg.w_sash_survive
    return out, debug


# ── Compute principal ────────────────────────────────────────────

def compute(
    curr: TurnSnapshot,
    prev: Optional[TurnSnapshot],
    last_actions,            # tuple[DecodedAction, DecodedAction] | None
    data:  dict,             # {"type_chart": ..., "moves": ...}
    cfg:   RewardConfig,
) -> tuple[float, dict[str, float]]:
    if not cfg.enable_layer_b:
        return 0.0, {}

    total = 0.0
    breakdown: dict[str, float] = {}

    # B1-B4 siempre
    for fn in (_b_damage, _b_ko):
        v, b = fn(curr, prev, cfg)
        total += v; breakdown.update(b)

    # B-extra
    if cfg.enable_b_extra:
        type_chart = data.get("type_chart", {})
        moves      = data.get("moves", {})
        v, b = _b_protect(curr, prev, last_actions, cfg, type_chart);            total += v; breakdown.update(b)
        v, b = _b_super_effective(curr, prev, last_actions, cfg, type_chart, moves); total += v; breakdown.update(b)
        v, b = _b_ko_outspeed(curr, prev, cfg);                                  total += v; breakdown.update(b)
        v, b = _b_status(curr, prev, cfg);                                       total += v; breakdown.update(b)
        v, b = _b_focus_fire(curr, prev, last_actions, cfg);                     total += v; breakdown.update(b)
        v, b = _b_spread(curr, prev, last_actions, cfg, moves);                  total += v; breakdown.update(b)
        v, b = _b_pp_waste(curr, prev, last_actions, cfg);                       total += v; breakdown.update(b)
        v, b = _b_sash_survive(curr, prev, cfg);                                 total += v; breakdown.update(b)

    return total, breakdown
