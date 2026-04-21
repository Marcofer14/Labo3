#!/usr/bin/env python3
"""
test_state_encoding.py
─────────────────────────────────────────────────────────────────
Visualiza cómo el state encoder transforma una batalla en un vector numérico.

Muestra:
  • Qué variables se encodearon
  • Dónde está cada variable en el vector
  • Valores normalizados (0-1)
  • Interpretación legible de cada parte

Uso:
    python scripts/test_state_encoding.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.utils import load_all_data, parse_team, calc_all_stats, get_pokemon
from src.state_encoder import StateEncoder


def print_section(title: str, width: int = 80):
    """Imprime un encabezado de sección."""
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}\n")


def visualize_encoding():
    """Visualiza el encoding del estado."""
    print_section("VGC BOT — VISUALIZACIÓN DEL STATE ENCODING")

    data      = load_all_data()
    team_path = Path(__file__).resolve().parent.parent / "team.txt"
    team      = parse_team(team_path)

    encoder = StateEncoder(data["type_chart"], data["moves"])

    # Simular un estado de batalla:
    # Turno 1: Kyogre vs Incineroar, Calyrex vs Rillaboom
    # Kyogre ha usado Setup, está en 75% de HP, sin rain aún

    kyogre      = team[0]
    k_stats     = calc_all_stats(kyogre, data["pokemon"])
    k_pinfo     = get_pokemon("kyogre", data["pokemon"])
    k_types     = k_pinfo["types"] if k_pinfo else ["water"]

    calyrex     = team[1]
    c_stats     = calc_all_stats(calyrex, data["pokemon"])
    c_pinfo     = get_pokemon("calyrex-shadow", data["pokemon"])
    c_types     = c_pinfo["types"] if c_pinfo else ["psychic", "ghost"]

    incineroar  = team[2]
    i_stats     = calc_all_stats(incineroar, data["pokemon"])
    i_pinfo     = get_pokemon("incineroar", data["pokemon"])
    i_types     = i_pinfo["types"] if i_pinfo else ["fire"]

    rillaboom   = team[3]
    r_stats     = calc_all_stats(rillaboom, data["pokemon"])
    r_pinfo     = get_pokemon("rillaboom", data["pokemon"])
    r_types     = r_pinfo["types"] if r_pinfo else ["grass"]

    # Crear el estado
    print_section("DESCRIPCIÓN DEL ESTADO")

    print("TURNO 1 — Situación inicial:")
    print(f"  • Propio: Kyogre (100% HP) + Calyrex (100% HP)")
    print(f"  • Rival: Incineroar (100% HP) + Rillaboom (100% HP)")
    print(f"  • Clima: Ninguno (Kyogre no ha hecho Drizzle aún)")
    print(f"  • Banqueados: Incineroar, Rillaboom (2 más)")

    obs = encoder.encode_manual(
        own_field = [
            {
                "hp_pct": 1.0,
                "types": k_types,
                "stats": {s: v / 400 for s, v in k_stats.items()},  # normalizar
                "stat_mods": {},
                "status": None,
                "moves": kyogre["moves"],
                "tera_available": True,
                "tera_type": "grass",
                "item": "mystic-water",
            },
            {
                "hp_pct": 1.0,
                "types": c_types,
                "stats": {s: v / 400 for s, v in c_stats.items()},
                "stat_mods": {},
                "status": None,
                "moves": calyrex["moves"],
                "tera_available": True,
                "tera_type": "dark",
                "item": "focus-sash",
            },
        ],
        rival_field = [
            {
                "hp_pct": 1.0,
                "types": i_types,
                "stats": {s: v / 400 for s, v in i_stats.items()},
                "stat_mods": {},
                "status": None,
                "moves": [],
                "tera_available": True,
                "tera_type": None,
                "item": None,
            },
            {
                "hp_pct": 1.0,
                "types": r_types,
                "stats": {s: v / 400 for s, v in r_stats.items()},
                "stat_mods": {},
                "status": None,
                "moves": [],
                "tera_available": True,
                "tera_type": None,
                "item": None,
            },
        ],
        benched_own = [
            {"hp_pct": 1.0, "types": i_types, "status": None},
            {"hp_pct": 1.0, "types": r_types, "status": None},
        ],
        conditions = {
            "weather": "none",
            "terrain": "none",
            "trick_room": False,
            "turn": 1,
        },
    )

    print_section("VECTOR RESULTANTE")

    print(f"Forma del vector: {obs.shape}")
    print(f"Tipo: numpy array float32")
    print(f"Rango de valores: [{obs.min():.4f}, {obs.max():.4f}]")
    print(f"Valores no-cero: {(obs != 0).sum()} / {len(obs)} ({(obs != 0).sum() / len(obs) * 100:.1f}%)")

    print_section("ESTRUCTURA DEL VECTOR")

    idx = 0

    # Pokémon propio 1 en campo
    print(f"\n[Índices 0–{idx + 120}] POKÉMON PROPIO 1 EN CAMPO (Kyogre)")
    print(f"  Subsección 0–0:    HP % = {obs[idx]:.4f}")
    idx += 1
    print(f"  Subsección 1–36:   Tipos (one-hot 2 slots × 18 tipos)")
    print(f"                      Slot 1 (Water): {obs[idx:idx+18]}")
    idx += 18
    print(f"                      Slot 2 (none):  {obs[idx:idx+18]}")
    idx += 18
    print(f"  Subsección 37–41:  Stats normalizados (Atk/Def/SpA/SpD/Spe)")
    print(f"                      {obs[idx:idx+5]}")
    idx += 5
    print(f"  Subsección 42–46:  Modificadores de stat (-6 a +6 → 0 a 1)")
    print(f"                      {obs[idx:idx+5]} (todos sin modificar = 0.5)")
    idx += 5
    print(f"  Subsección 47–53:  Estado alterado (one-hot 7 estados)")
    print(f"                      {obs[idx:idx+7]} (sin estado = último slot)")
    idx += 7
    print(f"  Subsecciones 54–153 (4 × 25): Movimientos")
    for move_i in range(4):
        print(f"    Move {move_i}: indices {idx + move_i*25}–{idx + (move_i+1)*25}")
    idx += 4 * 25
    print(f"  Subsección 154:    Tera disponible (1.0 = sí)")
    print(f"                      {obs[idx]:.4f}")
    idx += 1
    print(f"  Subsección 155–172: Tipo Tera (one-hot 18)")
    print(f"                      {obs[idx:idx+18]}")
    idx += 18
    print(f"  Subsección 173–197: Item (one-hot ~25 items comunes)")
    print(f"                      [Mystic Water slot = 1.0, otros = 0.0]")
    idx += 25

    print(f"\n[Índices {idx}–{idx + 120}] POKÉMON PROPIO 2 EN CAMPO (Calyrex)")
    print(f"  (estructura idéntica, {120} valores)")
    idx += 120

    print(f"\n[Índices {idx}–{idx + 90}] POKÉMON RIVAL 1 EN CAMPO (Incineroar)")
    print(f"  (estructura similar sin Tera + item, ~90 valores)")
    idx += 90

    print(f"\n[Índices {idx}–{idx + 90}] POKÉMON RIVAL 2 EN CAMPO (Rillaboom)")
    print(f"  (estructura similar, ~90 valores)")
    idx += 90

    print(f"\n[Índices {idx}–{idx + 16}] CONDICIONES DEL CAMPO (~16 valores)")
    print(f"  Clima (one-hot 5):            {obs[idx:idx+5]} (none)")
    idx += 5
    print(f"  Terreno (one-hot 5):          {obs[idx:idx+5]} (none)")
    idx += 5
    print(f"  Trick Room:                   {obs[idx]:.4f}")
    idx += 1
    print(f"  Turno normalizado:            {obs[idx]:.4f} (turno 1 / 50)")
    idx += 1

    print(f"\n[Índices {idx}–] POKÉMON BANQUEADOS PROPIOS (2 restantes)")
    print(f"  Total: ~40 valores")

    print_section("INTERPRETACIÓN")

    print("El vector completo (~380 valores) se interpreta así por la red neuronal:")
    print()
    print("  ┌─ ENTRADA A LA RED")
    print("  │  (vector de 380 floats, cada uno en [0, 1])")
    print("  │")
    print("  ├─ CAPAS OCULTAS")
    print("  │  (MLP típico: 256 → 128 → 64 unidades)")
    print("  │  Aprende patrones como:")
    print("  │    • 'Si rival Incineroar en 100% HP y yo Kyogre en 100%'")
    print("  │    • 'Usar Water Spout hace x2 contra Fire, tera disponible'")
    print("  │    • 'Cambiar a Kyogre counter a Incineroar es buena idea'")
    print("  │")
    print("  └─ OUTPUT")
    print("     (logits de 16 acciones)")
    print("     policy = softmax(logits) → acción con mayor probabilidad")
    print()

    print_section("VALORES REALES")

    print("Algunos valores interesantes del vector actual:")
    print()

    non_zero = np.argwhere(obs != 0).flatten()
    if len(non_zero) > 0:
        print(f"Primeros 20 índices no-cero:")
        for i in non_zero[:20]:
            print(f"  obs[{i:3d}] = {obs[i]:.4f}")
    print()


def main():
    visualize_encoding()


if __name__ == "__main__":
    main()
