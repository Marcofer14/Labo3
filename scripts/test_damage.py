#!/usr/bin/env python3
"""
test_damage.py
─────────────────────────────────────────────────────────────────
Script para probar y visualizar todos los cálculos de daño del equipo.

Imprime:
  • Stats calculados de cada Pokémon
  • Daño de cada movimiento contra cada rival
  • Efectividad de tipos
  • Si es OHKO, 2HKO, o necesita más golpes
  • Comparación con/sin condiciones favorables (rain para Kyogre)

Uso:
    python scripts/test_damage.py
    python scripts/test_damage.py --detailed    # + cálculos paso a paso
    python scripts/test_damage.py --vs "pyogre"  # solo vs rival específico
"""

import sys
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.utils import (
    load_all_data, parse_team, calc_all_stats, get_pokemon,
    summarize_team
)
from src.damage_calc import calc_damage, BattleConditions
from tabulate import tabulate  # pip install tabulate


def format_pct(pct: float, damage_result) -> str:
    """Formatea el daño % con emoji indicador."""
    pct_str = f"{pct*100:5.1f}%"
    if damage_result.ohko:
        return f"{pct_str} ✓ OHKO"
    elif damage_result.two_hit_ko:
        return f"{pct_str} ✓ 2HKO"
    else:
        hits = 100 / (pct * 100) if pct > 0 else float('inf')
        if hits < 100:
            return f"{pct_str} ({hits:.1f} hits)"
        else:
            return f"{pct_str} (no daño)"


def test_damage(detailed: bool = False, vs_only: str = None):
    """
    Prueba el damage calculator contra todos los matchups.
    """
    print("\n" + "="*80)
    print("  VGC BOT — TEST DE DAÑO")
    print("="*80)

    data      = load_all_data()
    team_path = Path(__file__).resolve().parent.parent / "team.txt"
    team      = parse_team(team_path)

    # Calcular stats de todos
    team_stats = []
    for p in team:
        stats = calc_all_stats(p, data["pokemon"])
        pinfo = get_pokemon(p["name"], data["pokemon"])
        types = pinfo["types"] if pinfo else []
        team_stats.append((p, stats, types))

    # Mostrar resumen del equipo
    print()
    summarize_team(team, data["pokemon"])

    # Iterar sobre cada Pokémon atacante
    for attacker_idx, (attacker, att_stats, att_types) in enumerate(team_stats):
        print(f"\n{'='*80}")
        print(f"  ATACANTE: {attacker['name'].upper()}")
        print(f"  Movimientos: {', '.join(attacker['moves'])}")
        print(f"{'='*80}")

        # Iterar sobre cada movimiento
        for move_name in attacker["moves"]:
            move = data["moves"].get(move_name)
            if not move or move.get("category") == "status":
                if move and move.get("category") == "status":
                    print(f"\n  [{move_name.upper()}] Status move (no calcula daño)")
                continue

            print(f"\n  ┌─ {move_name.upper()}")
            print(f"  │  Tipo: {move['type']:8} | Potencia: {move.get('power', '—'):3} | "
                  f"Cat: {move['category']}")

            # Tabla de daño contra cada rival
            headers = [
                "Rival",
                "Tipo",
                "Sin condiciones",
                "Con rain (óptimo)",
                "Efectividad",
            ]
            rows = []

            for def_idx, (defender, def_stats, def_types) in enumerate(team_stats):
                if def_idx == attacker_idx:
                    continue  # No calcular contra uno mismo

                if vs_only and vs_only not in defender["name"]:
                    continue  # Filtrar si se especificó --vs

                # Cálculo sin condiciones especiales
                result_normal = calc_damage(
                    attacker_stats   = att_stats,
                    attacker_types   = att_types,
                    attacker_ability = attacker.get("ability", ""),
                    attacker_item    = attacker.get("item"),
                    attacker_name    = attacker["name"],
                    move             = move,
                    defender_stats   = def_stats,
                    defender_types   = def_types,
                    defender_ability = "",
                    defender_item    = None,
                    defender_name    = defender["name"],
                    type_chart       = data["type_chart"],
                    conditions       = BattleConditions(weather="none"),
                )

                # Con condiciones favorables (rain para agua, sun para fuego, etc.)
                weather = "none"
                if move["type"] == "water":
                    weather = "rain"
                elif move["type"] == "fire":
                    weather = "sun"

                result_favorable = calc_damage(
                    attacker_stats   = att_stats,
                    attacker_types   = att_types,
                    attacker_ability = attacker.get("ability", ""),
                    attacker_item    = attacker.get("item"),
                    attacker_name    = attacker["name"],
                    move             = move,
                    defender_stats   = def_stats,
                    defender_types   = def_types,
                    defender_ability = "",
                    defender_item    = None,
                    defender_name    = defender["name"],
                    type_chart       = data["type_chart"],
                    conditions       = BattleConditions(weather=weather),
                )

                eff_str = f"x{result_favorable.effectiveness}"
                if result_favorable.effectiveness == 0:
                    eff_str += " (inmune)"
                elif result_favorable.effectiveness == 0.25:
                    eff_str += " (muy débil)"
                elif result_favorable.effectiveness == 0.5:
                    eff_str += " (débil)"
                elif result_favorable.effectiveness == 2.0:
                    eff_str += " (SE)"
                elif result_favorable.effectiveness == 4.0:
                    eff_str += " (muy SE)"

                rows.append([
                    defender["name"],
                    "/".join(def_types),
                    format_pct(result_normal.avg_pct, result_normal),
                    format_pct(result_favorable.avg_pct, result_favorable),
                    eff_str,
                ])

            if rows:
                table = tabulate(rows, headers=headers, tablefmt="plain")
                for line in table.split("\n"):
                    print(f"  │  {line}")
            else:
                print(f"  │  (sin rivales para mostrar)")

            print(f"  └")

        print()


def main():
    parser = argparse.ArgumentParser(description="Prueba el damage calculator del VGC Bot")
    parser.add_argument("--detailed", action="store_true",
                        help="Mostrar cálculos paso a paso")
    parser.add_argument("--vs", type=str, default=None,
                        help="Filtrar: solo mostrar daño vs este Pokémon")
    args = parser.parse_args()

    test_damage(detailed=args.detailed, vs_only=args.vs)


if __name__ == "__main__":
    main()
