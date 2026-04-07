"""
train.py
─────────────────────────────────────────────────────────────────
Script principal de entrenamiento del VGC Bot.

Fases:
  1. Verificación del entorno (utils, damage_calc, state_encoder)
  2. Entrenamiento con RL usando PPO (stable-baselines3)
  3. Evaluación y logging del progreso
  4. Guardado de checkpoints del modelo

Uso:
    # Entrenar desde cero
    python train.py

    # Continuar desde checkpoint
    python train.py --resume checkpoints/model_10000.zip

    # Solo verificar que todo funcione
    python train.py --dry-run
"""

import argparse
import os
from pathlib import Path
from typing import Optional

# ── Verificación de dependencias ──────────────────────────────────

def check_dependencies():
    missing = []
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import gymnasium
    except ImportError:
        missing.append("gymnasium")
    try:
        import stable_baselines3
    except ImportError:
        missing.append("stable-baselines3")
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import poke_env
    except ImportError:
        missing.append("poke-env")

    if missing:
        print(f"\n⚠ Dependencias faltantes: {', '.join(missing)}")
        print("Instalá con: pip install " + " ".join(missing))
        return False
    return True


def dry_run():
    """
    Verifica que todos los módulos funcionen correctamente sin conectarse
    a Pokémon Showdown. Ideal para testear antes del entrenamiento real.
    """
    print("═" * 55)
    print("  VGC Bot — Dry Run (verificación de módulos)")
    print("═" * 55)

    # 1. utils
    print("\n[1/4] Verificando utils...")
    from src.utils import load_all_data, parse_team, calc_all_stats, get_pokemon

    team_path = Path(__file__).resolve().parent / "team.txt"
    data      = load_all_data()
    team      = parse_team(team_path)

    print(f"  ✓ {len(data['pokemon'])} Pokémon | {len(data['moves'])} movimientos | "
          f"{len(data['items'])} items")
    print(f"  ✓ Equipo: {[p['name'] for p in team]}")

    # Stats del equipo
    for p in team:
        stats = calc_all_stats(p, data["pokemon"])
        print(f"     {p['name']:25} HP={stats['hp']} | "
              f"SpA={stats['special-attack']} | Spe={stats['speed']}")

    # 2. damage_calc
    print("\n[2/4] Verificando damage_calc...")
    from src.damage_calc import calc_damage, calc_all_matchups, BattleConditions

    kyogre      = team[0]
    k_stats     = calc_all_stats(kyogre, data["pokemon"])
    k_poke_info = get_pokemon("kyogre", data["pokemon"])
    k_types     = k_poke_info["types"] if k_poke_info else ["water"]

    # Calyrex como defensor de prueba
    calyrex      = team[1]
    c_stats      = calc_all_stats(calyrex, data["pokemon"])
    c_poke_info  = get_pokemon("calyrex-shadow", data["pokemon"]) or \
                   get_pokemon("calyrex", data["pokemon"])
    c_types      = c_poke_info["types"] if c_poke_info else ["psychic", "ghost"]

    result = calc_damage(
        attacker_stats   = k_stats,
        attacker_types   = k_types,
        attacker_ability = "drizzle",
        attacker_item    = "mystic-water",
        attacker_name    = "kyogre",
        move             = data["moves"]["water-spout"],
        defender_stats   = c_stats,
        defender_types   = c_types,
        defender_ability = "",
        defender_item    = None,
        defender_name    = "calyrex-shadow",
        type_chart       = data["type_chart"],
        conditions       = BattleConditions(weather="rain"),
    )
    print(f"  ✓ {result}")

    # Todos los matchups de Kyogre
    defenders = []
    for p in team[1:]:
        stats = calc_all_stats(p, data["pokemon"])
        pinfo = get_pokemon(p["name"], data["pokemon"])
        types = pinfo["types"] if pinfo else []
        defenders.append((p["name"], stats, types))

    results = calc_all_matchups(
        kyogre, k_stats, k_types, defenders,
        data["moves"], data["type_chart"],
        BattleConditions(weather="rain"),
    )
    print(f"  ✓ {len(results)} matchups calculados para Kyogre")

    # 3. state_encoder
    print("\n[3/4] Verificando state_encoder...")
    from src.state_encoder import StateEncoder

    encoder = StateEncoder(data["type_chart"], data["moves"])
    obs = encoder.encode_manual(
        own_field = [
            {
                "hp_pct": 1.0, "types": k_types, "stats": k_stats,
                "stat_mods": {}, "status": None,
                "moves": kyogre["moves"], "tera_available": True,
                "tera_type": "grass", "item": "mystic-water",
            }
        ],
        rival_field = [
            {"hp_pct": 0.7, "types": ["fire"], "stats": {}, "stat_mods": {},
             "status": None, "moves": [], "tera_available": True,
             "tera_type": None, "item": None},
        ],
        benched_own = [
            {"hp_pct": 1.0, "types": ["psychic", "ghost"], "status": None},
        ],
        conditions = {"weather": "rain", "terrain": "grassy", "trick_room": False, "turn": 1},
    )
    print(f"  ✓ Observation vector: shape={obs.shape} | "
          f"min={obs.min():.3f} | max={obs.max():.3f}")

    # 4. vgc_env (sin conexión a Showdown)
    print("\n[4/4] Verificando vgc_env...")
    from src.vgc_env import VGCEnv
    env = VGCEnv(team_path=team_path)
    print(f"  ✓ action_space:      Discrete({env.action_space.n})")
    print(f"  ✓ observation_space: Box({env.observation_space.shape[0]},)")

    print("\n" + "═" * 55)
    print("  ✓ Todos los módulos funcionan correctamente.")
    print("  Listo para conectar con Pokémon Showdown y entrenar.")
    print("═" * 55)


def train(resume_path: Optional[str] = None):
    """
    Lanza el entrenamiento completo del agente con PPO.

    Conecta a Pokémon Showdown y entrena contra oponentes
    usando stable-baselines3.
    """
    if not check_dependencies():
        return

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback
    )
    import poke_env

    team_path   = Path(__file__).resolve().parent / "team.txt"
    checkpoints = Path(__file__).resolve().parent / "checkpoints"
    logs_dir    = Path(__file__).resolve().parent / "logs"
    checkpoints.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    print("═" * 55)
    print("  VGC Bot — Entrenamiento RL (PPO)")
    print("═" * 55)

    # ── Crear entorno ─────────────────────────────────────────────
    # server_configuration apunta a un servidor local de Showdown
    # Ver: https://github.com/hsahovic/poke-env para setup del servidor
    from poke_env.ps_client import AccountConfiguration, ServerConfiguration

    account = AccountConfiguration("vgc-bot", None)
    server  = ServerConfiguration(
        "localhost:8000",
        "https://play.pokemonshowdown.com/action.php"
    )

    env = VGCEnv(
        team_path   = team_path,
        account_configuration = account,
        server_configuration  = server,
    )

    # ── PPO config ────────────────────────────────────────────────
    LEARNING_RATE   = 3e-4
    N_STEPS         = 2048    # pasos por update
    BATCH_SIZE      = 64
    N_EPOCHS        = 10
    GAMMA           = 0.99    # discount factor
    TOTAL_TIMESTEPS = 1_000_000

    if resume_path:
        print(f"\nCargando modelo desde: {resume_path}")
        model = PPO.load(resume_path, env=env)
    else:
        model = PPO(
            policy          = "MlpPolicy",
            env             = env,
            learning_rate   = LEARNING_RATE,
            n_steps         = N_STEPS,
            batch_size      = BATCH_SIZE,
            n_epochs        = N_EPOCHS,
            gamma           = GAMMA,
            verbose         = 1,
            tensorboard_log = str(logs_dir),
        )

    print(f"\n  Parámetros del modelo: {model.policy}")
    print(f"  Total de timesteps:    {TOTAL_TIMESTEPS:,}")
    print(f"  Checkpoints en:        {checkpoints}/")

    # ── Callbacks ─────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = 10_000,
        save_path   = str(checkpoints),
        name_prefix = "vgc_ppo",
    )

    # ── Entrenamiento ─────────────────────────────────────────────
    print("\nIniciando entrenamiento...")
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = checkpoint_cb,
        progress_bar    = True,
    )

    # Guardar modelo final
    final_path = checkpoints / "vgc_ppo_final"
    model.save(str(final_path))
    print(f"\n✓ Modelo final guardado en: {final_path}.zip")


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGC Bot — Entrenamiento RL")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verificar módulos sin conectar a Showdown")
    parser.add_argument("--resume", type=str, default=None,
                        help="Ruta a checkpoint .zip para continuar entrenamiento")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        train(resume_path=args.resume)
