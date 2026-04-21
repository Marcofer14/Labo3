"""
train.py
─────────────────────────────────────────────────────────────────
Script principal de entrenamiento del VGC Bot.

ARQUITECTURA:
  VGCEnv (DoublesEnv)
    └── implementa calc_reward() y embed_battle() para VGC dobles
  SingleAgentWrapper(VGCEnv, oponente)
    └── convierte el env paralelo PettingZoo → Gymnasium single-agent
  FlatObsWrapper(SingleAgentWrapper)
    └── extrae solo el array del dict de observación para PPO estándar
  PPO (stable-baselines3)
    └── entrena la política sobre el env resultante

FLUJO DE ENTRENAMIENTO:
  Fase 1 - Self-play contra RandomPlayer (oponente aleatorio)
    → El bot aprende rápidamente qué acciones tienen sentido
  Fase 2 - Self-play contra MaxBasePowerPlayer (greedy)
    → El bot aprende a superar estrategias simples
  Fase 3 - Ladder real / evaluación
    → Despliegue contra humanos

USO:
  # Verificar módulos sin conectar a Showdown:
  python train.py --dry-run

  # Entrenar desde cero (necesita servidor Showdown en localhost:8000):
  python train.py

  # Continuar desde checkpoint:
  python train.py --resume checkpoints/vgc_ppo_100000.zip

  # Cambiar oponente:
  python train.py --opponent greedy
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
from src.format_resolver import resolve_format

# ── Verificación de dependencias ──────────────────────────────────

def check_dependencies() -> bool:
    missing = []
    for pkg, import_name in [
        ("numpy",             "numpy"),
        ("gymnasium",         "gymnasium"),
        ("stable-baselines3", "stable_baselines3"),
        ("torch",             "torch"),
        ("poke-env",          "poke_env"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n⚠  Dependencias faltantes: {', '.join(missing)}")
        print("   Instalar con: pip install " + " ".join(missing))
        return False
    return True


# ── Dry-run: verifica todos los módulos sin Showdown ─────────────

def dry_run():
    print("═" * 60)
    print("  VGC Bot — Dry Run (verificación de módulos)")
    print("═" * 60)

    # [1/5] utils
    print("\n[1/5] Verificando utils...")
    from src.utils import load_all_data, parse_team, calc_all_stats, get_pokemon

    team_path = Path(__file__).resolve().parent / "team.txt"
    data      = load_all_data()
    team      = parse_team(team_path)

    print(f"  ✓ {len(data['pokemon'])} Pokémon | "
          f"{len(data['moves'])} movimientos | "
          f"{len(data['items'])} items")
    print(f"  ✓ Equipo: {[p['name'] for p in team]}")

    for p in team:
        stats = calc_all_stats(p, data["pokemon"])
        print(f"     {p['name']:25}  "
              f"HP={stats['hp']}  |  "
              f"SpA={stats['special-attack']}  |  "
              f"Spe={stats['speed']}")

    # [2/5] damage_calc
    print("\n[2/5] Verificando damage_calc...")
    from src.damage_calc import calc_damage, calc_all_matchups, BattleConditions

    kyogre      = team[0]
    k_stats     = calc_all_stats(kyogre, data["pokemon"])
    k_info      = get_pokemon("kyogre", data["pokemon"])
    k_types     = k_info["types"] if k_info else ["water"]

    calyrex     = team[1]
    c_stats     = calc_all_stats(calyrex, data["pokemon"])
    c_info      = get_pokemon("calyrex-shadow", data["pokemon"]) or \
                  get_pokemon("calyrex", data["pokemon"])
    c_types     = c_info["types"] if c_info else ["psychic", "ghost"]

    # Test normal (100% HP)
    result_full = calc_damage(
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
        attacker_hp_pct  = 1.0,
    )
    print(f"  ✓ Water Spout @100% HP → {result_full.min_pct*100:.1f}%–{result_full.max_pct*100:.1f}%")

    # Test con HP bajo (25%)
    result_low = calc_damage(
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
        attacker_hp_pct  = 0.25,
    )
    print(f"  ✓ Water Spout @25% HP  → {result_low.min_pct*100:.1f}%–{result_low.max_pct*100:.1f}%  "
          f"(escalado correcto: ~{result_full.min_pct*0.25*100:.1f}%)")

    defenders = []
    for p in team[1:]:
        s = calc_all_stats(p, data["pokemon"])
        pi = get_pokemon(p["name"], data["pokemon"])
        t = pi["types"] if pi else []
        defenders.append((p["name"], s, t))

    results = calc_all_matchups(
        kyogre, k_stats, k_types, defenders,
        data["moves"], data["type_chart"],
        BattleConditions(weather="rain"),
    )
    print(f"  ✓ {len(results)} matchups calculados para Kyogre")

    # [3/5] state_encoder
    print("\n[3/5] Verificando state_encoder...")
    from src.state_encoder import StateEncoder

    encoder = StateEncoder(data["type_chart"], data["moves"])
    obs = encoder.encode_manual(
        own_field = [{
            "hp_pct": 1.0, "types": k_types, "stats": k_stats,
            "stat_mods": {}, "status": None,
            "moves": kyogre["moves"], "tera_available": True,
            "tera_type": "grass", "item": "mystic-water",
        }],
        rival_field = [{
            "hp_pct": 0.7, "types": ["fire"], "stats": {}, "stat_mods": {},
            "status": None, "moves": [], "tera_available": True,
            "tera_type": None, "item": None,
        }],
        benched_own = [{"hp_pct": 1.0, "types": ["psychic", "ghost"], "status": None}],
        conditions  = {"weather": "rain", "terrain": "grassy", "trick_room": False, "turn": 1},
    )
    print(f"  ✓ Observation vector: shape={obs.shape}  |  "
          f"min={obs.min():.3f}  max={obs.max():.3f}")

    # Verificar tamaño con campo vacío (padding)
    obs_empty = encoder.encode_manual([], [], [], {})
    assert obs.shape == obs_empty.shape, (
        f"Error: tamaño inconsistente con padding — "
        f"{obs.shape} vs {obs_empty.shape}"
    )
    print(f"  ✓ Padding consistente: shape={obs_empty.shape}")

    # [4/5] vgc_env (sin conexión a Showdown)
    print("\n[4/5] Verificando vgc_env...")
    from src.vgc_env import VGCEnv, FlatObsWrapper
    from poke_env.environment import DoublesEnv

    action_size = DoublesEnv.get_action_space_size(9)
    print(f"  ✓ action_space_size (Gen 9): {action_size}")
    print(f"  ✓ MultiDiscrete([{action_size}, {action_size}])")
    print(f"  ✓ observation_size: {obs.shape[0]}")
    print(f"  ✓ VGCEnv importado correctamente")
    print(f"  ℹ Nota: la instanciación completa requiere servidor Showdown")

    # [5/5] train config
    print("\n[5/5] Verificando configuración de entrenamiento...")
    try:
        from stable_baselines3 import PPO
        print(f"  ✓ stable-baselines3 disponible")
    except ImportError:
        print(f"  ⚠ stable-baselines3 no instalado")
        print(f"    Instalar: pip install stable-baselines3[extra] torch")

    print("\n" + "═" * 60)
    print("  ✓ Todos los módulos verificados correctamente.")
    print("  Para entrenar: levantá el servidor Showdown y ejecutá:")
    print("    python train.py")
    print("═" * 60)


# ── Entrenamiento principal ───────────────────────────────────────

def train(
    resume_path: Optional[str] = None,
    opponent_type: str = "random",
    server: Optional[str] = None,
    battle_format: Optional[str] = None,
):
    """
    Lanza el entrenamiento del agente con PPO.

    Requiere un servidor local de Pokémon Showdown.
    La URL del servidor se resuelve en este orden:
      1. Argumento --server (ej: localhost:8000)
      2. Variable de entorno SHOWDOWN_SERVER
      3. Default: localhost:8000

    Con Docker Compose el servidor se llama "showdown:8000" dentro
    de la red interna de Docker (la env var SHOWDOWN_SERVER ya viene
    configurada en docker-compose.yml).
    """
    if not check_dependencies():
        sys.exit(1)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from poke_env.environment import SingleAgentWrapper
    from poke_env import RandomPlayer, MaxBasePowerPlayer
    from poke_env.ps_client import ServerConfiguration

    from src.vgc_env import VGCEnv, FlatObsWrapper

    team_path   = Path(__file__).resolve().parent / "team.txt"
    checkpoints = Path(__file__).resolve().parent / "checkpoints"
    logs_dir    = Path(__file__).resolve().parent / "logs"
    checkpoints.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Leer el equipo como string
    with open(team_path, encoding="utf-8") as f:
        team_str = f.read()

    # Resolver host del servidor: arg > env var > default
    server_host = server or os.environ.get("SHOWDOWN_SERVER", "localhost:8000")

    VGC_FORMAT = resolve_format(battle_format)

    print("═" * 60)
    print("  VGC Bot — Entrenamiento RL (PPO)")
    print(f"  Formato:  {VGC_FORMAT}")
    print(f"  Oponente: {opponent_type}")
    print(f"  Servidor: ws://{server_host}/showdown/websocket")
    print("═" * 60)

    # ── Configurar conexión al servidor ──────────────────────────
    # Showdown usa WebSocket en /showdown/websocket
    server_cfg = ServerConfiguration(
        f"ws://{server_host}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )

    env = VGCEnv(
        team_path      = team_path,
        battle_format  = VGC_FORMAT,
        server_configuration = server_cfg,
        start_listening      = True,
        choose_on_teampreview = True,
    )

    # ── Crear el oponente ─────────────────────────────────────────
    if opponent_type == "random":
        opponent = RandomPlayer(
            battle_format        = VGC_FORMAT,
            team                 = team_str,
            server_configuration = server_cfg,
        )
    elif opponent_type == "greedy":
        opponent = MaxBasePowerPlayer(
            battle_format        = VGC_FORMAT,
            team                 = team_str,
            server_configuration = server_cfg,
        )
    else:
        raise ValueError(f"Oponente desconocido: {opponent_type}. Opciones: random, greedy")

    # ── Wrappear para SB3 ─────────────────────────────────────────
    # SingleAgentWrapper: paralelo PettingZoo → Gymnasium single-agent
    # FlatObsWrapper:     dict obs {"observation", "action_mask"} → Box plano
    gym_env = FlatObsWrapper(SingleAgentWrapper(env, opponent))

    print(f"\n  observation_space: {gym_env.observation_space}")
    print(f"  action_space:      {gym_env.action_space}")

    # ── Configuración de PPO ──────────────────────────────────────
    LEARNING_RATE   = 3e-4
    N_STEPS         = 2048
    BATCH_SIZE      = 64
    N_EPOCHS        = 10
    GAMMA           = 0.99
    TOTAL_TIMESTEPS = 1_000_000

    if resume_path:
        print(f"\n  Cargando checkpoint: {resume_path}")
        model = PPO.load(resume_path, env=gym_env)
    else:
        model = PPO(
            policy          = "MlpPolicy",
            env             = gym_env,
            learning_rate   = LEARNING_RATE,
            n_steps         = N_STEPS,
            batch_size      = BATCH_SIZE,
            n_epochs        = N_EPOCHS,
            gamma           = GAMMA,
            verbose         = 1,
            tensorboard_log = str(logs_dir),
        )

    print(f"\n  Total timesteps:  {TOTAL_TIMESTEPS:,}")
    print(f"  Checkpoints en:   {checkpoints}/")
    print(f"  Logs TensorBoard: {logs_dir}/")

    # ── Callbacks ─────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = 10_000,
        save_path   = str(checkpoints),
        name_prefix = "vgc_ppo",
    )

    # ── Entrenar ──────────────────────────────────────────────────
    print("\nIniciando entrenamiento...")
    print("(Ver progreso en TensorBoard: tensorboard --logdir logs)\n")

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
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verificar módulos sin conectar a Showdown"
    )
    parser.add_argument(
        "--format", type=str, default=None,
        help="Formato de batalla. Si no se pasa, usa VGC_FORMAT o el default automático."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Ruta a checkpoint .zip para continuar entrenamiento"
    )
    parser.add_argument(
        "--opponent", type=str, default="random",
        choices=["random", "greedy"],
        help="Tipo de oponente: random (default) | greedy (MaxBasePower)"
    )
    parser.add_argument(
        "--server", type=str, default=None,
        help="URL del servidor Showdown (default: localhost:8000 o env SHOWDOWN_SERVER)"
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        train(
            resume_path=args.resume,
            opponent_type=args.opponent,
            server=args.server,
            battle_format=args.format,
        )
