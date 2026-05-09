"""
prepare.py
─────────────────────────────────────────────────────────────────
Pre-flight check antes de empezar a entrenar.

Verifica:
  1. Dependencias instaladas
  2. Estructura de carpetas (data/raw, rivalteams, checkpoints, logs, reports)
  3. Datos de PokeAPI cargables
  4. team.txt parseable + todos los Pokémon en el dataset
  5. 10 equipos rivales válidos
  6. Resolución del formato Reg I
  7. Importación correcta del paquete src.training
  8. Conectividad al servidor Showdown (con timeout corto)
  9. PyTorch disponible y device (CPU/GPU)

Uso:
  python prepare.py
  python prepare.py --skip-server         # no checkear Showdown
  python prepare.py --server my-host:8000
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path


GREEN = "\033[92m"
RED   = "\033[91m"
YEL   = "\033[93m"
BLUE  = "\033[94m"
END   = "\033[0m"


def ok(msg):    print(f"  {GREEN}✓{END} {msg}")
def fail(msg):  print(f"  {RED}✗{END} {msg}")
def warn(msg):  print(f"  {YEL}!{END} {msg}")
def info(msg):  print(f"  {BLUE}·{END} {msg}")


def check_python():
    print("\n[1/9] Python")
    info(f"versión: {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        fail("se requiere Python 3.9+"); return False
    ok("versión OK")
    return True


def check_deps():
    print("\n[2/9] Dependencias")
    deps = [
        "numpy", "gymnasium", "stable_baselines3", "sb3_contrib",
        "torch", "poke_env", "matplotlib", "rich", "tensorboard",
    ]
    missing = []
    for d in deps:
        try:
            mod = __import__(d)
            v = getattr(mod, "__version__", "?")
            info(f"{d:25s} {v}")
        except ImportError:
            missing.append(d)
    if missing:
        fail(f"faltan: {missing}")
        info("→ pip install " + " ".join(missing))
        return False
    ok("todas presentes")
    return True


def check_folders():
    print("\n[3/9] Estructura de carpetas")
    base = Path(__file__).resolve().parent
    needed = ["data/raw", "rivalteams", "src", "src/rewards", "src/training"]
    for f in needed:
        if (base / f).exists():
            ok(f"{f}/")
        else:
            fail(f"falta: {f}/")
            return False

    # crear si no existen
    for f in ["checkpoints", "checkpoints/league", "logs", "reports"]:
        (base / f).mkdir(parents=True, exist_ok=True)
        info(f"asegurado: {f}/")
    return True


def check_data():
    print("\n[4/9] Datos PokeAPI")
    try:
        from src.utils import load_all_data, DATA_DIR
        info(f"DATA_DIR: {DATA_DIR}")
        data = load_all_data()
        ok(f"{len(data['pokemon']):,} Pokemon")
        ok(f"{len(data['moves']):,} moves")
        ok(f"{len(data['items']):,} items")
        ok(f"{len(data['abilities']):,} abilities")
        ok(f"type chart: {len(data['type_chart'])} tipos")
        return True
    except Exception as ex:
        fail(f"{ex}")
        return False


def check_team():
    print("\n[5/9] team.txt (equipo del bot)")
    try:
        from src.utils import load_all_data, parse_team, calc_all_stats, get_pokemon
        base = Path(__file__).resolve().parent
        team = parse_team(base / "team.txt")
        data = load_all_data()
        if len(team) != 6:
            warn(f"el equipo tiene {len(team)} Pokemon (lo usual son 6)")
        for p in team:
            poke = get_pokemon(p["name"], data["pokemon"])
            if poke is None:
                fail(f"no encontrado: {p['name']}"); return False
            stats = calc_all_stats(p, data["pokemon"])
            tera = p.get("tera_type", "?")
            info(f"{p['name']:25s} HP={stats['hp']} SpA={stats['special-attack']} Spe={stats['speed']}  Tera={tera}")
        ok("equipo válido")
        return True
    except Exception as ex:
        fail(f"{ex}")
        return False


def check_rivals():
    print("\n[6/9] rivalteams (pool de oponentes)")
    try:
        from src.rival_teams import load_rival_pool
        base = Path(__file__).resolve().parent
        pool = load_rival_pool(base / "rivalteams")
        info(f"{pool.num_teams} equipos cargados:")
        for i in range(pool.num_teams):
            first = pool._raw[i].splitlines()[0]
            info(f"  [{i}] {first}")
        if pool.num_teams < 2:
            warn("muy pocos equipos en el pool")
        ok("rival pool OK")
        return True
    except Exception as ex:
        fail(f"{ex}")
        return False


def check_format():
    print("\n[7/9] Formato de batalla")
    try:
        from src.format_resolver import resolve_format
        fmt = resolve_format(None)
        ok(f"resuelto: {fmt}")
        return True
    except Exception as ex:
        fail(f"{ex}")
        return False


def check_training_package():
    print("\n[8/9] src.training package")
    try:
        from src.training import (
            TrainingConfig, CurriculumScheduler, SelfPlayLeague,
            LeagueOpponent, run_tournament, generate_final_report,
            LossPlateauCallback, RewardBreakdownCallback,
            ActivationStatsCallback, WinRateCallback, PhaseLogCallback,
            SnapshotLeagueCallback, LeagueResultCallback,
        )
        ok("imports correctos")
        tcfg = TrainingConfig()
        info(f"algorithm={tcfg.algorithm} arch={tcfg.net_arch} act={tcfg.activation}")
        info(f"buffer={tcfg.rollout_buffer_size():,} (n_envs×n_steps)")
        info(f"plateau win={tcfg.plateau_window} eps={tcfg.plateau_eps}")
        return True
    except Exception as ex:
        fail(f"{ex}")
        return False


def check_server(host: str | None, skip: bool):
    print("\n[9/9] Servidor Showdown")
    if skip:
        warn("saltado (--skip-server)")
        return True
    host = host or os.environ.get("SHOWDOWN_SERVER", "localhost:8000")
    try:
        if ":" in host:
            h, p = host.split(":", 1)
            p = int(p)
        else:
            h, p = host, 8000
        info(f"probando conexión a {h}:{p}...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect((h, p))
        s.close()
        ok(f"servidor accesible en {h}:{p}")
        return True
    except Exception as ex:
        warn(f"no se pudo conectar: {ex}")
        info("levantar con: docker run -d -p 8000:8000 smogon/pokemon-showdown")
        info("o desde docker-compose.yml en el repo")
        return False  # no bloqueante para el train


def check_torch_device():
    print("\n[bonus] PyTorch / device")
    try:
        import torch
        info(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            ok(f"GPU disponible: {torch.cuda.get_device_name(0)}")
            ok(f"CUDA: {torch.version.cuda}")
        else:
            warn("GPU no disponible — entrenará en CPU (lento pero OK)")
        return True
    except Exception as ex:
        fail(f"{ex}")
        return False


# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VGC Bot - pre-flight check")
    parser.add_argument("--server", type=str, default=None)
    parser.add_argument("--skip-server", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  VGC Bot - Pre-flight check")
    print("=" * 60)

    results = {
        "python":   check_python(),
        "deps":     check_deps(),
        "folders":  check_folders(),
        "data":     check_data(),
        "team":     check_team(),
        "rivals":   check_rivals(),
        "format":   check_format(),
        "training": check_training_package(),
        "server":   check_server(args.server, args.skip_server),
        "torch":    check_torch_device(),
    }

    print("\n" + "=" * 60)
    failed = [k for k, v in results.items() if not v and k != "server"]
    if failed:
        print(f"  {RED}✗ Pre-flight FALLÓ:{END} {failed}")
        sys.exit(1)
    if not results["server"]:
        print(f"  {YEL}! Pre-flight OK pero el servidor Showdown no respondió.{END}")
        print(f"     El entrenamiento real lo necesita; asegurate de levantarlo.")
    else:
        print(f"  {GREEN}✓ Todo listo para entrenar.{END}")
    print(f"\n  Próximo paso: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
