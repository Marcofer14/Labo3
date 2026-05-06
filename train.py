"""
train.py
─────────────────────────────────────────────────────────────────
Entrenamiento del VGC Bot con curriculum learning + self-play league.

Pipeline:
  · RecurrentPPO (sb3-contrib) con LSTM → recuerda moves del rival y Tera
  · Hidden [256] · ReLU · LSTM 128
  · 4 envs paralelos × n_steps=1024 → buffer 4096, batch 256, 4 epochs
  · LR schedule lineal 3e-4 → 1e-5
  · Curriculum por plateau de loss (no por timesteps fijos)
  · Layer C ramp-up gradual al entrar a stage 3
  · Snapshots automáticos al league desde stage 3 con eviction por win-rate
  · Stage 4 transición: rebuild de envs con LeagueOpponent
  · Round-robin entre miembros del league al cierre (opcional)
  · 10 equipos rivales rotando + métricas detalladas + reporte HTML

Uso:
  python train.py --dry-run
  python train.py
  python train.py --algorithm maskable_ppo
  python train.py --resume checkpoints/vgc_t500000.zip
  python train.py --no-tournament      # saltar round-robin final
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Callable


def check_dependencies() -> bool:
    missing = []
    deps = [
        ("numpy",             "numpy"),
        ("gymnasium",         "gymnasium"),
        ("stable-baselines3", "stable_baselines3"),
        ("sb3-contrib",       "sb3_contrib"),
        ("torch",             "torch"),
        ("poke-env",          "poke_env"),
        ("matplotlib",        "matplotlib"),
    ]
    for pkg, mod in deps:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n[!] Dependencias faltantes: {', '.join(missing)}")
        print("   Instalar: pip install " + " ".join(missing))
        return False
    return True


# ─────────────────────────────────────────────────────────────────
# DRY RUN
# ─────────────────────────────────────────────────────────────────

def dry_run():
    print("=" * 60)
    print("  VGC Bot - Dry Run (verificación de módulos)")
    print("=" * 60)

    from src.utils import load_all_data, parse_team, calc_all_stats
    from src.rewards import RewardConfig
    from src.rewards.action_decoder import decode_action
    from src.rival_teams import load_rival_pool
    from src.training import (
        TrainingConfig, CurriculumScheduler, SelfPlayLeague,
        LeagueOpponent, action_int_to_order, run_tournament,
    )

    team_path = Path(__file__).resolve().parent / "team.txt"
    data = load_all_data()
    team = parse_team(team_path)
    print(f"\n[1/8] utils + team")
    print(f"  ✓ {len(data['pokemon'])} Pokemon | {len(data['moves'])} moves")
    print(f"  ✓ Equipo: {[p['name'] for p in team]}")

    print(f"\n[2/8] reward framework")
    for s in (1, 2, 3, 4, 5):
        cfg = getattr(RewardConfig, f'stage_{s}')()
        flags = [k for k in ('enable_layer_a', 'enable_layer_b', 'enable_b_extra',
                             'enable_layer_c', 'enable_layer_d') if getattr(cfg, k)]
        print(f"  ✓ stage {s}: {flags}")

    print(f"\n[3/8] action decoder + action-to-order helper")
    for a in [0, 1, 7, 11, 87, 106]:
        print(f"  {a:3d} → {decode_action(a)}")
    print(f"  ✓ action_int_to_order importado")

    print(f"\n[4/8] rivalteams pool")
    pool = load_rival_pool(Path(__file__).resolve().parent / "rivalteams")
    print(f"  ✓ {pool.num_teams} equipos cargados")

    print(f"\n[5/8] training config")
    tcfg = TrainingConfig()
    print(f"  ✓ algo={tcfg.algorithm}  arch={tcfg.net_arch}  act={tcfg.activation}")
    print(f"  ✓ envs={tcfg.n_envs} steps={tcfg.n_steps} batch={tcfg.batch_size} epochs={tcfg.n_epochs}")
    print(f"  ✓ buffer={tcfg.rollout_buffer_size():,} ({tcfg.minibatches_per_epoch()} mb/epoch)")
    print(f"  ✓ LSTM h={tcfg.lstm_hidden_size} layers={tcfg.lstm_layers}")
    print(f"  ✓ plateau win={tcfg.plateau_window} eps={tcfg.plateau_eps}")

    print(f"\n[6/8] curriculum scheduler")
    sched = CurriculumScheduler(tcfg)
    sched.start()
    print(f"  ✓ scheduler iniciado, stage={sched.current_stage}")

    print(f"\n[7/8] self-play league")
    league = SelfPlayLeague(max_size=tcfg.league_max_size,
                            eviction_kind=tcfg.league_eviction)
    print(f"  ✓ league: max={league.max_size} eviction={league.eviction_kind}")

    print(f"\n[8/8] LeagueOpponent + tournament")
    print(f"  ✓ LeagueOpponent importable")
    print(f"  ✓ run_tournament importable")

    print("\n" + "=" * 60)
    print("  ✓ Todos los módulos verificados.")
    print("  Próximo paso: python prepare.py  (pre-flight)")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────

def train(
    resume_path:     Optional[str]   = None,
    server:          Optional[str]   = None,
    battle_format:   Optional[str]   = None,
    algorithm:       Optional[str]   = None,
    n_envs:          Optional[int]   = None,
    total_timesteps: Optional[int]   = None,
    skip_tournament: bool            = False,
):
    if not check_dependencies():
        sys.exit(1)

    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env  import DummyVecEnv
    from poke_env.environment              import SingleAgentWrapper, DoublesEnv
    from poke_env                          import RandomPlayer, MaxBasePowerPlayer
    from poke_env.ps_client                import ServerConfiguration

    from src.format_resolver import resolve_format
    from src.vgc_env         import VGCEnv, FlatObsWrapper
    from src.rival_teams     import load_rival_pool
    from src.training        import (
        TrainingConfig,
        CurriculumScheduler,
        SelfPlayLeague,
        LossPlateauCallback,
        RewardBreakdownCallback,
        ActivationStatsCallback,
        WinRateCallback,
        PhaseLogCallback,
        SnapshotLeagueCallback,
        LeagueResultCallback,
        LeagueOpponent,
        run_tournament,
        generate_final_report,
    )
    from src.training.policy import (
        ActivationRecorder,
        build_policy_kwargs,
        import_recurrent_ppo,
        import_maskable_ppo,
    )

    # ── Config ───────────────────────────────────────────────────
    tcfg = TrainingConfig()
    if algorithm:        tcfg.algorithm       = algorithm
    if n_envs:           tcfg.n_envs          = n_envs
    if total_timesteps:  tcfg.total_timesteps = total_timesteps

    base_dir   = Path(__file__).resolve().parent
    team_path  = base_dir / "team.txt"
    rivals_dir = base_dir / tcfg.rivalteams_dir
    ckpt_dir   = base_dir / tcfg.checkpoint_dir
    log_dir    = base_dir / tcfg.log_dir
    league_dir = ckpt_dir / "league"
    for d in (ckpt_dir, log_dir, league_dir):
        d.mkdir(parents=True, exist_ok=True)

    server_host = server or os.environ.get("SHOWDOWN_SERVER", "localhost:8000")
    fmt = resolve_format(battle_format)

    print("=" * 60)
    print(f"  VGC Bot - Curriculum + Self-Play League")
    print(f"  Algoritmo: {tcfg.algorithm}")
    print(f"  Formato:   {fmt}")
    print(f"  Servidor:  ws://{server_host}/showdown/websocket")
    print(f"  Envs:      {tcfg.n_envs}  · n_steps:{tcfg.n_steps}  · batch:{tcfg.batch_size}")
    print("=" * 60)

    server_cfg = ServerConfiguration(
        f"ws://{server_host}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )

    rival_pool = load_rival_pool(rivals_dir, seed=tcfg.seed)
    print(f"  ✓ Cargados {rival_pool.num_teams} equipos rivales\n")

    # ── State compartido para rotación de oponentes ──────────────
    # Cada env guarda qué snapshot_id de league está usando como opp
    env_opp_snapshot: dict[int, int] = {i: -1 for i in range(tcfg.n_envs)}

    def opponent_provider(env_idx: int) -> int:
        return env_opp_snapshot.get(env_idx, -1)

    # ── Env factories ────────────────────────────────────────────
    def make_heuristic_env(rank: int):
        """Stage 1-3: oponente heurístico (Random)."""
        def _init():
            env = VGCEnv(
                team_path             = team_path,
                battle_format         = fmt,
                server_configuration  = server_cfg,
                start_listening       = True,
                choose_on_teampreview = True,
            )
            opp = RandomPlayer(
                battle_format        = fmt,
                team                 = rival_pool,
                server_configuration = server_cfg,
            )
            return FlatObsWrapper(SingleAgentWrapper(env, opp))
        return _init

    def make_league_env(rank: int, model, encoder_env_factory: Callable):
        """Stage 4+: oponente del league (LeagueOpponent con snapshot)."""
        def _init():
            env = VGCEnv(
                team_path             = team_path,
                battle_format         = fmt,
                server_configuration  = server_cfg,
                start_listening       = True,
                choose_on_teampreview = True,
            )
            # Encodificador proxy (no se conecta) que LeagueOpponent reusa
            enc_env = encoder_env_factory()

            # Inicialmente con un snapshot del league sampleado o RandomPlayer
            entry = league.sample_pfsp() if not league.is_empty() else None
            if entry is not None:
                from sb3_contrib import RecurrentPPO, MaskablePPO
                ModelCls = RecurrentPPO if tcfg.algorithm == "recurrent_ppo" else MaskablePPO
                snapshot_model = ModelCls.load(entry.path)
                opp = LeagueOpponent(
                    model                = snapshot_model,
                    encoder_env          = enc_env,
                    algorithm            = tcfg.algorithm,
                    snapshot_id          = entry.snapshot_id,
                    battle_format        = fmt,
                    team                 = rival_pool,
                    server_configuration = server_cfg,
                )
                env_opp_snapshot[rank] = entry.snapshot_id
            else:
                opp = RandomPlayer(
                    battle_format        = fmt,
                    team                 = rival_pool,
                    server_configuration = server_cfg,
                )
                env_opp_snapshot[rank] = -1
            return FlatObsWrapper(SingleAgentWrapper(env, opp))
        return _init

    def encoder_env_factory():
        """Crea un VGCEnv 'proxy' sin start_listening para encoding."""
        try:
            return VGCEnv(
                team_path            = team_path,
                battle_format        = fmt,
                server_configuration = server_cfg,
                start_listening      = False,
            )
        except TypeError:
            # Algunas versiones de poke-env no permiten start_listening=False
            return VGCEnv(
                team_path            = team_path,
                battle_format        = fmt,
                server_configuration = server_cfg,
            )

    # ── Construcción inicial del VecEnv (heurístico) ─────────────
    venv = DummyVecEnv([make_heuristic_env(i) for i in range(tcfg.n_envs)])

    # ── Modelo ───────────────────────────────────────────────────
    if tcfg.algorithm == "recurrent_ppo":
        ModelCls, PolicyCls = import_recurrent_ppo()
    elif tcfg.algorithm == "maskable_ppo":
        ModelCls, PolicyCls = import_maskable_ppo()
    else:
        raise ValueError(f"Algoritmo desconocido: {tcfg.algorithm}")

    if resume_path:
        print(f"\n  Cargando checkpoint: {resume_path}")
        model = ModelCls.load(resume_path, env=venv)
    else:
        model = ModelCls(
            policy          = PolicyCls,
            env             = venv,
            learning_rate   = tcfg.linear_lr_schedule(),
            n_steps         = tcfg.n_steps,
            batch_size      = tcfg.batch_size,
            n_epochs        = tcfg.n_epochs,
            gamma           = tcfg.gamma,
            gae_lambda      = tcfg.gae_lambda,
            clip_range      = tcfg.clip_range,
            ent_coef        = tcfg.ent_coef,
            vf_coef         = tcfg.vf_coef,
            max_grad_norm   = tcfg.max_grad_norm,
            policy_kwargs   = build_policy_kwargs(tcfg, tcfg.algorithm),
            tensorboard_log = str(log_dir),
            verbose         = 1,
            seed            = tcfg.seed,
        )

    # ── Curriculum + League ──────────────────────────────────────
    scheduler = CurriculumScheduler(tcfg)
    scheduler.start(current_timestep=0)

    league = SelfPlayLeague(
        max_size      = tcfg.league_max_size,
        eviction_kind = tcfg.league_eviction,
        min_battles_for_eviction = tcfg.league_min_battles_for_eviction,
        seed          = tcfg.seed,
    )

    def push_reward_config_to_envs(cfg):
        for i in range(tcfg.n_envs):
            try:
                inner = venv.envs[i]
                e = inner
                while hasattr(e, "env") and not hasattr(e, "set_reward_config"):
                    e = e.env
                if hasattr(e, "set_reward_config"):
                    e.set_reward_config(cfg)
            except Exception as ex:
                print(f"  [stage advance] env {i}: {ex}")

    def rebuild_envs_with_league():
        """Stage 4 transition: rearma el VecEnv con LeagueOpponents."""
        nonlocal venv
        if league.is_empty():
            print("  [stage4] league vacío, manteniendo opp heurístico")
            return
        try:
            # Cierre limpio del VecEnv anterior
            venv.close()
        except Exception:
            pass
        venv = DummyVecEnv([
            make_league_env(i, model, encoder_env_factory)
            for i in range(tcfg.n_envs)
        ])
        model.set_env(venv)
        print(f"  [stage4] envs reconstruidos con LeagueOpponent (pool={len(league)})")

    def on_stage_advance(sched, new_stage: int):
        cfg = sched.current_reward_config(model.num_timesteps)
        push_reward_config_to_envs(cfg)
        if new_stage == 4:
            rebuild_envs_with_league()

    # ── Callbacks ────────────────────────────────────────────────
    activation_recorder = ActivationRecorder()

    cb_loss      = LossPlateauCallback(scheduler, on_stage_advance, verbose=1)
    cb_breakdown = RewardBreakdownCallback(scheduler)
    cb_activ     = ActivationStatsCallback(activation_recorder, log_every=tcfg.activation_log_every)
    cb_winrate   = WinRateCallback(window=tcfg.win_rate_window)
    cb_phase     = PhaseLogCallback(scheduler)
    cb_snapshot  = SnapshotLeagueCallback(
        league=league, scheduler=scheduler,
        every=tcfg.league_snapshot_every,
        snapshot_dir=league_dir,
        min_stage=3,
        verbose=1,
    )
    cb_lresult   = LeagueResultCallback(league=league, opponent_provider=opponent_provider)
    cb_ckpt      = CheckpointCallback(
        save_freq   = tcfg.checkpoint_every // max(1, tcfg.n_envs),
        save_path   = str(ckpt_dir),
        name_prefix = "vgc",
    )

    callbacks = CallbackList([
        cb_loss, cb_breakdown, cb_activ, cb_winrate, cb_phase,
        cb_snapshot, cb_lresult, cb_ckpt,
    ])

    # ── Entrenamiento ────────────────────────────────────────────
    print(f"\n  Total timesteps: {tcfg.total_timesteps:,}")
    print(f"  Iniciando entrenamiento...\n")

    try:
        model.learn(
            total_timesteps = tcfg.total_timesteps,
            callback        = callbacks,
            progress_bar    = True,
        )
    except KeyboardInterrupt:
        print("\n  [!] Entrenamiento interrumpido por usuario")
    finally:
        scheduler.finalize(model.num_timesteps)

        final_path = ckpt_dir / "vgc_final"
        model.save(str(final_path))
        print(f"\n  ✓ Modelo final: {final_path}.zip")

        # ── Tournament round-robin entre miembros del league ─────
        if not skip_tournament and len(league) >= 2:
            print(f"\n  Ejecutando round-robin entre {len(league)} miembros del league...")
            try:
                run_tournament(
                    league              = league,
                    encoder_env_factory = encoder_env_factory,
                    server_cfg          = server_cfg,
                    battle_format       = fmt,
                    rival_pool          = rival_pool,
                    algorithm           = tcfg.algorithm,
                    n_battles_per_pair  = 5,
                    verbose             = True,
                )
            except Exception as ex:
                print(f"  [!] tournament falló: {ex}")

        # ── Reporte final ────────────────────────────────────────
        report_path = generate_final_report(
            scheduler  = scheduler,
            league     = league,
            train_cfg  = tcfg,
            output_dir = base_dir / tcfg.report_dir,
            extra_meta = {
                "algorithm":     tcfg.algorithm,
                "team_size":     6,
                "rival_pool":    rival_pool.num_teams,
                "rival_yields":  rival_pool.yield_counts,
            },
        )
        print(f"\n  ✓ Reporte: {report_path}/report.html")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGC Bot - Curriculum Training")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--format",  type=str, default=None)
    parser.add_argument("--resume",  type=str, default=None)
    parser.add_argument("--server",  type=str, default=None)
    parser.add_argument("--algorithm", type=str, default=None,
                        choices=["recurrent_ppo", "maskable_ppo"])
    parser.add_argument("--n-envs",  type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--no-tournament", action="store_true",
                        help="No correr round-robin del league al cierre")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        train(
            resume_path     = args.resume,
            server          = args.server,
            battle_format   = args.format,
            algorithm       = args.algorithm,
            n_envs          = args.n_envs,
            total_timesteps = args.total_timesteps,
            skip_tournament = args.no_tournament,
        )
