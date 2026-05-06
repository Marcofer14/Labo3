"""
src.training.report · Reporte detallado por fase al cierre del run.

Genera:
  reports/<timestamp>/report.html       · informe HTML legible
  reports/<timestamp>/report.json       · datos crudos para análisis
  reports/<timestamp>/plots/*.png       · curvas de loss, reward, win-rate
  reports/<timestamp>/league_stats.json · stats del self-play league
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

# Lazy import de matplotlib para no romper si no está
def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ── HTML helpers ─────────────────────────────────────────────────

def _html_table(rows: list[dict], cols: list[str]) -> str:
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{r.get(c, '')}</td>" for c in cols) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


_HTML_CSS = """
body { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }
h1, h2, h3 { color: #222; }
table { border-collapse: collapse; margin: 1em 0; width: 100%; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
th { background: #f5f5f5; }
.stage { background: #eef; padding: 1em; border-radius: 6px; margin: 1em 0; }
.metric { display: inline-block; margin: 0.5em 1em; }
.metric .label { font-weight: bold; }
img { max-width: 100%; margin: 1em 0; border: 1px solid #ddd; }
code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }
"""


# ── Plot helpers ─────────────────────────────────────────────────

def _plot_loss_curves(stages, plots_dir: Path) -> str:
    plt = _import_plt()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = ["value_loss", "policy_loss", "approx_kl", "explained_variance"]
    keys   = ["losses_value", "losses_policy", "approx_kl", "explained_variance"]
    for ax, title, key in zip(axes.flatten(), titles, keys):
        for s in stages:
            data = getattr(s, key, [])
            if data:
                ax.plot(data, label=f"stage {s.stage}", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("update")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = plots_dir / "loss_curves.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out.name


def _plot_winrate_per_stage(stages, plots_dir: Path) -> str:
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(10, 5))
    win_rates = [s.win_rate for s in stages]
    labels    = [f"stage {s.stage}" for s in stages]
    bars = ax.bar(labels, win_rates, color="#4a90e2")
    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                f"{wr*100:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("win rate")
    ax.set_title("Win rate por stage")
    ax.grid(True, axis="y", alpha=0.3)
    out = plots_dir / "winrate_per_stage.png"
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()
    return out.name


def _plot_reward_breakdown(stages, plots_dir: Path) -> str:
    plt = _import_plt()
    # Recoger todas las claves
    keys = sorted({k for s in stages for k in s.reward_breakdown.keys()})
    if not keys:
        return ""
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.8 / max(1, len(stages))
    x = range(len(keys))
    for i, s in enumerate(stages):
        vals = [s.reward_breakdown.get(k, 0.0) for k in keys]
        ax.bar([xi + i*width for xi in x], vals, width=width, label=f"stage {s.stage}")
    ax.set_xticks([xi + width*(len(stages)-1)/2 for xi in x])
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("acumulado")
    ax.set_title("Reward breakdown acumulado por stage")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = plots_dir / "reward_breakdown.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out.name


def _plot_episode_rewards(stages, plots_dir: Path) -> str:
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(11, 5))
    offset = 0
    for s in stages:
        n = len(s.episode_rewards)
        if n == 0:
            continue
        ax.plot(range(offset, offset+n), s.episode_rewards,
                label=f"stage {s.stage}", alpha=0.5)
        offset += n
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title("Reward por episodio (concatenado)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = plots_dir / "episode_rewards.png"
    plt.savefig(out, dpi=110)
    plt.close()
    return out.name


# ── Report principal ─────────────────────────────────────────────

def generate_final_report(
    scheduler,                  # CurriculumScheduler
    league=None,                # SelfPlayLeague | None
    train_cfg=None,             # TrainingConfig | None
    output_dir: str | Path = "reports",
    extra_meta: Optional[dict] = None,
) -> Path:
    """Genera el reporte completo y devuelve el path del run."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = Path(output_dir) / f"run_{ts}"
    plots_dir = base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    stages = scheduler.history

    # ── JSON crudo ──────────────────────────────────────────────
    json_data = {
        "timestamp": ts,
        "stages": [
            {
                "stage":              s.stage,
                "started":            s.started_timestep,
                "ended":              s.ended_timestep,
                "duration":           s.duration_timesteps,
                "transition_reason":  s.transition_reason,
                "n_episodes":         s.n_episodes,
                "win_rate":           s.win_rate,
                "avg_reward":         s.avg_reward,
                "avg_episode_length": s.avg_episode_length,
                "final_value_loss":   s.losses_value[-1] if s.losses_value else None,
                "final_policy_loss":  s.losses_policy[-1] if s.losses_policy else None,
                "reward_breakdown":   s.reward_breakdown,
                "n_updates":          len(s.losses_value),
            }
            for s in stages
        ],
        "training_config": (train_cfg.__dict__ if train_cfg else None),
        "extra_meta":      extra_meta or {},
    }
    if league is not None:
        json_data["league"] = league.stats()

    (base / "report.json").write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")

    # ── Plots ──────────────────────────────────────────────────
    plot_paths = {}
    try:
        plot_paths["loss"]      = _plot_loss_curves(stages, plots_dir)
        plot_paths["winrate"]   = _plot_winrate_per_stage(stages, plots_dir)
        plot_paths["breakdown"] = _plot_reward_breakdown(stages, plots_dir)
        plot_paths["episodes"]  = _plot_episode_rewards(stages, plots_dir)
    except Exception as ex:
        print(f"  [report] error al generar plots: {ex}")

    # ── HTML ───────────────────────────────────────────────────
    html_parts = [f"<!doctype html><html><head><meta charset='utf-8'>"
                  f"<title>VGC Bot Report — {ts}</title>"
                  f"<style>{_HTML_CSS}</style></head><body>"]
    html_parts.append(f"<h1>VGC Bot — Reporte de entrenamiento</h1>")
    html_parts.append(f"<p>Generado: <code>{ts}</code></p>")

    # Resumen global
    total_episodes = sum(s.n_episodes for s in stages)
    total_updates  = sum(len(s.losses_value) for s in stages)
    overall_wr     = (sum(sum(s.episode_won) for s in stages) /
                      max(1, sum(len(s.episode_won) for s in stages)))
    final_t        = stages[-1].ended_timestep if stages else 0

    html_parts.append("<h2>Resumen global</h2>")
    html_parts.append("<div class='stage'>")
    html_parts.append(f"<div class='metric'><span class='label'>Stages completados:</span> {len(stages)}</div>")
    html_parts.append(f"<div class='metric'><span class='label'>Timesteps totales:</span> {final_t:,}</div>")
    html_parts.append(f"<div class='metric'><span class='label'>Episodios totales:</span> {total_episodes:,}</div>")
    html_parts.append(f"<div class='metric'><span class='label'>Updates de PPO:</span> {total_updates:,}</div>")
    html_parts.append(f"<div class='metric'><span class='label'>Win rate global:</span> {overall_wr*100:.2f}%</div>")
    html_parts.append("</div>")

    # Per stage
    html_parts.append("<h2>Detalle por stage</h2>")
    for s in stages:
        html_parts.append(f"<div class='stage'><h3>Stage {s.stage}</h3>")
        html_parts.append(f"<div class='metric'><span class='label'>Timesteps:</span> {s.duration_timesteps:,} ({s.started_timestep:,} → {s.ended_timestep:,})</div>")
        html_parts.append(f"<div class='metric'><span class='label'>Razón cierre:</span> {s.transition_reason}</div>")
        html_parts.append(f"<div class='metric'><span class='label'>Episodios:</span> {s.n_episodes:,}</div>")
        html_parts.append(f"<div class='metric'><span class='label'>Win rate:</span> {s.win_rate*100:.2f}%</div>")
        html_parts.append(f"<div class='metric'><span class='label'>Reward avg:</span> {s.avg_reward:.2f}</div>")
        html_parts.append(f"<div class='metric'><span class='label'>Episodio avg len:</span> {s.avg_episode_length:.1f}</div>")
        html_parts.append(f"<div class='metric'><span class='label'>Updates PPO:</span> {len(s.losses_value):,}</div>")
        if s.losses_value:
            html_parts.append(f"<div class='metric'><span class='label'>value_loss final:</span> {s.losses_value[-1]:.4f}</div>")
            html_parts.append(f"<div class='metric'><span class='label'>policy_loss final:</span> {s.losses_policy[-1]:.4f}</div>")
        # Top componentes del reward
        bd = sorted(s.reward_breakdown.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        if bd:
            html_parts.append("<h4>Top 10 componentes del reward (acumulado)</h4>")
            html_parts.append(_html_table(
                [{"componente": k, "valor": f"{v:+.2f}"} for k, v in bd],
                ["componente", "valor"]
            ))
        html_parts.append("</div>")

    # Plots
    html_parts.append("<h2>Curvas y plots</h2>")
    for label, fname in plot_paths.items():
        if not fname:
            continue
        html_parts.append(f"<h3>{label}</h3><img src='plots/{fname}' alt='{label}'>")

    # League
    if league is not None:
        html_parts.append("<h2>Self-play league</h2>")
        stats = league.stats()
        if stats:
            html_parts.append(_html_table(stats, ["id", "label", "timestep", "battles", "wins", "losses", "win_rate"]))
        else:
            html_parts.append("<p>No hubo entradas en el league.</p>")

    # Config
    if train_cfg is not None:
        html_parts.append("<h2>Configuración del entrenamiento</h2>")
        items = []
        for k, v in train_cfg.__dict__.items():
            items.append({"clave": k, "valor": str(v)})
        html_parts.append(_html_table(items, ["clave", "valor"]))

    html_parts.append("</body></html>")
    (base / "report.html").write_text("\n".join(html_parts), encoding="utf-8")

    print(f"\n  ✓ Reporte generado en: {base}")
    print(f"    · {base / 'report.html'}")
    print(f"    · {base / 'report.json'}")
    return base
