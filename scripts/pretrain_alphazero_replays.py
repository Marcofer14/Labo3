"""Pretrain the AlphaZero policy/value ranker from replay double decisions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alphazero.features import (
    ACTION_FEATURE_SIZE,
    STATE_FEATURE_SIZE,
    outcome_to_value,
    replay_action_features,
    replay_state_features,
)
from src.alphazero.network import AlphaZeroPolicyValueNet, save_checkpoint


class ReplayDecisionDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.states = []
        self.actions = []
        self.values = []
        for row in rows:
            if not row.get("actions"):
                continue
            self.states.append(torch.tensor(replay_state_features(row), dtype=torch.float32))
            self.actions.append(torch.tensor(replay_action_features(row), dtype=torch.float32))
            self.values.append(torch.tensor(outcome_to_value(row.get("outcome")), dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return self.states[index], self.actions[index], self.values[index]


def load_rows(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def run_epoch(model, loader, optimizer, device, value_weight: float, train: bool) -> dict[str, float]:
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    total_correct = 0
    total = 0
    model.train(train)

    for states, actions, values in loader:
        states = states.to(device)
        actions = actions.to(device)
        values = values.to(device)
        batch_size = states.shape[0]
        candidates = actions.unsqueeze(0).expand(batch_size, batch_size, ACTION_FEATURE_SIZE)
        mask = torch.ones((batch_size, batch_size), dtype=torch.bool, device=device)
        targets = torch.arange(batch_size, dtype=torch.long, device=device)

        with torch.set_grad_enabled(train):
            logits, pred_values = model(states, candidates, mask)
            policy_loss = F.cross_entropy(logits, targets)
            value_loss = F.mse_loss(pred_values, values)
            loss = policy_loss + value_weight * value_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += float(loss.detach().cpu()) * batch_size
        total_policy += float(policy_loss.detach().cpu()) * batch_size
        total_value += float(value_loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().detach().cpu())
        total += batch_size

    return {
        "loss": total_loss / max(total, 1),
        "policy_loss": total_policy / max(total, 1),
        "value_mse": total_value / max(total, 1),
        "acc": total_correct / max(total, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain AlphaZero ranker from VGC replay decisions")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/replays/datasets/gen9vgc2026regi_double_decisions.jsonl"),
        help="Path to *_double_decisions.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/alphazero_pretrain"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--embedding-size", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    rows = load_rows(args.dataset, args.limit or None)
    dataset = ReplayDecisionDataset(rows)
    if len(dataset) < 2:
        raise RuntimeError("Need at least two replay decisions for in-batch contrastive training")

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(7),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device)
    model = AlphaZeroPolicyValueNet(
        state_size=STATE_FEATURE_SIZE,
        action_size=ACTION_FEATURE_SIZE,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        layers=args.layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded replay decisions: train={train_size} val={val_size} total={len(dataset)}")
    print(f"Model: state={STATE_FEATURE_SIZE} action={ACTION_FEATURE_SIZE}")
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.value_weight, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args.value_weight, train=False)
        elapsed = time.time() - start
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"value_mse={val_metrics['value_mse']:.4f} time={elapsed:.1f}s",
            flush=True,
        )
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            extra={"epoch": epoch, "train_metrics": train_metrics, "val_metrics": val_metrics},
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                extra={"epoch": epoch, "train_metrics": train_metrics, "val_metrics": val_metrics},
            )

    print(f"Saved checkpoints to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
