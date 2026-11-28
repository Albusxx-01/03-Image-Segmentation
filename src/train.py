"""Seeded training loop with checkpoints, LR scheduling, and early stopping.

The loop trains a U-Net with the configured loss, validates every epoch, writes
``models/best.pt`` and ``models/last.pt``, and appends to
``results/train_history.json``. CUDA runs use automatic mixed precision.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from src.config import (
    DEFAULT_CONFIG_PATH,
    config_path,
    load_config,
    resolve_device,
    resolve_path,
    seed_from_config,
    set_seed,
)
from src.data import build_dataloaders
from src.losses import build_loss
from src.metrics import compute_metrics
from src.model import build_model


def _autocast(device: torch.device, enabled: bool) -> Any:
    if enabled:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def _make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _run_epoch(
    model: torch.nn.Module,
    loader: Any,
    loss_fn: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    use_amp: bool = False,
    threshold: float = 0.5,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with _autocast(device, use_amp):
                logits = model(images)
                loss = loss_fn(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * images.size(0)
        if not training:
            logits_chunks.append(logits.detach().cpu())
            target_chunks.append(targets.detach().cpu())

    metrics = {"loss": total_loss / max(len(loader.dataset), 1)}
    if not training:
        metrics.update(
            compute_metrics(
                torch.cat(logits_chunks), torch.cat(target_chunks), threshold
            )
        )
    return metrics


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    best_val_loss: float,
    config: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_val_loss": best_val_loss,
            "metrics": metrics,
            "config": {key: value for key, value in config.items() if key != "_meta"},
        },
        path,
    )


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict) or "model_state" not in state:
        return {"model_state": state, "epoch": -1, "best_val_loss": float("inf")}
    return state


def _build_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: dict[str, Any]
) -> Any | None:
    name = str(train_cfg.get("scheduler", "plateau")).lower()
    if name in ("none", "off", ""):
        return None
    if name == "plateau":
        patience = max(1, int(train_cfg.get("patience", 5)) // 2)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience
        )
    raise ValueError(f"unknown scheduler: {name}")


def train(config: dict[str, Any], resume: str | Path | None = None) -> dict[str, Any]:
    """Train a U-Net and write checkpoints plus ``train_history.json``."""

    set_seed(seed_from_config(config))
    device = torch.device(resolve_device(config))
    model = build_model(config).to(device)
    loss_fn = build_loss(config)
    loaders = build_dataloaders(config)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    train_cfg = config["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = _build_scheduler(optimizer, train_cfg)
    use_amp = device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)

    start_epoch = 0
    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []
    if resume is not None:
        resume_path = resolve_path(resume, config["_meta"]["root"])
        state = _load_checkpoint(resume_path, device)
        model.load_state_dict(state["model_state"])
        if state.get("optimizer_state") is not None:
            optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state.get("epoch", -1)) + 1
        best_val_loss = float(state.get("best_val_loss", best_val_loss))

    checkpoint_dir = config_path(config, "train.checkpoint_dir")
    best_path = checkpoint_dir / str(train_cfg["best_checkpoint"])
    last_path = checkpoint_dir / str(train_cfg["last_checkpoint"])
    history_path = config_path(config, "train.history_path")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg.get("patience", 5))
    min_delta = float(train_cfg.get("min_delta", 0.0))
    threshold = float(config["eval"].get("threshold", 0.5))
    stale_epochs = 0

    for epoch in range(start_epoch, epochs):
        train_metrics = _run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            use_amp=use_amp,
            threshold=threshold,
        )

        learning_rate = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "lr": learning_rate,
        }
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        improved = val_metrics["loss"] < best_val_loss - min_delta
        if improved:
            best_val_loss = val_metrics["loss"]
            stale_epochs = 0
            _save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                config,
                val_metrics,
            )
        else:
            stale_epochs += 1

        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        _save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            config,
            val_metrics,
        )

        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"lr={learning_rate:.6f}"
        )

        if stale_epochs >= patience:
            print(f"early stopping after {stale_epochs} stale epoch(s)")
            break

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "checkpoint_dir": str(checkpoint_dir),
        "history_path": str(history_path),
        "device": str(device),
        "epochs_run": len(history),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.train",
        description="Train the U-Net on image/mask pairs.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file (default: config/unet.yaml).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint to resume training from.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = train(config, resume=args.resume)
    print(f"device: {result['device']}")
    print(f"epochs run: {result['epochs_run']}")
    print(f"best val loss: {result['best_val_loss']:.4f}")
    print(f"history: {result['history_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
