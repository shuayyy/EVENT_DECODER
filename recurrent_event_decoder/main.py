import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from recurrent_event_decoder.collate import recurrent_event_collate
from recurrent_event_decoder.config import RecurrentEventConfig
from recurrent_event_decoder.dataset import PreparedRecurrentEventDataset
from recurrent_event_decoder.losses import (
    recurrent_event_vector_loss,
)
from recurrent_event_decoder.metrics import compute_recurrent_metrics
from recurrent_event_decoder.model import RecurrentEventDecoder
from recurrent_event_decoder.train import move_batch_to_device


class ProgressBar:
    def __init__(self, total, label):
        self.total = max(int(total), 1)
        self.label = label
        self.current = 0
        self.render()

    def update(self, suffix=""):
        self.current += 1
        self.render(suffix)

    def render(self, suffix=""):
        width = 30
        ratio = self.current / self.total
        filled = min(width, int(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r{self.label}: [{bar}] {self.current}/{self.total} {suffix}",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def close(self):
        print(file=sys.stderr)


class RecurrentTrainingPlotter:
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.enabled = False
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_bit_accuracy = []
        self.val_bit_accuracy = []
        self.val_continue_accuracy = []
        self.val_time_mae_us = []

        try:
            import os

            os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            import matplotlib

            matplotlib.use("Agg")

            import matplotlib.pyplot as plt

            self.plt = plt
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.figure, self.axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
            self.figure.suptitle("Recurrent Event Decoder Training")

            (self.train_loss_line,) = self.axes[0].plot(
                [], [], label="train_loss", color="#1f77b4"
            )
            (self.val_loss_line,) = self.axes[0].plot(
                [], [], label="val_loss", color="#ff7f0e"
            )
            self.axes[0].set_ylabel("Loss")
            self.axes[0].grid(True, alpha=0.3)
            self.axes[0].legend()

            (self.train_bit_line,) = self.axes[1].plot(
                [], [], label="train_bit_accuracy", color="#2ca02c"
            )
            (self.val_bit_line,) = self.axes[1].plot(
                [], [], label="val_bit_accuracy", color="#d62728"
            )
            (self.val_continue_line,) = self.axes[1].plot(
                [], [], label="val_continue_accuracy", color="#9467bd"
            )
            self.axes[1].set_ylabel("Accuracy")
            self.axes[1].set_ylim(0.0, 1.0)
            self.axes[1].grid(True, alpha=0.3)
            self.axes[1].legend()

            (self.val_time_line,) = self.axes[2].plot(
                [], [], label="val_time_mae_us", color="#8c564b"
            )
            self.axes[2].set_xlabel("Epoch")
            self.axes[2].set_ylabel("Time MAE (us)")
            self.axes[2].grid(True, alpha=0.3)
            self.axes[2].legend()

            self.figure.tight_layout()
            self.enabled = True
        except Exception as error:
            print(f"Live plot disabled: {error}", file=sys.stderr)

    def update(self, epoch, train_metrics, val_metrics):
        if not self.enabled:
            return

        self.epochs.append(epoch)
        self.train_loss.append(train_metrics["loss"])
        self.val_loss.append(val_metrics["loss"])
        self.train_bit_accuracy.append(train_metrics.get("bit_accuracy", 0.0))
        self.val_bit_accuracy.append(val_metrics.get("bit_accuracy", 0.0))
        self.val_continue_accuracy.append(val_metrics.get("continue_accuracy", 0.0))
        self.val_time_mae_us.append(val_metrics.get("time_mae_us", 0.0))

        self.train_loss_line.set_data(self.epochs, self.train_loss)
        self.val_loss_line.set_data(self.epochs, self.val_loss)
        self.train_bit_line.set_data(self.epochs, self.train_bit_accuracy)
        self.val_bit_line.set_data(self.epochs, self.val_bit_accuracy)
        self.val_continue_line.set_data(self.epochs, self.val_continue_accuracy)
        self.val_time_line.set_data(self.epochs, self.val_time_mae_us)

        for axis in self.axes:
            axis.relim()
            axis.autoscale_view()
        self.axes[1].set_ylim(0.0, 1.0)
        if len(self.epochs) == 1:
            self.axes[2].set_xlim(0.5, 1.5)
        else:
            self.axes[2].set_xlim(1, max(self.epochs))

        self.figure.tight_layout()
        self.figure.canvas.draw_idle()
        self.figure.savefig(self.output_path, dpi=150)

    def close(self):
        if self.enabled:
            self.plt.close(self.figure)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the recurrent event decoder from a prepared manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/recurrent_event_processed/manifest.json"),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--encoder-stride", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Use only the first N recurrent windows from each prepared episode.",
    )
    parser.add_argument(
        "--bptt-windows",
        type=int,
        default=None,
        help=(
            "Deprecated. Full-sequence backpropagation is now used; this option "
            "is accepted for command compatibility and ignored."
        ),
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/recurrent_event_decoder/checkpoints"),
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("outputs/recurrent_event_decoder/live_training.png"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("outputs/recurrent_event_decoder/metrics.json"),
    )
    return parser.parse_args()


def split_dataset(dataset, val_ratio, seed):
    if len(dataset) < 2:
        return dataset, dataset
    val_count = max(1, int(len(dataset) * val_ratio))
    val_count = min(val_count, len(dataset) - 1)
    train_count = len(dataset) - val_count
    return torch.utils.data.random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(seed),
    )


def average_metrics(metric_sums, batch_count):
    return {
        key: value / max(batch_count, 1)
        for key, value in metric_sums.items()
    }


def run_epoch(model, loader, optimizer, device, config, epoch, phase):
    is_train = optimizer is not None
    model.train(is_train)
    progress = ProgressBar(len(loader), f"{phase} epoch {epoch}")
    total_loss = 0.0
    metric_sums = {}
    batch_count = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch.windows, batch.metadata, batch.valid_window_mask)
            loss = recurrent_event_vector_loss(
                outputs,
                batch.targets,
                batch.valid_window_mask,
                config,
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            metrics = compute_recurrent_metrics(
                outputs.detach(),
                batch.targets,
                batch.valid_window_mask,
                config,
            )
            total_loss += loss.item()
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            batch_count += 1
            avg_loss = total_loss / batch_count
            avg_bit = metric_sums.get("bit_accuracy", 0.0) / batch_count
            progress.update(f"loss={avg_loss:.4f} bit_acc={avg_bit:.4f}")

    progress.close()
    metrics = average_metrics(metric_sums, batch_count)
    metrics["loss"] = total_loss / max(batch_count, 1)
    return metrics


def save_checkpoint(path, model, optimizer, config, epoch, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def main():
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Prepared manifest not found: {args.manifest}. "
            "Run recurrent_event_decoder.prepare_incoming_events first."
        )

    config = replace(
        RecurrentEventConfig(),
        prepared_manifest=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        encoder_stride=args.encoder_stride,
        dropout=args.dropout,
        max_windows=args.max_windows,
    )
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset = PreparedRecurrentEventDataset(args.manifest, config=config)
    train_dataset, val_dataset = split_dataset(dataset, args.val_ratio, args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=recurrent_event_collate,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=recurrent_event_collate,
        num_workers=args.num_workers,
    )

    model = RecurrentEventDecoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    plotter = RecurrentTrainingPlotter(args.plot_path)
    history = []
    best_val_bit_accuracy = -1.0
    run_started_at = datetime.now().isoformat(timespec="seconds")

    print(f"Run started: {run_started_at}", flush=True)
    print(f"Manifest: {args.manifest}", flush=True)
    print(f"Dataset: train={len(train_dataset)} val={len(val_dataset)}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Live plot: {args.plot_path}", flush=True)
    print(f"Checkpoints: {args.checkpoint_dir}", flush=True)

    try:
        for epoch in range(1, config.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                config,
                epoch,
                "train",
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device,
                config,
                epoch,
                "val",
            )

            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(epoch_record)
            plotter.update(epoch, train_metrics, val_metrics)

            print(
                "Epoch "
                f"{epoch}/{config.epochs}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"train_bit_acc={train_metrics.get('bit_accuracy', 0.0):.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_bit_acc={val_metrics.get('bit_accuracy', 0.0):.4f}, "
                f"val_continue_acc={val_metrics.get('continue_accuracy', 0.0):.4f}, "
                f"val_time_mae_us={val_metrics.get('time_mae_us', 0.0):.2f}"
            )

            save_checkpoint(
                args.checkpoint_dir / "latest.pt",
                model,
                optimizer,
                config,
                epoch,
                epoch_record,
            )
            val_bit_accuracy = val_metrics.get("bit_accuracy", 0.0)
            if val_bit_accuracy > best_val_bit_accuracy:
                best_val_bit_accuracy = val_bit_accuracy
                save_checkpoint(
                    args.checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    config,
                    epoch,
                    epoch_record,
                )

            args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_path.write_text(
                json.dumps(
                    {
                        "started_at": run_started_at,
                        "history": history,
                        "best_val_bit_accuracy": best_val_bit_accuracy,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    finally:
        plotter.close()

    print(f"Saved metrics: {args.metrics_path}")
    print(f"Saved live plot: {args.plot_path}")
    print(f"Saved latest checkpoint: {args.checkpoint_dir / 'latest.pt'}")
    print(f"Saved best checkpoint: {args.checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
