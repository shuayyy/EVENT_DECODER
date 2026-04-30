import argparse
from dataclasses import fields, replace
from datetime import datetime
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from recurrent_event_decoder.collate import recurrent_event_collate
from recurrent_event_decoder.config import RecurrentEventConfig
from recurrent_event_decoder.dataset import PreparedRecurrentEventDataset
from recurrent_event_decoder.losses import cutoff_recurrent_event_vector_loss
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
        self.val_progress_mae = []

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

            (self.val_progress_line,) = self.axes[2].plot(
                [], [], label="val_progress_mae", color="#8c564b"
            )
            self.axes[2].set_xlabel("Epoch")
            self.axes[2].set_ylabel("Progress MAE")
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
        self.val_progress_mae.append(val_metrics.get("progress_mae", 0.0))

        self.train_loss_line.set_data(self.epochs, self.train_loss)
        self.val_loss_line.set_data(self.epochs, self.val_loss)
        self.train_bit_line.set_data(self.epochs, self.train_bit_accuracy)
        self.val_bit_line.set_data(self.epochs, self.val_bit_accuracy)
        self.val_continue_line.set_data(self.epochs, self.val_continue_accuracy)
        self.val_progress_line.set_data(self.epochs, self.val_progress_mae)

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
    parser.add_argument(
        "--no-weighted-bce",
        action="store_true",
        help="Disable per-bit positive weighting in BCE loss.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--encoder-stride", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--window-encoder-type",
        choices=("summary", "conv"),
        default="summary",
        help="summary pools each event window; conv uses the old Conv1d encoder.",
    )
    parser.add_argument("--event-summary-hidden-dim", type=int, default=64)
    parser.add_argument("--temp-encoder-1-dim", type=int, default=512)
    parser.add_argument("--temp-encoder-2-dim", type=int, default=128)
    parser.add_argument("--temp-encoder-3-dim", type=int, default=64)
    parser.add_argument("--recurrent-1-dim", type=int, default=80)
    parser.add_argument("--temp-encoder-4-dim", type=int, default=80)
    parser.add_argument("--linear-dim", type=int, default=80)
    parser.add_argument("--temp-decoder-dim", type=int, default=80)
    parser.add_argument("--recurrent-2-dim", type=int, default=128)
    parser.add_argument("--output-fc-layers", type=int, default=2)
    parser.add_argument("--output-fc-dim", type=int, default=128)
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Trade extra compute for lower activation memory in temporal encoders.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision to reduce memory.",
    )
    parser.add_argument(
        "--no-normalize-inputs",
        action="store_true",
        help="Disable event and metadata input normalization.",
    )
    parser.add_argument("--sensor-width", type=float, default=1280.0)
    parser.add_argument("--sensor-height", type=float, default=720.0)
    parser.add_argument(
        "--time-scale-us",
        type=float,
        default=1_000_000.0,
        help="Divisor for relative_time_us input. Default converts us to seconds.",
    )
    parser.add_argument("--repetition-scale", type=float, default=15.0)
    parser.add_argument("--frequency-scale", type=float, default=1000.0)
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
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume model and optimizer state from a previous checkpoint.",
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


def compute_bit_pos_weight(dataset, bit_count):
    positives = torch.zeros(bit_count, dtype=torch.float64)
    total = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        positives += sample.target[:bit_count].to(dtype=torch.float64)
        total += 1
    if total == 0:
        return torch.ones(bit_count, dtype=torch.float32).tolist()

    negatives = float(total) - positives
    weights = negatives / positives.clamp_min(1.0)
    weights = torch.where(positives > 0, weights, torch.ones_like(weights))
    return weights.to(dtype=torch.float32).tolist()


def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_epoch(model, loader, optimizer, device, config, epoch, phase, scaler=None):
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
            autocast_enabled = config.use_amp and device.type == "cuda"
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                outputs = model(batch.windows, batch.metadata, batch.valid_window_mask)
                loss = cutoff_recurrent_event_vector_loss(
                    outputs,
                    batch.targets,
                    batch.valid_window_mask,
                    batch.windows,
                    config,
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if config.grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config.grad_clip,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if config.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()

            metrics = compute_recurrent_metrics(
                outputs.detach(),
                batch.targets,
                batch.valid_window_mask,
                config,
                batch.windows,
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


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def load_existing_history(metrics_path):
    if not metrics_path.exists():
        return [], -1.0, None

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    history = metrics.get("history", [])
    best_average = metrics.get("best_average_bit_accuracy", -1.0)
    started_at = metrics.get("started_at")
    return history, best_average, started_at


def coerce_config(config):
    defaults = RecurrentEventConfig()
    values = {
        field.name: getattr(config, field.name, getattr(defaults, field.name))
        for field in fields(defaults)
    }
    return replace(defaults, **values)


def infer_window_encoder_type_from_state_dict(state_dict):
    if any(key.startswith("window_summary_encoder.") for key in state_dict):
        return "summary"
    if any(key.startswith("temp_encoder_1.") for key in state_dict):
        return "conv"
    return None


def infer_output_fc_layers_from_state_dict(state_dict):
    if any(key.startswith("post_recurrent_fc.") for key in state_dict):
        layer_indices = set()
        for key in state_dict:
            if key.startswith("post_recurrent_fc.") and key.split(".")[-1] == "weight":
                parts = key.split(".")
                if len(parts) > 2 and parts[1].isdigit() and int(parts[1]) % 4 == 1:
                    layer_indices.add(int(parts[1]))
        return len(layer_indices)
    return 0


def main():
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Prepared manifest not found: {args.manifest}. "
            "Run recurrent_event_decoder.prepare_incoming_events first."
        )

    base_config = RecurrentEventConfig()
    resume_checkpoint = None
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        resume_checkpoint = torch.load(
            args.resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        raw_config = resume_checkpoint.get("config", base_config)
        checkpoint_encoder_type = infer_window_encoder_type_from_state_dict(
            resume_checkpoint["model_state_dict"]
        )
        checkpoint_output_fc_layers = infer_output_fc_layers_from_state_dict(
            resume_checkpoint["model_state_dict"]
        )
        base_config = coerce_config(raw_config)
        if checkpoint_encoder_type is not None:
            base_config = replace(base_config, window_encoder_type=checkpoint_encoder_type)
        base_config = replace(base_config, output_fc_layers=checkpoint_output_fc_layers)

    config = replace(
        base_config,
        prepared_manifest=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_weighted_bce=not args.no_weighted_bce,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        encoder_stride=args.encoder_stride,
        dropout=args.dropout,
        window_encoder_type=args.window_encoder_type
        if resume_checkpoint is None
        else base_config.window_encoder_type,
        event_summary_hidden_dim=args.event_summary_hidden_dim
        if resume_checkpoint is None
        else getattr(base_config, "event_summary_hidden_dim", 64),
        temp_encoder_1_dim=args.temp_encoder_1_dim,
        temp_encoder_2_dim=args.temp_encoder_2_dim,
        temp_encoder_3_dim=args.temp_encoder_3_dim,
        recurrent_1_dim=args.recurrent_1_dim,
        temp_encoder_4_dim=args.temp_encoder_4_dim,
        linear_dim=args.linear_dim,
        temp_decoder_dim=args.temp_decoder_dim,
        recurrent_2_dim=args.recurrent_2_dim,
        output_fc_layers=args.output_fc_layers
        if resume_checkpoint is None
        else base_config.output_fc_layers,
        output_fc_dim=args.output_fc_dim
        if resume_checkpoint is None
        else base_config.output_fc_dim,
        activation_checkpointing=args.activation_checkpointing,
        use_amp=args.amp,
        normalize_inputs=not args.no_normalize_inputs,
        sensor_width=args.sensor_width,
        sensor_height=args.sensor_height,
        time_scale_us=args.time_scale_us,
        repetition_scale=args.repetition_scale,
        frequency_scale=args.frequency_scale,
        max_windows=args.max_windows,
    )
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset = PreparedRecurrentEventDataset(args.manifest, config=config)
    train_dataset, val_dataset = split_dataset(dataset, args.val_ratio, args.seed)
    if config.use_weighted_bce and resume_checkpoint is None:
        config = replace(
            config,
            bit_pos_weight=compute_bit_pos_weight(train_dataset, config.bit_count),
        )

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
    scaler = make_grad_scaler(config.use_amp and device.type == "cuda")
    plotter = RecurrentTrainingPlotter(args.plot_path)
    history = []
    best_average_bit_accuracy = -1.0
    run_started_at = datetime.now().isoformat(timespec="seconds")
    start_epoch = 1

    if args.resume_checkpoint is not None:
        checkpoint = load_checkpoint(args.resume_checkpoint, model, optimizer, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        history, best_average_bit_accuracy, previous_started_at = load_existing_history(
            args.metrics_path
        )
        if previous_started_at is not None:
            run_started_at = previous_started_at

    print(f"Run started: {run_started_at}", flush=True)
    print(f"Manifest: {args.manifest}", flush=True)
    print(f"Dataset: train={len(train_dataset)} val={len(val_dataset)}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Live plot: {args.plot_path}", flush=True)
    print(f"Checkpoints: {args.checkpoint_dir}", flush=True)
    if args.resume_checkpoint is not None:
        print(
            f"Resumed from: {args.resume_checkpoint} at epoch {start_epoch}",
            flush=True,
        )

    try:
        for epoch in range(start_epoch, config.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                config,
                epoch,
                "train",
                scaler,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device,
                config,
                epoch,
                "val",
                None,
            )

            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            average_bit_accuracy = (
                train_metrics.get("bit_accuracy", 0.0)
                + val_metrics.get("bit_accuracy", 0.0)
            ) / 2.0
            epoch_record["average_bit_accuracy"] = average_bit_accuracy
            history.append(epoch_record)
            plotter.update(epoch, train_metrics, val_metrics)

            print(
                "Epoch "
                f"{epoch}/{config.epochs}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"train_bit_acc={train_metrics.get('bit_accuracy', 0.0):.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_bit_acc={val_metrics.get('bit_accuracy', 0.0):.4f}, "
                f"avg_bit_acc={average_bit_accuracy:.4f}, "
                f"val_continue_acc={val_metrics.get('continue_accuracy', 0.0):.4f}, "
                f"val_progress_mae={val_metrics.get('progress_mae', 0.0):.4f}"
            )

            save_checkpoint(
                args.checkpoint_dir / "latest.pt",
                model,
                optimizer,
                config,
                epoch,
                epoch_record,
            )
            if average_bit_accuracy > best_average_bit_accuracy:
                best_average_bit_accuracy = average_bit_accuracy
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
                        "best_average_bit_accuracy": best_average_bit_accuracy,
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
