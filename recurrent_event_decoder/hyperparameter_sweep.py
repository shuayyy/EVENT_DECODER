import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run recurrent event decoder training repeatedly across a hyperparameter "
            "grid and keep the checkpoint with the best average train/test bit accuracy."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/recurrent_event_processed/manifest.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/recurrent_event_decoder/sweeps"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Single epoch target. Use --epoch-values for an epoch sweep.",
    )
    parser.add_argument(
        "--epoch-values",
        default="50,60,70,80,90,100",
        help=(
            "Comma-separated epoch targets. For each hyperparameter combination, "
            "training resumes from the previous target instead of restarting."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--learning-rates", default="0.0003,0.0001")
    parser.add_argument("--weight-decays", default="0.0,0.0001")
    parser.add_argument("--dropouts", default="0.1,0.2")
    parser.add_argument("--grad-clips", default="1.0")
    parser.add_argument("--encoder-strides", default="4")
    parser.add_argument("--output-fc-layers", type=int, default=2)
    parser.add_argument("--output-fc-dim", type=int, default=128)
    parser.add_argument(
        "--window-encoder-type",
        choices=("summary", "conv"),
        default="summary",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--no-weighted-bce", action="store_true")
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Optional name for this sweep. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned runs without launching training.",
    )
    return parser.parse_args()


def build_grid(args):
    keys = [
        "learning_rate",
        "weight_decay",
        "dropout",
        "grad_clip",
        "encoder_stride",
    ]
    values = [
        parse_float_list(args.learning_rates),
        parse_float_list(args.weight_decays),
        parse_float_list(args.dropouts),
        parse_float_list(args.grad_clips),
        parse_int_list(args.encoder_strides),
    ]
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def epoch_values(args):
    if args.epochs is not None:
        return [args.epochs]
    values = sorted(set(parse_int_list(args.epoch_values)))
    if not values:
        raise ValueError("--epoch-values must contain at least one epoch")
    return values


def run_command(command):
    print(" ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def score_from_metrics(metrics_path, epoch=None):
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    best_score = -1.0
    best_record = None
    for record in metrics["history"]:
        if epoch is not None and record.get("epoch") != epoch:
            continue
        score = record.get("average_bit_accuracy")
        if score is None:
            train_bit = record["train"].get("bit_accuracy", 0.0)
            val_bit = record["val"].get("bit_accuracy", 0.0)
            score = (train_bit + val_bit) / 2.0
        if score > best_score:
            best_score = score
            best_record = record
    if best_record is None:
        raise ValueError(f"No metrics found for epoch {epoch} in {metrics_path}")
    return best_score, best_record


def main():
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    sweep_name = args.run_prefix or datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
    sweep_dir = args.output_root / sweep_name
    grid = build_grid(args)
    epochs = epoch_values(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_count": len(grid) * len(epochs),
                    "base_hyperparameter_count": len(grid),
                    "epochs": epochs,
                    "runs": grid,
                },
                indent=2,
            )
        )
        return

    sweep_dir.mkdir(parents=True, exist_ok=False)
    results = []
    best_result = None

    for base_run_index, params in enumerate(grid, start=1):
        run_name = f"run_{base_run_index:03d}"
        run_dir = sweep_dir / run_name
        checkpoint_dir = run_dir / "checkpoints"
        metrics_path = run_dir / "metrics.json"
        plot_path = run_dir / "live_training.png"
        run_dir.mkdir(parents=True, exist_ok=False)

        previous_latest = None
        for target_epoch in epochs:
            target_dir = run_dir / f"epoch_{target_epoch:03d}"
            target_dir.mkdir(parents=True, exist_ok=True)

            command = [
                sys.executable,
                "-m",
                "recurrent_event_decoder.main",
                "--manifest",
                str(args.manifest),
                "--epochs",
                str(target_epoch),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
                "--num-workers",
                str(args.num_workers),
                "--val-ratio",
                str(args.val_ratio),
                "--seed",
                str(args.seed),
                "--learning-rate",
                str(params["learning_rate"]),
                "--weight-decay",
                str(params["weight_decay"]),
                "--dropout",
                str(params["dropout"]),
                "--grad-clip",
                str(params["grad_clip"]),
                "--encoder-stride",
                str(params["encoder_stride"]),
                "--output-fc-layers",
                str(args.output_fc_layers),
                "--output-fc-dim",
                str(args.output_fc_dim),
                "--window-encoder-type",
                args.window_encoder_type,
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--plot-path",
                str(plot_path),
                "--metrics-path",
                str(metrics_path),
            ]
            if previous_latest is not None:
                command.extend(["--resume-checkpoint", str(previous_latest)])
            if args.max_windows is not None:
                command.extend(["--max-windows", str(args.max_windows)])
            if args.amp:
                command.append("--amp")
            if args.activation_checkpointing:
                command.append("--activation-checkpointing")
            if args.no_weighted_bce:
                command.append("--no-weighted-bce")

            run_command(command)
            previous_latest = checkpoint_dir / "latest.pt"
            epoch_checkpoint = target_dir / "latest.pt"
            shutil.copy2(previous_latest, epoch_checkpoint)
            score, epoch_record = score_from_metrics(metrics_path, epoch=target_epoch)
            result = {
                "run": run_name,
                "epoch": target_epoch,
                "score": score,
                "params": params,
                "epoch_record": epoch_record,
                "checkpoint": str(epoch_checkpoint),
                "metrics": str(metrics_path),
                "reused_from_previous_epoch": target_epoch != epochs[0],
            }
            results.append(result)
            if best_result is None or score > best_result["score"]:
                best_result = result
                shutil.copy2(epoch_checkpoint, sweep_dir / "best.pt")

            summary = {
                "manifest": str(args.manifest),
                "epochs": epochs,
                "best_result": best_result,
                "results": results,
            }
            with (sweep_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)

    print(f"Sweep complete: {sweep_dir}")
    print(f"Best score: {best_result['score']:.6f}")
    print(f"Best checkpoint: {sweep_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
