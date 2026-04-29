import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader

from recurrent_event_decoder.collate import recurrent_event_collate
from recurrent_event_decoder.config import RecurrentEventConfig
from recurrent_event_decoder.dataset import PreparedRecurrentEventDataset
from recurrent_event_decoder.main import run_epoch, split_dataset
from recurrent_event_decoder.main import (
    coerce_config,
    infer_output_fc_layers_from_state_dict,
    infer_window_encoder_type_from_state_dict,
)
from recurrent_event_decoder.model import RecurrentEventDecoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a recurrent event decoder checkpoint and preserve it."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/recurrent_event_processed_stride10/manifest.json"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--renamed-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def load_config(checkpoint, args):
    raw_config = checkpoint.get("config")
    if raw_config is None:
        raw_config = RecurrentEventConfig()
    config = coerce_config(raw_config)
    checkpoint_encoder_type = infer_window_encoder_type_from_state_dict(
        checkpoint["model_state_dict"]
    )
    checkpoint_output_fc_layers = infer_output_fc_layers_from_state_dict(
        checkpoint["model_state_dict"]
    )
    if checkpoint_encoder_type is not None:
        config = replace(config, window_encoder_type=checkpoint_encoder_type)
    config = replace(config, output_fc_layers=checkpoint_output_fc_layers)
    return replace(
        config,
        prepared_manifest=args.manifest,
        batch_size=args.batch_size,
        use_amp=args.amp,
    )


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = load_config(checkpoint, args)

    dataset = PreparedRecurrentEventDataset(args.manifest, config=config)
    train_dataset, test_dataset = split_dataset(dataset, args.val_ratio, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=recurrent_event_collate,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=recurrent_event_collate,
        num_workers=args.num_workers,
    )

    model = RecurrentEventDecoder(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_metrics = run_epoch(model, train_loader, None, device, config, 0, "eval-train")
    test_metrics = run_epoch(model, test_loader, None, device, config, 0, "eval-test")
    average_bit_accuracy = (
        train_metrics.get("bit_accuracy", 0.0) + test_metrics.get("bit_accuracy", 0.0)
    ) / 2.0

    args.renamed_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint, args.renamed_checkpoint)

    report = {
        "checkpoint": str(args.checkpoint),
        "renamed_checkpoint": str(args.renamed_checkpoint),
        "manifest": str(args.manifest),
        "epoch": checkpoint.get("epoch"),
        "train": train_metrics,
        "test": test_metrics,
        "average_bit_accuracy": average_bit_accuracy,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Test bit accuracy: {test_metrics.get('bit_accuracy', 0.0):.6f}")
    print(f"Average bit accuracy: {average_bit_accuracy:.6f}")
    print(f"Saved report: {args.output_json}")
    print(f"Saved checkpoint copy: {args.renamed_checkpoint}")


if __name__ == "__main__":
    main()
