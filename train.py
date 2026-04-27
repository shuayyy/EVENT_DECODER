import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from datasets.collate import ctc_collate
from datasets.dataset import ProcessedRepetitionDataset
from datasets.tokenizer import Tokenizer
from model.model import LSTMdecoder, MambaDecoder, TransformerDecoder


CONFIG_PATH = Path("config/hyperparameter.yaml")


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_model_config(config):
    shared_model_config = dict(config["model"])
    model_type = shared_model_config["model_type"].strip().lower()
    per_model_config = config["models"][model_type]
    return model_type, {**shared_model_config, **per_model_config}


def build_model(model_type, model_config):
    if model_type == "lstm":
        return LSTMdecoder(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            output_dim=model_config["output_dim"],
            num_layers=model_config.get("num_layers", 1),
            bidirectional=model_config.get("bidirectional", False),
            dropout=model_config.get("dropout", 0.0),
        )
    if model_type == "transformer":
        return TransformerDecoder(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            output_dim=model_config["output_dim"],
            num_layers=model_config.get("num_layers", 1),
            nhead=model_config.get("nhead", 4),
            ff_dim=model_config.get("ff_dim"),
            dropout=model_config.get("dropout", 0.0),
            use_positional_encoding=model_config.get("use_positional_encoding", True),
            max_len=model_config.get("max_len", 5000),
        )
    if model_type == "mamba":
        return MambaDecoder(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            output_dim=model_config["output_dim"],
            num_layers=model_config.get("num_layers", 1),
            d_state=model_config.get("d_state", 16),
            d_conv=model_config.get("d_conv", 4),
            expand=model_config.get("expand", 2),
            dropout=model_config.get("dropout", 0.0),
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def validate_model_runtime(model_config, device):
    model_type = model_config["model_type"].strip().lower()
    if model_type == "mamba" and device.type != "cuda":
        raise RuntimeError("Mamba does not work on CPU. Use CUDA or choose a different model_type.")


def split_sequences(samples, split_config):
    sequences = sorted({sample["sequence"] for sample in samples})
    rng = random.Random(split_config["seed"])
    rng.shuffle(sequences)

    total_ratio = split_config["train_ratio"] + split_config["val_ratio"]
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio and val_ratio must sum to 1.0, "
            f"got {split_config['train_ratio']} + {split_config['val_ratio']} = {total_ratio}"
        )

    train_count = int(len(sequences) * split_config["train_ratio"])
    train_count = min(max(train_count, 1), len(sequences) - 1)

    train_sequences = sequences[:train_count]
    val_sequences = sequences[train_count:]

    return {
        "train": train_sequences,
        "val": val_sequences,
    }


def recover_target_strings(targets, target_lengths, tokenizer):
    target_strings = []
    offset = 0

    for length in target_lengths.tolist():
        target_slice = targets[offset : offset + length].tolist()
        target_strings.append(tokenizer.decode(target_slice))
        offset += length

    return target_strings


def greedy_ctc_decode(logits, tokenizer):
    predicted_ids = logits.argmax(dim=-1).detach().cpu()
    decoded_predictions = []

    for row in predicted_ids.tolist():
        collapsed_tokens = []
        previous_token = None

        for token_id in row:
            if token_id != previous_token:
                collapsed_tokens.append(token_id)
            previous_token = token_id

        bit_string = "".join(
            "0" if token_id == tokenizer.zero_id
            else "1" if token_id == tokenizer.one_id
            else ""
            for token_id in collapsed_tokens
            if token_id != tokenizer.blank_id
        )
        decoded_predictions.append(bit_string)

    return decoded_predictions


def compute_sequence_metrics(predictions, targets, target_bit_length):
    exact_matches = 0
    matching_bits = 0
    total_target_bits = 0

    for prediction, target in zip(predictions, targets):
        exact_matches += int(prediction == target)

        compare_length = min(len(prediction), target_bit_length)
        overlap_matches = sum(
            pred_bit == target_bit
            for pred_bit, target_bit in zip(
                prediction[:compare_length],
                target[:compare_length],
            )
        )
        overlap_wrong = compare_length - overlap_matches
        length_penalty = abs(len(prediction) - target_bit_length)
        total_wrong_bits = overlap_wrong + length_penalty
        correct_bits = max(target_bit_length - total_wrong_bits, 0)

        matching_bits += correct_bits
        total_target_bits += target_bit_length

    return exact_matches, matching_bits, total_target_bits


def evaluate(
    model,
    loader,
    criterion,
    device,
    tokenizer,
    target_bit_length,
    debug_first_batch=False,
    debug_logger=None,
):
    model.eval()
    total_loss = 0.0
    exact_matches = 0
    matching_bits = 0
    total_target_bits = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            inputs, targets, input_lengths, target_lengths = [
                tensor.to(device) for tensor in batch
            ]

            logits = model(inputs, input_lengths)
            log_probs = nn.functional.log_softmax(logits, dim=-1).permute(1, 0, 2)

            loss = criterion(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )
            total_loss += loss.item()

            predictions = greedy_ctc_decode(logits, tokenizer)
            references = recover_target_strings(
                targets.detach().cpu(),
                target_lengths.detach().cpu(),
                tokenizer,
            )
            batch_exact_matches, batch_matching_bits, batch_total_target_bits = (
                compute_sequence_metrics(
                    predictions,
                    references,
                    target_bit_length,
                )
            )
            exact_matches += batch_exact_matches
            matching_bits += batch_matching_bits
            total_target_bits += batch_total_target_bits

            if debug_first_batch and batch_index == 0 and predictions:
                debug_message = (
                    "Validation debug: "
                    f"target={references[0]} "
                    f"pred={predictions[0]} "
                    f"pred_len={len(predictions[0])} "
                    f"target_len={len(references[0])}"
                )
                if debug_logger is not None:
                    debug_logger(debug_message)
                else:
                    print(debug_message)

    return {
        "loss": total_loss / max(len(loader), 1),
        "bit_accuracy": matching_bits / max(total_target_bits, 1),
        "exact_accuracy": exact_matches / max(len(loader.dataset), 1),
    }


def write_log_file(log_path, log_lines):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def train():
    config = load_config()
    model_type, model_config = resolve_model_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_model_runtime(model_config, device)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config["training"].get("log_dir", "outputs/logs"))
    log_path = log_dir / f"{run_timestamp}_{model_type}.txt"
    log_lines = []
    tokenizer = Tokenizer()
    target_bit_length = int(config["data"]["target_bit_length"])

    def log(message):
        print(message)
        log_lines.append(message)

    try:
        manifest_path = Path(
            config["data"].get(
                "processed_manifest",
                "data/dataset_processed/manifest.json",
            )
        )
        dataset = ProcessedRepetitionDataset(manifest_path=manifest_path)
        split_map = split_sequences(dataset.samples, config["split"])

        train_dataset = ProcessedRepetitionDataset(
            manifest_path=manifest_path,
            sequences=split_map["train"],
        )
        val_dataset = ProcessedRepetitionDataset(
            manifest_path=manifest_path,
            sequences=split_map["val"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=model_config["batch_size"],
            shuffle=config["training"]["shuffle"],
            collate_fn=ctc_collate,
            num_workers=config["training"].get("num_workers", 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=model_config["batch_size"],
            shuffle=False,
            collate_fn=ctc_collate,
            num_workers=config["training"].get("num_workers", 0),
        )

        model = build_model(model_type, model_config).to(device)

        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_config["learning_rate"],
            weight_decay=model_config.get("weight_decay", 0.0),
        )

        log(f"Run timestamp: {run_timestamp}")
        log(f"Using processed manifest: {manifest_path}")
        log(f"Model type: {model_type}")
        log(f"Device: {device}")
        log(f"Model config: {model_config}")
        log(
            "Split summary: "
            f"train={{sequences: {len(split_map['train'])}, samples: {len(train_dataset)}}}, "
            f"val={{sequences: {len(split_map['val'])}, samples: {len(val_dataset)}}}"
        )

        grad_clip = model_config.get("grad_clip")
        debug_first_val_batch = config["training"].get("debug_first_val_batch", False)

        for epoch in range(model_config["epochs"]):
            model.train()
            total_loss = 0.0
            exact_matches = 0
            matching_bits = 0
            total_target_bits = 0

            for batch in train_loader:
                inputs, targets, input_lengths, target_lengths = batch
                inputs, targets, input_lengths, target_lengths = [
                    b.to(device)
                    for b in (inputs, targets, input_lengths, target_lengths)
                ]

                logits = model(inputs, input_lengths)
                log_probs = nn.functional.log_softmax(logits, dim=-1)

                log_probs = log_probs.permute(1, 0, 2)

                loss = criterion(
                    log_probs,
                    targets,
                    input_lengths,
                    target_lengths,
                )

                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                total_loss += loss.item()
                predictions = greedy_ctc_decode(logits, tokenizer)
                references = recover_target_strings(
                    targets.detach().cpu(),
                    target_lengths.detach().cpu(),
                    tokenizer,
                )
                batch_exact_matches, batch_matching_bits, batch_total_target_bits = (
                    compute_sequence_metrics(
                        predictions,
                        references,
                        target_bit_length,
                    )
                )
                exact_matches += batch_exact_matches
                matching_bits += batch_matching_bits
                total_target_bits += batch_total_target_bits

            train_loss = total_loss / max(len(train_loader), 1)
            train_bit_accuracy = matching_bits / max(total_target_bits, 1)
            train_exact_accuracy = exact_matches / max(len(train_loader.dataset), 1)
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                tokenizer,
                target_bit_length,
                debug_first_batch=debug_first_val_batch,
                debug_logger=log,
            )
            log(
                f"Epoch {epoch + 1}: "
                f"train_loss = {train_loss:.4f}, "
                f"train_bit_accuracy = {train_bit_accuracy:.4f}, "
                f"train_exact_accuracy = {train_exact_accuracy:.4f}, "
                f"val_loss = {val_metrics['loss']:.4f}, "
                f"val_bit_accuracy = {val_metrics['bit_accuracy']:.4f}, "
                f"val_exact_accuracy = {val_metrics['exact_accuracy']:.4f}"
            )
    except Exception as exc:
        log(f"Training failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        log_lines.append(f"Saved training log to: {log_path}")
        write_log_file(log_path, log_lines)
        print(f"Saved training log to: {log_path}")


if __name__ == "__main__":
    train()
