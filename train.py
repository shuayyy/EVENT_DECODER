import random
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from datasets.collate import ctc_collate, fixed_bits_collate, grouped_fixed_bits_collate
from datasets.dataset import ProcessedRepetitionDataset, GroupedProcessedRepetitionDataset
from datasets.tokenizer import Tokenizer
from losses.ctc import (
    compute_sequence_metrics,
    ctc_prefix_beam_search_decode,
    ctc_loss,
    greedy_ctc_decode,
    recover_target_strings,
)
from losses.fixed_bits import (
    bits_to_strings,
    compute_fixed_bits_metrics,
    decode_fixed_bits,
    fixed_bits_loss,
    aggregate_repetition_log_probs,
    grouped_fixed_bits_loss,
)
from model.model import LSTMdecoder, MambaDecoder, TransformerDecoder
from log import TrainingLogger, format_epoch_log


CONFIG_PATH = Path("config/hyperparameter.yaml")
PAD_COUNT_BIT_LENGTH = 6


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_training_inputs(inputs):
    return inputs


def resolve_model_config(config):
    shared_model_config = dict(config["model"])
    model_type = shared_model_config["model_type"].strip().lower()
    per_model_config = config["models"][model_type]
    return model_type, {**shared_model_config, **per_model_config}


def build_model(model_type, model_config, output_mode="ctc", target_bit_length=70):
    if model_type == "lstm":
        return LSTMdecoder(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            output_dim=model_config["output_dim"],
            num_layers=model_config.get("num_layers", 1),
            bidirectional=model_config.get("bidirectional", False),
            dropout=model_config.get("dropout", 0.0),
            output_mode=output_mode,
            target_bit_length=target_bit_length,
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
            output_mode=output_mode,
            target_bit_length=target_bit_length,
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
            output_mode=output_mode,
            target_bit_length=target_bit_length,
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


def init_segment_metric_counts():
    return {
        "start_matches": 0,
        "start_total": 0,
        "end_matches": 0,
        "end_total": 0,
        "data_matches": 0,
        "data_total": 0,
        "padding_matches": 0,
        "padding_total": 0,
    }


def accumulate_segment_metric_counts(
    counts,
    predictions,
    targets,
    base_target_bit_length,
    include_start_end_bits,
    start_flag,
    end_flag,
):
    payload_bit_length = base_target_bit_length - PAD_COUNT_BIT_LENGTH
    start_length = len(start_flag) if include_start_end_bits else 0
    end_length = len(end_flag) if include_start_end_bits else 0
    body_offset = start_length

    for prediction, target in zip(predictions, targets):
        if include_start_end_bits:
            for index in range(start_length):
                pred_bit = prediction[index] if index < len(prediction) else None
                if pred_bit == target[index]:
                    counts["start_matches"] += 1
                counts["start_total"] += 1

        pad_count_start = body_offset + payload_bit_length
        pad_count_end = pad_count_start + PAD_COUNT_BIT_LENGTH
        pad_count_bits = target[pad_count_start:pad_count_end]
        pad_count = int(pad_count_bits, 2)
        real_data_length = max(0, min(payload_bit_length, payload_bit_length - pad_count))

        for index in range(body_offset, body_offset + real_data_length):
            pred_bit = prediction[index] if index < len(prediction) else None
            if pred_bit == target[index]:
                counts["data_matches"] += 1
            counts["data_total"] += 1

        padding_indices = list(range(body_offset + real_data_length, body_offset + payload_bit_length))
        padding_indices.extend(range(pad_count_start, pad_count_end))
        for index in padding_indices:
            pred_bit = prediction[index] if index < len(prediction) else None
            if pred_bit == target[index]:
                counts["padding_matches"] += 1
            counts["padding_total"] += 1

        if include_start_end_bits:
            end_start = body_offset + base_target_bit_length
            end_end = end_start + end_length
            for index in range(end_start, end_end):
                pred_bit = prediction[index] if index < len(prediction) else None
                if pred_bit == target[index]:
                    counts["end_matches"] += 1
                counts["end_total"] += 1


def finalize_segment_metric_accuracies(counts):
    metrics = {
        "data_accuracy": counts["data_matches"] / max(counts["data_total"], 1),
        "padding_accuracy": counts["padding_matches"] / max(counts["padding_total"], 1),
    }
    if counts["start_total"] > 0:
        metrics["start_bit_accuracy"] = counts["start_matches"] / counts["start_total"]
    if counts["end_total"] > 0:
        metrics["end_bit_accuracy"] = counts["end_matches"] / counts["end_total"]
    return metrics


def decode_ctc_predictions(logits, tokenizer, decode_config):
    method = decode_config.get("method", "greedy").strip().lower()
    if method == "greedy":
        return greedy_ctc_decode(logits, tokenizer)
    if method == "beam":
        return ctc_prefix_beam_search_decode(
            logits,
            tokenizer,
            beam_width=int(decode_config.get("beam_width", 25)),
            target_bit_length=int(decode_config["target_bit_length"]),
            length_penalty=float(decode_config.get("length_penalty", 2.0)),
            exact_length_preferred=bool(
                decode_config.get("exact_length_preferred", True)
            ),
        )
    raise ValueError(f"Unsupported ctc_decode.method: {method}")


def compute_voted_chunk_metrics(predictions, targets, metadata, target_bit_length):
    grouped = {}
    for prediction, target, sample_metadata in zip(predictions, targets, metadata):
        group_id = sample_metadata["vote_group_id"]
        if group_id not in grouped:
            grouped[group_id] = {
                "target": target,
                "votes": [[0, 0] for _ in range(target_bit_length)],
            }
        for bit_index, bit in enumerate(prediction[:target_bit_length]):
            if bit == "0":
                grouped[group_id]["votes"][bit_index][0] += 1
            elif bit == "1":
                grouped[group_id]["votes"][bit_index][1] += 1

    voted_predictions = []
    voted_targets = []
    for group in grouped.values():
        voted_bits = []
        for zero_votes, one_votes in group["votes"]:
            if zero_votes == 0 and one_votes == 0:
                continue
            voted_bits.append("1" if one_votes > zero_votes else "0")
        voted_predictions.append("".join(voted_bits))
        voted_targets.append(group["target"])

    exact_matches, matching_bits, total_target_bits = compute_sequence_metrics(
        voted_predictions,
        voted_targets,
        target_bit_length,
    )
    total_groups = max(len(voted_targets), 1)
    return {
        "voted_chunk_bit_accuracy": matching_bits / max(total_target_bits, 1),
        "voted_chunk_exact_accuracy": exact_matches / total_groups,
        "voted_chunk_count": len(voted_targets),
    }


def save_per_bit_accuracy_history_plot(per_bit_accuracy_history, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not per_bit_accuracy_history:
        return

    num_epochs = len(per_bit_accuracy_history)
    num_bits = len(per_bit_accuracy_history[0])
    epoch_positions = list(range(num_epochs))
    epoch_labels_step = max(1, num_epochs // 10)
    epoch_tick_positions = list(range(0, num_epochs, epoch_labels_step))
    if epoch_tick_positions[-1] != num_epochs - 1:
        epoch_tick_positions.append(num_epochs - 1)
    epoch_tick_labels = [str(position + 1) for position in epoch_tick_positions]

    bit_labels_step = max(1, num_bits // 12)
    bit_tick_positions = list(range(0, num_bits, bit_labels_step))
    if bit_tick_positions[-1] != num_bits - 1:
        bit_tick_positions.append(num_bits - 1)

    plt.figure(figsize=(14, 6))
    image = plt.imshow(
        per_bit_accuracy_history,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    plt.colorbar(image, label="Validation accuracy")
    plt.xlabel("Bit position")
    plt.ylabel("Epoch")
    plt.title("Per-bit validation accuracy across training")
    plt.xticks(bit_tick_positions)
    plt.yticks(epoch_tick_positions, epoch_tick_labels)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def grouped_fixed_bits_forward(model, batch, device, target_bit_length):
    inputs, targets, rep_mask, input_lengths, metadata = batch

    inputs = inputs.to(device)
    targets = targets.to(device)
    rep_mask = rep_mask.to(device)
    input_lengths = input_lengths.to(device)

    batch_size, max_repetitions, max_length, feature_dim = inputs.shape

    flat_inputs = inputs.reshape(
        batch_size * max_repetitions,
        max_length,
        feature_dim,
    )

    flat_lengths = input_lengths.reshape(batch_size * max_repetitions)
    flat_mask = rep_mask.reshape(batch_size * max_repetitions)

    valid_inputs = select_training_inputs(flat_inputs[flat_mask])
    valid_lengths = flat_lengths[flat_mask]

    rep_counts = rep_mask.sum(dim=1).detach().cpu().tolist()

    rep_logits = model(valid_inputs, valid_lengths)

    chunk_log_probs = aggregate_repetition_log_probs(
        rep_logits,
        rep_counts,
    )

    if chunk_log_probs.shape != (batch_size, target_bit_length, 2):
        raise AssertionError(
            f"Expected grouped logits shape {(batch_size, target_bit_length, 2)}, "
            f"got {tuple(chunk_log_probs.shape)}"
        )

    avg_repetitions = sum(rep_counts) / max(len(rep_counts), 1)

    return chunk_log_probs, targets, metadata, avg_repetitions


def evaluate(
    model,
    loader,
    criterion,
    device,
    task,
    target_bit_length,
    base_target_bit_length,
    include_start_end_bits,
    start_flag,
    end_flag,
    tokenizer=None,
    debug_first_batch=False,
    debug_logger=None,
    ctc_decode_config=None,
    group_repetitions=False,
):
    model.eval()
    total_loss = 0.0
    exact_matches = 0
    matching_bits = 0
    total_target_bits = 0
    total_prediction_length = 0
    total_reference_length = 0
    total_predictions = 0
    segment_metric_counts = init_segment_metric_counts()
    ctc_predictions_for_voting = []
    ctc_references_for_voting = []
    ctc_metadata_for_voting = []
    per_bit_correct = torch.zeros(target_bit_length, dtype=torch.long)
    per_bit_total = torch.zeros(target_bit_length, dtype=torch.long)

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch_metadata = None
            if task == "ctc":
                if len(batch) == 5:
                    inputs, targets, input_lengths, target_lengths = [
                        tensor.to(device) for tensor in batch[:4]
                    ]
                    batch_metadata = batch[4]
                else:
                    inputs, targets, input_lengths, target_lengths = [
                        tensor.to(device) for tensor in batch
                    ]
                    batch_metadata = None
            elif task == "fixed_bits":
                if group_repetitions:
                    logits, targets, batch_metadata, _ = grouped_fixed_bits_forward(
                        model,
                        batch,
                        device,
                        target_bit_length,
                    )
                    if targets.shape[1] != target_bit_length:
                        raise AssertionError(
                            f"fixed_bits mode expects target length {target_bit_length}, "
                            f"got {targets.shape[1]}"
                        )
                else:
                    if len(batch) == 4:
                        inputs, targets, input_lengths = [
                            tensor.to(device) for tensor in batch[:3]
                        ]
                    else:
                        inputs, targets, input_lengths = [
                            tensor.to(device) for tensor in batch
                        ]
                    if targets.shape[1] != target_bit_length:
                        raise AssertionError(
                            f"fixed_bits mode expects target length {target_bit_length}, "
                            f"got {targets.shape[1]}"
                        )
            else:
                raise ValueError(f"Unsupported training task: {task}")

            if task == "ctc":
                inputs = select_training_inputs(inputs)
                logits = model(inputs, input_lengths)
                loss = ctc_loss(
                    logits,
                    targets,
                    input_lengths,
                    target_lengths,
                    criterion,
                )
                predictions = decode_ctc_predictions(
                    logits,
                    tokenizer,
                    ctc_decode_config or {},
                )
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
                total_prediction_length += sum(len(pred) for pred in predictions)
                total_reference_length += sum(len(ref) for ref in references)
                total_predictions += len(predictions)
                if batch_metadata is not None:
                    ctc_predictions_for_voting.extend(predictions)
                    ctc_references_for_voting.extend(references)
                    ctc_metadata_for_voting.extend(batch_metadata)
            else:
                if group_repetitions:
                    loss = grouped_fixed_bits_loss(logits, targets, criterion)
                else:
                    inputs = select_training_inputs(inputs)
                    logits = model(inputs, input_lengths)
                    loss = fixed_bits_loss(logits, targets, criterion)
                predicted_bits = decode_fixed_bits(logits)
                references_bits = targets.detach().cpu()
                batch_exact_matches, batch_matching_bits, batch_total_target_bits = (
                    compute_fixed_bits_metrics(
                        predicted_bits,
                        references_bits,
                        target_bit_length,
                    )
                )
                bit_matches = predicted_bits == references_bits
                per_bit_correct += bit_matches.sum(dim=0)
                per_bit_total += torch.full(
                    (target_bit_length,),
                    bit_matches.shape[0],
                    dtype=torch.long,
                )
                predictions = bits_to_strings(predicted_bits)
                references = bits_to_strings(references_bits)

            total_loss += loss.item()
            exact_matches += batch_exact_matches
            matching_bits += batch_matching_bits
            total_target_bits += batch_total_target_bits
            accumulate_segment_metric_counts(
                segment_metric_counts,
                predictions,
                references,
                base_target_bit_length,
                include_start_end_bits,
                start_flag,
                end_flag,
            )

            if debug_first_batch and batch_index == 0 and predictions:
                for example_index in range(min(3, len(predictions))):
                    debug_message = (
                        "Validation debug: "
                        f"target={references[example_index]} "
                        f"pred={predictions[example_index]} "
                        f"pred_len={len(predictions[example_index])} "
                        f"target_len={len(references[example_index])}"
                    )
                    if debug_logger is not None:
                        debug_logger(debug_message)
                    else:
                        print(debug_message)

    metrics = {
        "loss": total_loss / max(len(loader), 1),
        "bit_accuracy": matching_bits / max(total_target_bits, 1),
        "exact_accuracy": exact_matches / max(len(loader.dataset), 1),
        "avg_pred_length": total_prediction_length / max(total_predictions, 1),
        "avg_target_length": total_reference_length / max(total_predictions, 1),
    }
    if task == "fixed_bits":
        metrics["per_bit_accuracy"] = (
            per_bit_correct.float() / per_bit_total.clamp_min(1).float()
        ).tolist()
    metrics.update(finalize_segment_metric_accuracies(segment_metric_counts))
    if task == "ctc" and ctc_predictions_for_voting and ctc_decode_config and ctc_decode_config.get(
        "use_repetition_voting",
        False,
    ):
        metrics.update(
            compute_voted_chunk_metrics(
                ctc_predictions_for_voting,
                ctc_references_for_voting,
                ctc_metadata_for_voting,
                target_bit_length,
            )
        )
    return metrics


def fixed_bits_error_report(
    model,
    loader,
    device,
    target_bit_length,
    group_repetitions=False,
):
    model.eval()

    total = torch.zeros(target_bit_length)
    wrong = torch.zeros(target_bit_length)

    zero_total = 0
    zero_wrong = 0
    one_total = 0
    one_wrong = 0

    pred_ones = 0
    target_ones = 0
    total_bits = 0

    with torch.no_grad():
        for batch in loader:
            if group_repetitions:
                logits, targets, _, _ = grouped_fixed_bits_forward(
                    model,
                    batch,
                    device,
                    target_bit_length,
                )
            else:
                inputs, targets, input_lengths = batch[:3]
                inputs = inputs.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)

                logits = model(select_training_inputs(inputs), input_lengths)
            preds = decode_fixed_bits(logits).to(device)

            mismatch = preds != targets

            wrong += mismatch.float().sum(dim=0).cpu()
            total += torch.ones_like(targets).float().sum(dim=0).cpu()

            zero_mask = targets == 0
            one_mask = targets == 1

            zero_total += zero_mask.sum().item()
            zero_wrong += (mismatch & zero_mask).sum().item()

            one_total += one_mask.sum().item()
            one_wrong += (mismatch & one_mask).sum().item()

            pred_ones += preds.sum().item()
            target_ones += targets.sum().item()
            total_bits += targets.numel()

    per_bit_error = wrong / total.clamp_min(1)

    worst_k = min(10, target_bit_length)
    worst_idx = torch.topk(per_bit_error, worst_k).indices.tolist()
    worst_str = ", ".join(
        f"{i}:{per_bit_error[i].item():.3f}" for i in worst_idx
    )

    return {
        "zero_error": zero_wrong / max(zero_total, 1),
        "one_error": one_wrong / max(one_total, 1),
        "pred_one_rate": pred_ones / max(total_bits, 1),
        "target_one_rate": target_ones / max(total_bits, 1),
        "worst_bits": worst_str,
    }


def train():
    config = load_config()
    task = config["training"].get("task", "ctc").strip().lower()
    group_repetitions = (
        bool(config["training"].get("group_repetitions", False))
        and task == "fixed_bits"
    )
    model_type, model_config = resolve_model_config(config)
    ctc_decode_config = dict(config.get("ctc_decode", {}))
    base_target_bit_length = int(config["data"]["target_bit_length"])
    include_start_end_bits = config["data"].get("include_start_end_bits", False)
    start_flag = config["data"].get("start_flag", "")
    end_flag = config["data"].get("end_flag", "")
    use_repetitions = config["data"].get("rep", True)
    target_bit_length = base_target_bit_length
    if include_start_end_bits:
        target_bit_length += len(start_flag) + len(end_flag)
    ctc_target_bit_length = int(
        ctc_decode_config.get("target_bit_length", base_target_bit_length)
    )
    if include_start_end_bits:
        ctc_target_bit_length += len(start_flag) + len(end_flag)
    ctc_decode_config["target_bit_length"] = ctc_target_bit_length
    if task == "ctc":
        collate_fn = ctc_collate
        output_mode = "ctc"
        model_config["output_dim"] = 3
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    elif task == "fixed_bits":
        if group_repetitions:
            collate_fn = partial(
                grouped_fixed_bits_collate,
                target_bit_length=target_bit_length,
            )
            criterion = nn.NLLLoss()
        else:
            collate_fn = partial(
                fixed_bits_collate,
                target_bit_length=target_bit_length,
            )
            criterion = nn.CrossEntropyLoss()

        output_mode = "fixed_bits"
        model_config["output_dim"] = 2
    else:
        raise ValueError(f"Unsupported training task: {task}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_model_runtime(model_config, device)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config["training"].get("log_dir", "outputs/logs"))
    log_path = log_dir / f"{run_timestamp}_{model_type}.txt"
    tokenizer = Tokenizer()
    logger = TrainingLogger(log_path)
    log = logger.log
    per_bit_accuracy_history = []

    try:
        manifest_path = Path(
            config["data"].get(
                "processed_manifest",
                "data/dataset_processed/manifest.json",
            )
        )
        dataset = ProcessedRepetitionDataset(
            manifest_path=manifest_path,
            rep=use_repetitions,
            include_start_end_bits=include_start_end_bits,
            start_flag=start_flag,
            end_flag=end_flag,
        )
        split_map = split_sequences(dataset.samples, config["split"])

        if group_repetitions:
            train_dataset = GroupedProcessedRepetitionDataset(
                manifest_path=manifest_path,
                sequences=split_map["train"],
                include_start_end_bits=include_start_end_bits,
                start_flag=start_flag,
                end_flag=end_flag,
            )

            val_dataset = GroupedProcessedRepetitionDataset(
                manifest_path=manifest_path,
                sequences=split_map["val"],
                include_start_end_bits=include_start_end_bits,
                start_flag=start_flag,
                end_flag=end_flag,
            )
        else:
            train_dataset = ProcessedRepetitionDataset(
                manifest_path=manifest_path,
                sequences=split_map["train"],
                rep=use_repetitions,
                include_start_end_bits=include_start_end_bits,
                start_flag=start_flag,
                end_flag=end_flag,
            )

            val_dataset = ProcessedRepetitionDataset(
                manifest_path=manifest_path,
                sequences=split_map["val"],
                rep=use_repetitions,
                include_start_end_bits=include_start_end_bits,
                start_flag=start_flag,
                end_flag=end_flag,
                return_metadata=task == "ctc"
                and ctc_decode_config.get("use_repetition_voting", False),
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=model_config["batch_size"],
            shuffle=config["training"]["shuffle"],
            collate_fn=collate_fn,
            num_workers=config["training"].get("num_workers", 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=model_config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config["training"].get("num_workers", 0),
        )

        model = build_model(
            model_type,
            model_config,
            output_mode=output_mode,
            target_bit_length=target_bit_length,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_config["learning_rate"],
            weight_decay=model_config.get("weight_decay", 0.0),
        )

        log(f"Run timestamp: {run_timestamp}")
        log(f"Using processed manifest: {manifest_path}")
        log(f"Using repetitions: {use_repetitions}")
        log(f"Task: {task}")
        log(f"Group repetitions: {group_repetitions}")
        log(f"Model type: {model_type}")
        log(f"Device: {device}")
        log(f"Model config: {model_config}")
        log(
            "Ground truth config: "
            f"base_target_bit_length={base_target_bit_length}, "
            f"include_start_end_bits={include_start_end_bits}, "
            f"effective_target_bit_length={target_bit_length}"
        )
        if task == "ctc":
            # CTC training is unchanged; beam search and voting only affect decoding/evaluation.
            # Length preference is used because the current target length is fixed per run.
            # Repetition voting is used because the same transmitted chunk appears multiple times.
            log(f"CTC decode config: {ctc_decode_config}")
        log(
            "Split summary: "
            f"train={{sequences: {len(split_map['train'])}, samples: {len(train_dataset)}}}, "
            f"val={{sequences: {len(split_map['val'])}, samples: {len(val_dataset)}}}"
        )

        grad_clip = model_config.get("grad_clip")
        debug_first_val_batch = config["training"].get("debug_first_val_batch", False)
        best_val_bit_accuracy = 0.0
        best_epoch = 0

        for epoch in range(model_config["epochs"]):
            model.train()
            total_loss = 0.0
            exact_matches = 0
            matching_bits = 0
            total_target_bits = 0
            train_segment_metric_counts = init_segment_metric_counts()

            for batch in train_loader:
                if task == "ctc":
                    inputs, targets, input_lengths, target_lengths = batch
                    inputs, targets, input_lengths, target_lengths = [
                        b.to(device)
                        for b in (inputs, targets, input_lengths, target_lengths)
                    ]
                elif group_repetitions:
                    logits, targets, _, _ = grouped_fixed_bits_forward(
                        model,
                        batch,
                        device,
                        target_bit_length,
                    )
                    if targets.shape[1] != target_bit_length:
                        raise AssertionError(
                            f"fixed_bits mode expects target length {target_bit_length}, "
                            f"got {targets.shape[1]}"
                        )
                else:
                    inputs, targets, input_lengths = batch
                    inputs, targets, input_lengths = [
                        b.to(device)
                        for b in (inputs, targets, input_lengths)
                    ]
                    if targets.shape[1] != target_bit_length:
                        raise AssertionError(
                            f"fixed_bits mode expects target length {target_bit_length}, "
                            f"got {targets.shape[1]}"
                        )

                if task == "ctc":
                    inputs = select_training_inputs(inputs)
                    logits = model(inputs, input_lengths)
                    loss = ctc_loss(
                        logits,
                        targets,
                        input_lengths,
                        target_lengths,
                        criterion,
                    )
                elif group_repetitions:
                    loss = grouped_fixed_bits_loss(logits, targets, criterion)
                else:
                    inputs = select_training_inputs(inputs)
                    logits = model(inputs, input_lengths)
                    loss = fixed_bits_loss(logits, targets, criterion)

                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                total_loss += loss.item()
                if task == "ctc":
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
                else:
                    predicted_bits = decode_fixed_bits(logits)
                    references_bits = targets.detach().cpu()
                    predictions = bits_to_strings(predicted_bits)
                    references = bits_to_strings(references_bits)
                    batch_exact_matches, batch_matching_bits, batch_total_target_bits = (
                        compute_fixed_bits_metrics(
                            predicted_bits,
                            references_bits,
                            target_bit_length,
                        )
                    )
                exact_matches += batch_exact_matches
                matching_bits += batch_matching_bits
                total_target_bits += batch_total_target_bits
                accumulate_segment_metric_counts(
                    train_segment_metric_counts,
                    predictions,
                    references,
                    base_target_bit_length,
                    include_start_end_bits,
                    start_flag,
                    end_flag,
                )

            train_loss = total_loss / max(len(train_loader), 1)
            train_bit_accuracy = matching_bits / max(total_target_bits, 1)
            train_exact_accuracy = exact_matches / max(len(train_loader.dataset), 1)
            train_segment_metrics = finalize_segment_metric_accuracies(
                train_segment_metric_counts
            )
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                task,
                target_bit_length,
                base_target_bit_length,
                include_start_end_bits,
                start_flag,
                end_flag,
                tokenizer=tokenizer if task == "ctc" else None,
                debug_first_batch=debug_first_val_batch,
                debug_logger=log,
                ctc_decode_config=ctc_decode_config if task == "ctc" else None,
                group_repetitions=group_repetitions,
            )
            if task == "fixed_bits" and "per_bit_accuracy" in val_metrics:
                per_bit_accuracy_history.append(val_metrics["per_bit_accuracy"])

            if val_metrics["bit_accuracy"] > best_val_bit_accuracy:
                best_val_bit_accuracy = val_metrics["bit_accuracy"]
                best_epoch = epoch + 1

            overfit_gap = train_bit_accuracy - val_metrics["bit_accuracy"]

            epoch_log = format_epoch_log(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_bit_accuracy=train_bit_accuracy,
                train_exact_accuracy=train_exact_accuracy,
                train_segment_metrics=train_segment_metrics,
                val_metrics=val_metrics,
                task=task,
                best_val_bit_accuracy=best_val_bit_accuracy,
                best_epoch=best_epoch,
                overfit_gap=overfit_gap,
            )
            log(epoch_log)

            if task == "fixed_bits" and (
                (epoch + 1) == 1
                or (epoch + 1) % 25 == 0
            ):
                train_diag = fixed_bits_error_report(
                    model,
                    train_loader,
                    device,
                    target_bit_length,
                    group_repetitions=group_repetitions,
                )
                val_diag = fixed_bits_error_report(
                    model,
                    val_loader,
                    device,
                    target_bit_length,
                    group_repetitions=group_repetitions,
                )
                log(
                    f"Fixed-bit diagnostics epoch {epoch + 1}: "
                    f"train_zero_error={train_diag['zero_error']:.4f}, "
                    f"train_one_error={train_diag['one_error']:.4f}, "
                    f"train_pred_one_rate={train_diag['pred_one_rate']:.4f}, "
                    f"train_target_one_rate={train_diag['target_one_rate']:.4f}, "
                    f"train_worst_bits=[{train_diag['worst_bits']}], "
                    f"val_zero_error={val_diag['zero_error']:.4f}, "
                    f"val_one_error={val_diag['one_error']:.4f}, "
                    f"val_pred_one_rate={val_diag['pred_one_rate']:.4f}, "
                    f"val_target_one_rate={val_diag['target_one_rate']:.4f}, "
                    f"val_worst_bits=[{val_diag['worst_bits']}]"
                )
    except KeyboardInterrupt:
        log("Training interrupted by user (Ctrl+C).")
        raise
    except Exception as exc:
        log(f"Training failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        if task == "fixed_bits" and per_bit_accuracy_history:
            plot_path = (
                Path("outputs/per_bit_accuracy")
                / f"{run_timestamp}_{model_type}_whole_training.png"
            )
            save_per_bit_accuracy_history_plot(
                per_bit_accuracy_history,
                plot_path,
            )
            logger.log(f"Saved per-bit accuracy plot: {plot_path}")
        logger.log(f"Saved training log to: {log_path}")
        logger.save(print_message=False)


if __name__ == "__main__":
    train()
