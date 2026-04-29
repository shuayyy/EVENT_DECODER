import torch
import torch.nn as nn


def fixed_bits_loss(logits, targets, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    return criterion(
        logits.reshape(-1, 2),
        targets.reshape(-1),
    )


def decode_fixed_bits(logits):
    return logits.argmax(dim=-1).detach().cpu()


def bits_to_strings(bit_tensor):
    return ["".join(str(int(bit)) for bit in row.tolist()) for row in bit_tensor]


def compute_fixed_bits_metrics(predicted_bits, targets, target_bit_length):
    if predicted_bits.ndim != 2 or targets.ndim != 2:
        raise AssertionError("fixed_bits metrics expect [B, target_bit_length] tensors")
    if predicted_bits.shape[1] != target_bit_length or targets.shape[1] != target_bit_length:
        raise AssertionError(
            f"fixed_bits metrics expect target length {target_bit_length}, "
            f"got pred={predicted_bits.shape[1]} target={targets.shape[1]}"
        )
    if not torch.all((targets == 0) | (targets == 1)):
        raise AssertionError("fixed_bits targets must contain only 0/1 values")

    exact_matches = (predicted_bits == targets).all(dim=1).sum().item()
    matching_bits = (predicted_bits == targets).sum().item()
    total_target_bits = targets.numel()

    return exact_matches, matching_bits, total_target_bits


def aggregate_repetition_log_probs(rep_logits, rep_counts):
    rep_log_probs = torch.log_softmax(rep_logits, dim=-1)

    chunk_log_probs = []
    start = 0

    for rep_count in rep_counts:
        rep_count = int(rep_count)

        if rep_count <= 0:
            raise AssertionError("Each chunk must contain at least one repetition")

        end = start + rep_count
        group_log_probs = rep_log_probs[start:end]

        chunk_log_probs.append(
            torch.logsumexp(group_log_probs, dim=0)
            - torch.log(torch.tensor(float(rep_count), device=rep_logits.device))
        )

        start = end

    if start != rep_logits.shape[0]:
        raise AssertionError(
            f"rep_counts sum {start} does not match logits batch {rep_logits.shape[0]}"
        )

    return torch.stack(chunk_log_probs, dim=0)


def grouped_fixed_bits_loss(chunk_log_probs, targets, criterion=None):
    if criterion is None:
        criterion = nn.NLLLoss()

    return criterion(
        chunk_log_probs.reshape(-1, 2),
        targets.reshape(-1),
    )
