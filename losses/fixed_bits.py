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
