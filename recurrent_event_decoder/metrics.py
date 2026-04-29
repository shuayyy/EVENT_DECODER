import torch

from recurrent_event_decoder.config import RecurrentEventConfig
from recurrent_event_decoder.losses import (
    select_cutoff_indices,
    window_progress_targets,
)
from recurrent_event_decoder.types import RecurrentEventPrediction


def decode_recurrent_output(output, config=None):
    config = config or RecurrentEventConfig()
    if output.shape[-1] != config.output_dim:
        raise AssertionError(f"Expected output dim {config.output_dim}, got {output.shape[-1]}")
    bits = (torch.sigmoid(output[..., : config.bit_count]) >= 0.5).to(dtype=torch.long)
    return {
        "bits": bits,
        "continue_value": output[..., config.continue_index],
    }


def select_final_outputs(outputs, valid_window_mask, config=None):
    config = config or RecurrentEventConfig()
    cutoff_indices = select_cutoff_indices(outputs, valid_window_mask, config)
    batch_indices = torch.arange(outputs.shape[0], device=outputs.device)
    return outputs[batch_indices, cutoff_indices], cutoff_indices


def compute_recurrent_metrics(outputs, targets, valid_window_mask, config=None, windows=None):
    config = config or RecurrentEventConfig()
    targets = targets.to(device=outputs.device)
    valid_window_mask = valid_window_mask.to(device=outputs.device, dtype=torch.bool)
    final_outputs, stop_steps = select_final_outputs(outputs, valid_window_mask, config)
    decoded = decode_recurrent_output(final_outputs, config)
    target_bits = targets[:, : config.bit_count].to(dtype=torch.long)
    bit_matches = decoded["bits"] == target_bits
    exact_matches = bit_matches.all(dim=1)
    valid_counts = valid_window_mask.to(dtype=torch.long).sum(dim=1)
    target_stop_steps = (valid_counts - 1).clamp_min(0)
    pred_continue = decoded["continue_value"] >= config.stop_threshold
    target_progress = None
    progress_mae = torch.tensor(0.0, device=outputs.device)
    if windows is not None:
        target_progress = window_progress_targets(
            windows.to(device=outputs.device),
            valid_window_mask,
            stop_steps,
            config,
        )
        progress_mae = (decoded["continue_value"] - target_progress).abs().mean()

    return {
        "bit_accuracy": bit_matches.float().mean().item(),
        "exact_accuracy": exact_matches.float().mean().item(),
        "continue_accuracy": (pred_continue == (stop_steps == target_stop_steps)).float().mean().item(),
        "premature_stop_rate": (stop_steps < target_stop_steps).float().mean().item(),
        "stop_step_mae": (stop_steps - target_stop_steps).abs().float().mean().item(),
        "progress_mae": progress_mae.item(),
        "avg_stop_step": stop_steps.float().mean().item(),
    }


def build_prediction(outputs, valid_window_mask, config=None):
    config = config or RecurrentEventConfig()
    final_outputs, stop_steps = select_final_outputs(outputs, valid_window_mask, config)
    decoded = decode_recurrent_output(final_outputs, config)
    return RecurrentEventPrediction(
        bits=decoded["bits"],
        continue_value=decoded["continue_value"],
        stop_step=stop_steps,
        raw_output=final_outputs,
    )
