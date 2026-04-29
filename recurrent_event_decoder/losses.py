import torch
import torch.nn as nn

from recurrent_event_decoder.config import RecurrentEventConfig


def align_valid_window_mask(outputs, valid_window_mask):
    if outputs.ndim != 3:
        raise AssertionError("outputs must have shape [B, windows, output_dim]")
    if valid_window_mask.ndim != 2 or valid_window_mask.shape[0] != outputs.shape[0]:
        raise AssertionError("valid_window_mask must have shape [B, windows]")
    return valid_window_mask[:, : outputs.shape[1]].to(
        device=outputs.device,
        dtype=torch.bool,
    )


def select_cutoff_indices(outputs, valid_window_mask, config=None):
    config = config or RecurrentEventConfig()
    valid_window_mask = align_valid_window_mask(outputs, valid_window_mask)
    valid_counts = valid_window_mask.to(dtype=torch.long).sum(dim=1)
    last_valid = (valid_counts - 1).clamp_min(0)

    stop_predictions = (
        outputs[..., config.continue_index].detach() >= config.stop_threshold
    ) & valid_window_mask
    has_stop = stop_predictions.any(dim=1)
    first_stop = stop_predictions.to(dtype=torch.long).argmax(dim=1)
    return torch.where(has_stop, first_stop, last_valid)


def window_progress_targets(windows, valid_window_mask, cutoff_indices, config=None):
    config = config or RecurrentEventConfig()
    if windows.ndim != 4:
        raise AssertionError("windows must have shape [B, windows, 7500, 4]")

    valid_window_mask = valid_window_mask[:, : windows.shape[1]].to(
        device=windows.device,
        dtype=torch.bool,
    )
    window_times = windows[..., 3].amax(dim=2)
    window_times = window_times.masked_fill(~valid_window_mask, 0.0)
    episode_duration = window_times.amax(dim=1).clamp_min(1e-6)

    batch_indices = torch.arange(windows.shape[0], device=windows.device)
    cutoff_elapsed = window_times[batch_indices, cutoff_indices.to(device=windows.device)]
    return (cutoff_elapsed / episode_duration).clamp(0.0, config.stop_threshold)


def cutoff_recurrent_event_vector_loss(
    outputs,
    targets,
    valid_window_mask,
    windows,
    config=None,
):
    config = config or RecurrentEventConfig()
    targets = targets.to(device=outputs.device, dtype=outputs.dtype)
    full_valid_window_mask = valid_window_mask.to(device=outputs.device, dtype=torch.bool)
    valid_window_mask = align_valid_window_mask(outputs, full_valid_window_mask)
    cutoff_indices = select_cutoff_indices(outputs, valid_window_mask, config)

    batch_indices = torch.arange(outputs.shape[0], device=outputs.device)
    selected_outputs = outputs[batch_indices, cutoff_indices]
    selected_targets = targets[batch_indices]
    progress_targets = window_progress_targets(
        windows.to(device=outputs.device, dtype=outputs.dtype),
        full_valid_window_mask,
        cutoff_indices,
        config,
    ).to(dtype=outputs.dtype)

    pos_weight = None
    if config.use_weighted_bce and config.bit_pos_weight is not None:
        pos_weight = torch.as_tensor(
            config.bit_pos_weight,
            device=outputs.device,
            dtype=outputs.dtype,
        )
    bit_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
        selected_outputs[:, : config.bit_count],
        selected_targets[:, : config.bit_count],
    )
    continue_loss = nn.SmoothL1Loss()(
        selected_outputs[:, config.continue_index],
        progress_targets,
    )

    return config.bit_loss_weight * bit_loss + config.continue_loss_weight * continue_loss


def recurrent_event_vector_loss(
    outputs,
    targets,
    valid_window_mask,
    config=None,
    continue_targets=None,
    windows=None,
):
    if windows is None:
        raise ValueError("windows are required to compute progress-based continue loss")
    return cutoff_recurrent_event_vector_loss(
        outputs,
        targets,
        valid_window_mask,
        windows,
        config,
    )


def final_recurrent_event_vector_loss(
    outputs,
    targets,
    final_step_mask,
    config=None,
    windows=None,
):
    if windows is None:
        raise ValueError("windows are required to compute progress-based continue loss")
    return cutoff_recurrent_event_vector_loss(
        outputs,
        targets,
        final_step_mask,
        windows,
        config,
    )
