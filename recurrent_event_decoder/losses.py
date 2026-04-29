import torch
import torch.nn as nn

from recurrent_event_decoder.config import RecurrentEventConfig


def recurrent_step_mask(outputs, valid_window_mask, config=None):
    config = config or RecurrentEventConfig()
    if outputs.ndim != 3:
        raise AssertionError("outputs must have shape [B, windows, 72]")
    if valid_window_mask.shape != outputs.shape[:2]:
        raise AssertionError("valid_window_mask must have shape [B, windows]")

    valid_window_mask = valid_window_mask.to(dtype=torch.bool, device=outputs.device)
    if not config.stop_training_on_continue:
        return valid_window_mask

    continue_values = outputs[..., config.continue_index].detach()
    stop_predictions = (continue_values >= config.stop_threshold) & valid_window_mask
    stop_seen = stop_predictions.cumsum(dim=1) > 0
    after_first_stop = stop_seen & ~stop_predictions

    return valid_window_mask & ~after_first_stop


def build_continue_targets(targets, valid_window_mask, config=None):
    config = config or RecurrentEventConfig()
    continue_targets = torch.zeros(
        valid_window_mask.shape,
        dtype=targets.dtype,
        device=targets.device,
    )
    valid_counts = valid_window_mask.to(dtype=torch.long).sum(dim=1)
    final_indices = (valid_counts - 1).clamp_min(0)
    batch_indices = torch.arange(targets.shape[0], device=targets.device)
    final_continue_values = targets[:, config.continue_index]
    continue_targets[batch_indices, final_indices] = final_continue_values
    return continue_targets


def recurrent_event_vector_loss(
    outputs,
    targets,
    valid_window_mask,
    config=None,
    continue_targets=None,
):
    config = config or RecurrentEventConfig()
    targets = targets.to(device=outputs.device, dtype=outputs.dtype)
    valid_window_mask = valid_window_mask.to(device=outputs.device, dtype=torch.bool)
    train_step_mask = recurrent_step_mask(outputs, valid_window_mask, config)
    if not train_step_mask.any():
        return outputs.sum() * 0.0

    expanded_targets = targets.unsqueeze(1).expand_as(outputs)
    selected_outputs = outputs[train_step_mask]
    selected_targets = expanded_targets[train_step_mask]
    if continue_targets is None:
        continue_targets = build_continue_targets(targets, valid_window_mask, config)
    else:
        continue_targets = continue_targets.to(device=outputs.device, dtype=outputs.dtype)
    selected_continue_targets = (
        continue_targets[train_step_mask] >= config.stop_threshold
    ).to(dtype=selected_outputs.dtype)

    bit_loss = nn.BCEWithLogitsLoss()(
        selected_outputs[:, : config.bit_count],
        selected_targets[:, : config.bit_count],
    )
    continue_loss = nn.BCEWithLogitsLoss()(
        selected_outputs[:, config.continue_index],
        selected_continue_targets,
    )
    time_loss = nn.SmoothL1Loss()(
        selected_outputs[:, config.time_index],
        selected_targets[:, config.time_index],
    )

    return (
        config.bit_loss_weight * bit_loss
        + config.continue_loss_weight * continue_loss
        + config.time_loss_weight * time_loss
    )


def final_recurrent_event_vector_loss(
    outputs,
    targets,
    final_step_mask,
    config=None,
):
    config = config or RecurrentEventConfig()
    if outputs.ndim != 3:
        raise AssertionError("outputs must have shape [B, windows, 72]")
    if final_step_mask.shape != outputs.shape[:2]:
        raise AssertionError("final_step_mask must have shape [B, windows]")

    targets = targets.to(device=outputs.device, dtype=outputs.dtype)
    final_step_mask = final_step_mask.to(device=outputs.device, dtype=torch.bool)
    if not final_step_mask.any():
        return None

    selected_outputs = outputs[final_step_mask]
    batch_indices = torch.arange(outputs.shape[0], device=outputs.device)
    selected_batch_indices = batch_indices.unsqueeze(1).expand_as(final_step_mask)[
        final_step_mask
    ]
    selected_targets = targets[selected_batch_indices]

    bit_loss = nn.BCEWithLogitsLoss()(
        selected_outputs[:, : config.bit_count],
        selected_targets[:, : config.bit_count],
    )
    continue_targets = (
        selected_targets[:, config.continue_index] >= config.stop_threshold
    ).to(dtype=selected_outputs.dtype)
    continue_loss = nn.BCEWithLogitsLoss()(
        selected_outputs[:, config.continue_index],
        continue_targets,
    )
    time_loss = nn.SmoothL1Loss()(
        selected_outputs[:, config.time_index],
        selected_targets[:, config.time_index],
    )

    return (
        config.bit_loss_weight * bit_loss
        + config.continue_loss_weight * continue_loss
        + config.time_loss_weight * time_loss
    )
