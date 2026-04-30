import torch

from recurrent_event_decoder.config import RecurrentEventConfig
from recurrent_event_decoder.metrics import build_prediction


def run_recurrent_inference(model, windows, metadata, config=None):
    config = config or RecurrentEventConfig()
    model.eval()
    if windows.ndim == 3:
        windows = windows.unsqueeze(0)
    if metadata.ndim == 1:
        metadata = metadata.unsqueeze(0)

    outputs = []
    valid_mask = torch.ones(
        windows.shape[0],
        windows.shape[1],
        dtype=torch.bool,
        device=windows.device,
    )
    state = model.initial_state(windows.shape[0], device=windows.device)

    with torch.no_grad():
        for window_index in range(windows.shape[1]):
            output, state = model.step(windows[:, window_index], metadata, state)
            outputs.append(output)
            if (output[:, config.continue_index] >= config.stop_threshold).all():
                valid_mask[:, window_index + 1 :] = False
                break

    stacked_outputs = torch.stack(outputs, dim=1)
    valid_mask = valid_mask[:, : stacked_outputs.shape[1]]
    return build_prediction(stacked_outputs, valid_mask, config)
