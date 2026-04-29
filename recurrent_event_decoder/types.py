from dataclasses import dataclass

import torch


@dataclass
class RecurrentEventSample:
    windows: torch.Tensor
    num_repetitions: torch.Tensor
    transmission_frequency: torch.Tensor
    target: torch.Tensor
    valid_window_mask: torch.Tensor
    source_id: str = ""


@dataclass
class RecurrentEventBatch:
    windows: torch.Tensor
    metadata: torch.Tensor
    targets: torch.Tensor
    valid_window_mask: torch.Tensor
    source_ids: list[str]


@dataclass
class RecurrentEventPrediction:
    bits: torch.Tensor
    continue_value: torch.Tensor
    stop_step: torch.Tensor
    raw_output: torch.Tensor
