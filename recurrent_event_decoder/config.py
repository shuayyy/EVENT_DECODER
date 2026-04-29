from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecurrentEventConfig:
    event_window_size: int = 7500
    event_input_dim: int = 4
    output_dim: int = 72
    bit_count: int = 70
    continue_index: int = 70
    time_index: int = 71
    stop_threshold: float = 1.0
    stop_training_on_continue: bool = False
    hidden_dim: int = 256
    metadata_dim: int = 2
    encoder_layers: int = 3
    encoder_kernel_size: int = 7
    encoder_stride: int = 4
    dropout: float = 0.1
    batch_size: int = 4
    epochs: int = 5
    learning_rate: float = 0.0003
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    bit_loss_weight: float = 1.0
    continue_loss_weight: float = 1.0
    time_loss_weight: float = 0.1
    max_windows: int | None = None
    prepared_manifest: Path = Path("data/recurrent_event_processed/manifest.json")
