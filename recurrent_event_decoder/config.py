from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecurrentEventConfig:
    event_window_size: int = 7500
    event_input_dim: int = 4
    output_dim: int = 71
    bit_count: int = 70
    continue_index: int = 70
    stop_threshold: float = 1.0
    stop_training_on_continue: bool = True
    hidden_dim: int = 256
    metadata_dim: int = 2
    window_encoder_type: str = "summary"
    event_summary_hidden_dim: int = 64
    temp_encoder_1_dim: int = 512
    temp_encoder_2_dim: int = 128
    temp_encoder_3_dim: int = 64
    recurrent_1_dim: int = 80
    temp_encoder_4_dim: int = 80
    linear_dim: int = 80
    temp_decoder_dim: int = 80
    recurrent_2_dim: int = 128
    output_fc_layers: int = 2
    output_fc_dim: int = 128
    activation_checkpointing: bool = False
    use_amp: bool = False
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
    use_weighted_bce: bool = True
    bit_pos_weight: Any = None
    normalize_inputs: bool = True
    sensor_width: float = 1280.0
    sensor_height: float = 720.0
    time_scale_us: float = 1_000_000.0
    repetition_scale: float = 15.0
    frequency_scale: float = 1000.0
    max_windows: int | None = None
    prepared_manifest: Path = Path("data/recurrent_event_processed/manifest.json")
