import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from recurrent_event_decoder.config import RecurrentEventConfig


class TemporalEncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, config=None):
        super().__init__()
        config = config or RecurrentEventConfig()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_dim,
                out_dim,
                kernel_size=config.encoder_kernel_size,
                stride=config.encoder_stride,
                padding=config.encoder_kernel_size // 2,
            ),
            nn.GELU(),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.block(x)


class VectorBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


def build_fully_connected_stack(in_dim, hidden_dim, layer_count, dropout):
    if layer_count <= 0:
        return nn.Identity(), in_dim

    layers = []
    current_dim = in_dim
    for _ in range(layer_count):
        layers.extend(
            [
                nn.LayerNorm(current_dim),
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        current_dim = hidden_dim
    return nn.Sequential(*layers), current_dim


class SummaryWindowEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or RecurrentEventConfig()
        self.event_projection = nn.Sequential(
            nn.Linear(self.config.event_input_dim, self.config.event_summary_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.event_summary_hidden_dim, self.config.temp_encoder_3_dim),
            nn.GELU(),
        )
        pooled_dim = self.config.temp_encoder_3_dim * 2
        raw_stats_dim = self.config.event_input_dim * 4
        self.output_projection = nn.Sequential(
            nn.LayerNorm(pooled_dim + raw_stats_dim),
            nn.Linear(pooled_dim + raw_stats_dim, self.config.temp_encoder_3_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

    def project_events(self, events):
        return self.event_projection(events)

    def forward(self, events):
        if events.ndim != 3:
            raise AssertionError("events must have shape [B, 7500, 4]")
        if events.shape[-1] != self.config.event_input_dim:
            raise AssertionError(
                f"Expected event dim {self.config.event_input_dim}, got {events.shape[-1]}"
            )

        projected = self.project_events(events)
        learned_mean = projected.mean(dim=1)
        learned_max = projected.amax(dim=1)
        raw_mean = events.mean(dim=1)
        raw_std = events.std(dim=1, unbiased=False)
        raw_min = events.amin(dim=1)
        raw_max = events.amax(dim=1)
        summary = torch.cat(
            [
                learned_mean,
                learned_max,
                raw_mean,
                raw_std,
                raw_min,
                raw_max,
            ],
            dim=-1,
        )
        return self.output_projection(summary)


class RecurrentEventDecoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or RecurrentEventConfig()
        self.window_encoder_type = getattr(self.config, "window_encoder_type", "conv")
        if self.window_encoder_type == "conv":
            self.temp_encoder_1 = TemporalEncoderBlock(
                self.config.event_input_dim,
                self.config.temp_encoder_1_dim,
                self.config,
            )
            self.temp_encoder_2 = TemporalEncoderBlock(
                self.config.temp_encoder_1_dim,
                self.config.temp_encoder_2_dim,
                self.config,
            )
            self.temp_encoder_3 = TemporalEncoderBlock(
                self.config.temp_encoder_2_dim,
                self.config.temp_encoder_3_dim,
                self.config,
            )
        elif self.window_encoder_type == "summary":
            self.window_summary_encoder = SummaryWindowEncoder(self.config)
        else:
            raise ValueError(
                "window_encoder_type must be either 'summary' or 'conv', "
                f"got {self.window_encoder_type!r}"
            )
        self.recurrent_1 = nn.LSTMCell(
            self.config.temp_encoder_3_dim + self.config.metadata_dim,
            self.config.recurrent_1_dim,
        )
        self.temp_encoder_4 = VectorBlock(
            self.config.recurrent_1_dim,
            self.config.temp_encoder_4_dim,
            self.config.dropout,
        )
        self.linear = nn.Linear(self.config.temp_encoder_4_dim, self.config.linear_dim)
        self.temp_decoder = VectorBlock(
            self.config.linear_dim,
            self.config.temp_decoder_dim,
            self.config.dropout,
        )
        self.recurrent_2 = nn.LSTMCell(
            self.config.temp_decoder_dim,
            self.config.recurrent_2_dim,
        )
        self.post_recurrent_fc, output_input_dim = build_fully_connected_stack(
            self.config.recurrent_2_dim,
            self.config.output_fc_dim,
            self.config.output_fc_layers,
            self.config.dropout,
        )
        self.output_decoder = nn.Sequential(
            nn.LayerNorm(output_input_dim),
            nn.Dropout(self.config.dropout),
            nn.Linear(output_input_dim, self.config.output_dim),
        )

    def initial_lstm_state(self, batch_size, hidden_dim, device=None):
        hidden = torch.zeros(batch_size, hidden_dim, device=device)
        cell = torch.zeros(batch_size, hidden_dim, device=device)
        return hidden, cell

    def initial_state(self, batch_size, device=None):
        return (
            self.initial_lstm_state(batch_size, self.config.recurrent_1_dim, device),
            self.initial_lstm_state(batch_size, self.config.recurrent_2_dim, device),
        )

    def normalize_metadata(self, metadata, dtype):
        metadata = metadata.to(dtype=dtype)
        if not self.config.normalize_inputs:
            return metadata

        metadata = metadata.clone()
        metadata[:, 0] = metadata[:, 0] / float(self.config.repetition_scale)
        metadata[:, 1] = metadata[:, 1] / float(self.config.frequency_scale)
        return metadata

    def encode_events_with_conv(self, events):
        if events.ndim != 3:
            raise AssertionError("events must have shape [B, 7500, 4]")
        if events.shape[-1] != self.config.event_input_dim:
            raise AssertionError(
                f"Expected event dim {self.config.event_input_dim}, got {events.shape[-1]}"
            )

        def run_block(block, value):
            if (
                self.config.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                return checkpoint(block, value, use_reentrant=False)
            return block(value)

        x = events.transpose(1, 2)
        x = run_block(self.temp_encoder_1, x)
        x = run_block(self.temp_encoder_2, x)
        x = run_block(self.temp_encoder_3, x)
        return x.mean(dim=-1)

    def encode_events(self, events):
        if self.window_encoder_type == "conv":
            return self.encode_events_with_conv(events)
        if (
            self.config.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(self.window_summary_encoder, events, use_reentrant=False)
        return self.window_summary_encoder(events)

    def step(self, events, metadata, state):
        if metadata.ndim != 2 or metadata.shape[-1] != self.config.metadata_dim:
            raise AssertionError(
                f"metadata must have shape [B, {self.config.metadata_dim}]"
            )
        if state is None:
            state = self.initial_state(events.shape[0], device=events.device)

        state_1, state_2 = state
        event_embedding = self.encode_events(events)
        metadata = self.normalize_metadata(metadata, events.dtype)
        recurrent_1_input = torch.cat([event_embedding, metadata], dim=-1)
        next_state_1 = self.recurrent_1(recurrent_1_input, state_1)

        hidden_1, _ = next_state_1
        x = self.temp_encoder_4(hidden_1)
        x = self.linear(x)
        x = self.temp_decoder(x)
        next_state_2 = self.recurrent_2(x, state_2)

        hidden_2, _ = next_state_2
        x = self.post_recurrent_fc(hidden_2)
        output = self.output_decoder(x)
        return output, (next_state_1, next_state_2)

    def mask_state(self, next_state, current_state, valid):
        valid = valid.unsqueeze(-1)
        masked_state = []
        for next_lstm_state, current_lstm_state in zip(next_state, current_state):
            masked_lstm_state = tuple(
                torch.where(valid, next_part, current_part)
                for next_part, current_part in zip(next_lstm_state, current_lstm_state)
            )
            masked_state.append(masked_lstm_state)
        return tuple(masked_state)

    def forward_chunk(self, windows, metadata, state=None, valid_window_mask=None):
        if windows.ndim != 4:
            raise AssertionError("windows must have shape [B, W, 7500, 4]")
        if windows.shape[-2:] != (
            self.config.event_window_size,
            self.config.event_input_dim,
        ):
            raise AssertionError(
                "Expected window shape "
                f"[{self.config.event_window_size}, {self.config.event_input_dim}], "
                f"got {tuple(windows.shape[-2:])}"
            )

        batch_size, window_count = windows.shape[:2]
        if valid_window_mask is None:
            valid_window_mask = torch.ones(
                batch_size,
                window_count,
                dtype=torch.bool,
                device=windows.device,
            )
        else:
            valid_window_mask = valid_window_mask.to(
                device=windows.device,
                dtype=torch.bool,
            )

        if state is None:
            state = self.initial_state(batch_size, device=windows.device)
        outputs = []
        active = torch.ones(batch_size, dtype=torch.bool, device=windows.device)
        for window_index in range(window_count):
            if self.config.stop_training_on_continue and not active.any():
                break

            step_output, next_state = self.step(
                windows[:, window_index],
                metadata,
                state,
            )
            valid = valid_window_mask[:, window_index] & active
            state = self.mask_state(next_state, state, valid)
            outputs.append(step_output)

            if self.config.stop_training_on_continue:
                should_stop = step_output[:, self.config.continue_index].detach()
                active = active & valid_window_mask[:, window_index]
                active = active & (should_stop < self.config.stop_threshold)

        return torch.stack(outputs, dim=1), state

    def forward(self, windows, metadata, valid_window_mask=None):
        outputs, _ = self.forward_chunk(
            windows,
            metadata,
            state=None,
            valid_window_mask=valid_window_mask,
        )
        return outputs
