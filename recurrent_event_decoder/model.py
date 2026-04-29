import torch
import torch.nn as nn

from recurrent_event_decoder.config import RecurrentEventConfig


class EventWindowEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or RecurrentEventConfig()
        self.input_projection = nn.Linear(
            self.config.event_input_dim,
            self.config.hidden_dim,
        )
        conv_layers = []
        for _ in range(self.config.encoder_layers):
            conv_layers.extend(
                [
                    nn.Conv1d(
                        self.config.hidden_dim,
                        self.config.hidden_dim,
                        kernel_size=self.config.encoder_kernel_size,
                        stride=self.config.encoder_stride,
                        padding=self.config.encoder_kernel_size // 2,
                    ),
                    nn.GELU(),
                    nn.BatchNorm1d(self.config.hidden_dim),
                ]
            )
        self.temporal_encoder = nn.Sequential(*conv_layers)
        self.output_norm = nn.LayerNorm(self.config.hidden_dim)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, events):
        if events.ndim != 3:
            raise AssertionError("events must have shape [B, 7500, 4]")
        if events.shape[-1] != self.config.event_input_dim:
            raise AssertionError(
                f"Expected event dim {self.config.event_input_dim}, got {events.shape[-1]}"
            )

        x = self.input_projection(events)
        x = x.transpose(1, 2)
        x = self.temporal_encoder(x)
        x = x.mean(dim=-1)
        x = self.output_norm(x)
        return self.dropout(x)


class RecurrentEventDecoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or RecurrentEventConfig()
        self.event_encoder = EventWindowEncoder(self.config)
        self.metadata_encoder = nn.Sequential(
            nn.Linear(self.config.metadata_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
        )
        self.recurrent_cell = nn.GRUCell(
            self.config.hidden_dim * 2,
            self.config.hidden_dim,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
        )

    def initial_state(self, batch_size, device=None):
        return torch.zeros(batch_size, self.config.hidden_dim, device=device)

    def step(self, events, metadata, state):
        if metadata.ndim != 2 or metadata.shape[-1] != self.config.metadata_dim:
            raise AssertionError(
                f"metadata must have shape [B, {self.config.metadata_dim}]"
            )
        if state is None:
            state = self.initial_state(events.shape[0], device=events.device)

        event_embedding = self.event_encoder(events)
        metadata_embedding = self.metadata_encoder(metadata.to(dtype=events.dtype))
        recurrent_input = torch.cat([event_embedding, metadata_embedding], dim=-1)
        next_state = self.recurrent_cell(recurrent_input, state)
        output = self.output_head(next_state)
        return output, next_state

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
        for window_index in range(window_count):
            step_output, next_state = self.step(
                windows[:, window_index],
                metadata,
                state,
            )
            valid = valid_window_mask[:, window_index].unsqueeze(-1)
            state = torch.where(valid, next_state, state)
            outputs.append(step_output)

        return torch.stack(outputs, dim=1), state

    def forward(self, windows, metadata, valid_window_mask=None):
        outputs, _ = self.forward_chunk(
            windows,
            metadata,
            state=None,
            valid_window_mask=valid_window_mask,
        )
        return outputs
