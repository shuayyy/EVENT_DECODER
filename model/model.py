import math

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def pool_sequence_for_fixed_bits(hidden_states, output_mode, sequence_pool):
    if output_mode != "fixed_bits":
        return hidden_states
    return sequence_pool(hidden_states.transpose(1, 2)).transpose(1, 2)


class LSTMdecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=1,
        bidirectional=False,
        dropout=0.0,
        output_mode="ctc",
        target_bit_length=70,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.sequence_pool = (
            nn.AdaptiveAvgPool1d(target_bit_length)
            if output_mode == "fixed_bits"
            else None
        )

        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.temporal_model = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.decoder = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x, input_lengths=None):
        x = self.encoder(x)
        # `_` means ignore the LSTM hidden and cell states, since nn.LSTM returns `output, (hidden_state, cell_state)`.
        x, _ = self.temporal_model(x)
        x = pool_sequence_for_fixed_bits(
            x,
            self.output_mode,
            self.sequence_pool,
        )
        x = self.decoder(x)

        return x

class MambaDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=1,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
        output_mode="ctc",
        target_bit_length=70,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.sequence_pool = (
            nn.AdaptiveAvgPool1d(target_bit_length)
            if output_mode == "fixed_bits"
            else None
        )

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.temporal_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=hidden_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, input_lengths=None):
        x = self.encoder(x)
        for layer, layer_norm in zip(self.temporal_layers, self.layer_norms):
            residual = x
            x = layer(layer_norm(x))
            x = residual + self.dropout(x)
        x = self.output_norm(x)
        x = pool_sequence_for_fixed_bits(
            x,
            self.output_mode,
            self.sequence_pool,
        )
        x = self.decoder(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=1,
        nhead=4,
        ff_dim=None,
        dropout=0.0,
        use_positional_encoding=True,
        max_len=5000,
        output_mode="ctc",
        target_bit_length=70,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.sequence_pool = (
            nn.AdaptiveAvgPool1d(target_bit_length)
            if output_mode == "fixed_bits"
            else None
        )

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = (
            PositionalEncoding(hidden_dim, dropout=dropout, max_len=max_len)
            if use_positional_encoding
            else nn.Identity()
        )
        self.temporal_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=ff_dim if ff_dim is not None else hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, input_lengths=None):
        x = self.encoder(x)
        x = self.positional_encoding(x)
        x = self.temporal_model(x)

        x = pool_sequence_for_fixed_bits(
            x,
            self.output_mode,
            self.sequence_pool,
        )
        x = self.decoder(x)
        return x
