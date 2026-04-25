import torch
import torch.nn as nn
from mamba_ssm import Mamba


class LSTMdecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.temporal_model = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, input_lengths=None):
        x = self.encoder(x)
        # `_` means ignore the LSTM hidden and cell states, since nn.LSTM returns `output, (hidden_state, cell_state)`.
        x, _ = self.temporal_model(x)
        x = self.decoder(x)

        return x

class MambaDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.temporal_model = Mamba(
            d_model=hidden_dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, input_lengths=None):
        x = self.encoder(x)
        x = self.temporal_model(x)
        x = self.decoder(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.temporal_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            ),
            num_layers=1,
        )

        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, input_lengths=None):
        x = self.encoder(x)

        if input_lengths is not None:
            B, T, _ = x.shape
            padding_mask = (
                torch.arange(T, device=x.device).unsqueeze(0)
                >= input_lengths.unsqueeze(1)
            )
        else:
            padding_mask = None

        x = self.temporal_model(
            x,
            src_key_padding_mask=padding_mask,
        )

        x = self.decoder(x)
        return x
