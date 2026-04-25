import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.tokenizer import Tokenizer


class EventDataset(Dataset):
    def __init__(self, events_dir, labels_dir):
        self.events_dir = Path(events_dir)
        self.labels_dir = Path(labels_dir)
        self.event_files = sorted(self.events_dir.glob("*.csv"))
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.event_files)

    def __getitem__(self, idx):
        event_path = self.event_files[idx]
        sample_id = event_path.stem
        label_path = self.labels_dir / f"{sample_id}.json"

        events_df = pd.read_csv(event_path)
        x = torch.tensor(
            events_df[["x", "y", "value", "timestamp"]].values,
            dtype=torch.float32,
        )

        with label_path.open("r", encoding="utf-8") as f:
            label_data = json.load(f)

        bitstream = label_data["bitstream"]
        token = self.tokenizer.encode(bitstream)

        return x, token
