from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from recurrent_event_decoder.config import RecurrentEventConfig
from recurrent_event_decoder.types import RecurrentEventSample


class RecurrentEventDataset(Dataset):
    def __init__(self, dataset_root, config=None):
        self.dataset_root = Path(dataset_root)
        self.config = config or RecurrentEventConfig()
        self.samples = self.discover_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.load_sample(self.samples[index])

    def discover_samples(self):
        raise NotImplementedError("Raw recurrent event data is not available yet.")

    def load_sample(self, sample_ref) -> RecurrentEventSample:
        raise NotImplementedError("Implement once the raw file format is available.")

    def load_event_windows(self, sample_ref):
        raise NotImplementedError("Return [windows, 7500, 4] event tensors.")

    def load_target_vector(self, sample_ref):
        raise NotImplementedError("Return [71] target vector.")


class PreparedRecurrentEventDataset(Dataset):
    def __init__(self, manifest_path, config=None):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = self.manifest_path.parent
        self.config = config or RecurrentEventConfig()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.samples = manifest["episodes"]

    def __len__(self):
        return len(self.samples)

    def normalize_windows(self, windows):
        if not self.config.normalize_inputs:
            return windows

        windows = windows.clone()
        windows[..., 0] = windows[..., 0] / float(self.config.sensor_width)
        windows[..., 1] = windows[..., 1] / float(self.config.sensor_height)
        windows[..., 3] = windows[..., 3] / float(self.config.time_scale_us)
        return windows

    def __getitem__(self, index):
        sample = self.samples[index]
        windows = torch.from_numpy(
            np.load(self.dataset_root / sample["windows_path"], allow_pickle=False)
        ).to(dtype=torch.float32)
        if self.config.max_windows is not None:
            windows = windows[: self.config.max_windows]
        windows = self.normalize_windows(windows)
        target = torch.tensor(sample["target"], dtype=torch.float32)
        if target.numel() < self.config.output_dim:
            raise ValueError(
                f"{sample['sample_name']} target has {target.numel()} values; "
                f"expected at least {self.config.output_dim}"
            )
        target = target[: self.config.output_dim]
        valid_window_mask = torch.ones(windows.shape[0], dtype=torch.bool)

        return RecurrentEventSample(
            windows=windows,
            num_repetitions=torch.tensor(float(sample["repetitions"])),
            transmission_frequency=torch.tensor(float(sample["frequency"])),
            target=target,
            valid_window_mask=valid_window_mask,
            source_id=sample["sample_name"],
        )


class SyntheticRecurrentEventDataset(Dataset):
    def __init__(self, num_samples=32, config=None, seed=42):
        self.num_samples = num_samples
        self.config = config or RecurrentEventConfig()
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        max_windows = self.config.max_windows or 4
        window_count = int(
            torch.randint(1, max_windows + 1, (1,), generator=self.generator).item()
        )
        windows = torch.randn(
            window_count,
            self.config.event_window_size,
            self.config.event_input_dim,
            generator=self.generator,
        )
        target = torch.zeros(self.config.output_dim, dtype=torch.float32)
        target[: self.config.bit_count] = torch.randint(
            0,
            2,
            (self.config.bit_count,),
            generator=self.generator,
        ).to(dtype=torch.float32)
        target[self.config.continue_index] = self.config.stop_threshold

        return RecurrentEventSample(
            windows=windows,
            num_repetitions=torch.tensor(float(window_count)),
            transmission_frequency=torch.tensor(1_000.0),
            target=target,
            valid_window_mask=torch.ones(window_count, dtype=torch.bool),
            source_id=f"synthetic_{index:06d}",
        )
