import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.formats import EVENT_DTYPE
from datasets.tokenizer import Tokenizer


class EventDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        use_filtered=True,
        excluded_samples=None,
        excluded_datastrings=None,
        include_start_end_bits=False,
        start_flag="",
        end_flag="",
    ):
        self.dataset_root = Path(dataset_root)
        self.use_filtered = use_filtered
        self.excluded_samples = set(excluded_samples or [])
        self.excluded_datastrings = set(excluded_datastrings or [])
        self.include_start_end_bits = include_start_end_bits
        self.start_flag = start_flag
        self.end_flag = end_flag
        self.tokenizer = Tokenizer()
        self.samples = self._load_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = self._read_event_file(sample["event_path"])
        bitstream = self._compose_bitstream(sample["transmitted_bits"])
        token = self.tokenizer.encode(bitstream)

        return x, token

    def _compose_bitstream(self, transmitted_bits):
        if not self.include_start_end_bits:
            return transmitted_bits
        return f"{self.start_flag}{transmitted_bits}{self.end_flag}"

    def _load_samples(self):
        master_path = self.dataset_root / "master.json"
        with master_path.open("r", encoding="utf-8") as handle:
            master = json.load(handle)

        event_path_key = "filtered_events_bin" if self.use_filtered else "incoming_events_bin"
        samples = []
        for sample in master["samples"]:
            if sample["sample_name"] in self.excluded_samples:
                continue
            if sample["datastring"] in self.excluded_datastrings:
                continue

            samples.append(
                {
                    "sample_name": sample["sample_name"],
                    "event_path": self.dataset_root / sample[event_path_key],
                    "transmitted_bits": sample["transmitted_bits"],
                }
            )

        return samples

    def _read_event_file(self, event_path):
        events = np.fromfile(event_path, dtype=EVENT_DTYPE)

        if events.size == 0:
            return torch.empty((0, 4), dtype=torch.float32)

        relative_time = events["witnessed_utc_ns"].astype(np.float64)
        relative_time = (relative_time - relative_time[0]) / 1e9

        features = np.column_stack(
            (
                events["x"].astype(np.float32),
                events["y"].astype(np.float32),
                events["polarity"].astype(np.float32),
                relative_time.astype(np.float32),
            )
        )

        return torch.from_numpy(features)


class ProcessedRepetitionDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        sequences=None,
        include_start_end_bits=False,
        start_flag="",
        end_flag="",
        return_metadata=False,
    ):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = self.manifest_path.parent
        self.tokenizer = Tokenizer()
        self.sequence_filter = set(sequences) if sequences is not None else None
        self.include_start_end_bits = include_start_end_bits
        self.start_flag = start_flag
        self.end_flag = end_flag
        self.return_metadata = return_metadata
        self.samples = self._load_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = torch.from_numpy(
            np.load(sample["processed_path"], allow_pickle=False)
        ).to(dtype=torch.float32)
        token = self.tokenizer.encode(
            self._compose_bitstream(sample["transmitted_bits"])
        )
        if not self.return_metadata:
            return x, token

        metadata = {
            "sequence": sample["sequence"],
            "source_sample_name": sample["source_sample_name"],
            "data_dir": sample["data_dir"],
            "chunk_dir": sample["chunk_dir"],
            "chunk_name": sample["chunk_name"],
            "repetition_name": sample["repetition_name"],
            "repetition_index": sample["repetition_index"],
            "vote_group_id": sample["source_sample_name"],
        }
        return x, token, metadata

    def _compose_bitstream(self, transmitted_bits):
        if not self.include_start_end_bits:
            return transmitted_bits
        return f"{self.start_flag}{transmitted_bits}{self.end_flag}"

    def _load_samples(self):
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        samples = []
        for sample in manifest["repetitions"]:
            if self.sequence_filter is not None and sample["sequence"] not in self.sequence_filter:
                continue

            samples.append(
                {
                    "sequence": sample["sequence"],
                    "source_sample_name": sample["source_sample_name"],
                    "data_dir": sample["data_dir"],
                    "chunk_dir": sample["chunk_dir"],
                    "chunk_name": sample["chunk_name"],
                    "repetition_name": sample["repetition_name"],
                    "repetition_index": sample["repetition_index"],
                    "processed_path": self.dataset_root / sample["processed_path"],
                    "transmitted_bits": sample["transmitted_bits"],
                }
            )

        return samples
