import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from datasets.formats import EVENT_DTYPE


DEFAULT_CONFIG_PATH = Path("config/hyperparameter.yaml")
DEFAULT_OUTPUT_ROOT = Path("data/dataset_processed")
FEATURE_NAMES = [
    "log_pos",
    "log_neg",
    "log_total",
    "polarity_ratio",
    "delta_log_total",
    "delta_polarity_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a repetition-level processed dataset from chunk .bin files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where dataset_processed will be written.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=1750,
        help="Number of equal-time bins per repetition.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )
    return parser.parse_args()


def load_config(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_ordered_repetition_windows(chunk_info):
    transmission_times = chunk_info["timing"]["transmission_times"]
    ordered_keys = sorted(
        transmission_times.keys(),
        key=lambda item: int(item.split("_")[-1]),
    )

    windows = []
    for rep_index, key in enumerate(ordered_keys, start=1):
        timing = transmission_times[key]
        start_ns = int(timing["first_bit_time_micro"]) * 1000
        end_ns = int(timing["last_bit_time_micro"]) * 1000
        windows.append(
            {
                "repetition_name": f"rep_{rep_index:03d}",
                "repetition_index": rep_index,
                "source_key": key,
                "start_ns": start_ns,
                "end_ns": end_ns,
            }
        )

    return windows


def summarize_repetition(rep_events, start_ns, end_ns, num_bins):
    features = np.zeros((num_bins, len(FEATURE_NAMES)), dtype=np.float32)

    total_duration_ns = end_ns - start_ns
    if total_duration_ns <= 0:
        return features

    pos_counts = np.zeros(num_bins, dtype=np.float32)
    neg_counts = np.zeros(num_bins, dtype=np.float32)

    if rep_events.size != 0:
        elapsed_ns = rep_events["witnessed_utc_ns"].astype(np.int64) - int(start_ns)
        bin_indices = np.floor_divide(
            elapsed_ns * num_bins,
            total_duration_ns,
        )
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        polarity = rep_events["polarity"].astype(np.int64)

        pos_counts = np.bincount(
            bin_indices[polarity == 1],
            minlength=num_bins,
        ).astype(np.float32)
        neg_counts = np.bincount(
            bin_indices[polarity != 1],
            minlength=num_bins,
        ).astype(np.float32)

    log_pos = np.log1p(pos_counts)
    log_neg = np.log1p(neg_counts)
    total_counts = pos_counts + neg_counts
    log_total = np.log1p(total_counts)
    polarity_ratio = (pos_counts - neg_counts) / (total_counts + 1e-6)
    delta_log_total = np.zeros_like(log_total)
    delta_polarity_ratio = np.zeros_like(polarity_ratio)
    delta_log_total[1:] = log_total[1:] - log_total[:-1]
    delta_polarity_ratio[1:] = polarity_ratio[1:] - polarity_ratio[:-1]

    features[:, 0] = log_pos
    features[:, 1] = log_neg
    features[:, 2] = log_total
    features[:, 3] = polarity_ratio
    features[:, 4] = delta_log_total
    features[:, 5] = delta_polarity_ratio

    return features


def build_processed_dataset(config, output_root, num_bins, limit_samples=None):
    data_config = config["data"]
    dataset_root = Path(data_config["dataset_root"])
    use_filtered = data_config.get("use_filtered", True)
    excluded_samples = set(data_config.get("excluded_samples", []))
    excluded_datastrings = set(data_config.get("excluded_datastrings", []))

    master_path = dataset_root / "master.json"
    with master_path.open("r", encoding="utf-8") as handle:
        master = json.load(handle)

    event_path_key = "filtered_events_bin" if use_filtered else "incoming_events_bin"
    stream_name = "filtered" if use_filtered else "incoming"

    repetitions_root = output_root / "repetitions"
    repetitions_root.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    skipped = []

    total_samples = len(master["samples"])
    print(
        f"Building processed dataset from {dataset_root} "
        f"using {stream_name} events with {num_bins} bins."
    )
    print(f"Found {total_samples} chunk samples in master.json")

    sample_count = 0
    for sample_index, sample in enumerate(master["samples"], start=1):
        if sample["sample_name"] in excluded_samples:
            print(
                f"[skip sample {sample_index}/{total_samples}] "
                f"{sample['sample_name']} excluded by sample list"
            )
            skipped.append(
                {
                    "sample_name": sample["sample_name"],
                    "reason": "excluded_sample",
                }
            )
            continue
        if sample["datastring"] in excluded_datastrings:
            print(
                f"[skip sample {sample_index}/{total_samples}] "
                f"{sample['sample_name']} excluded by datastring "
                f"{sample['datastring']}"
            )
            skipped.append(
                {
                    "sample_name": sample["sample_name"],
                    "reason": "excluded_datastring",
                }
            )
            continue

        sample_count += 1
        if limit_samples is not None and sample_count > limit_samples:
            print(f"Reached --limit-samples={limit_samples}, stopping early.")
            break

        chunk_info_path = dataset_root / sample["chunk_info_json"]
        with chunk_info_path.open("r", encoding="utf-8") as handle:
            chunk_info = json.load(handle)

        event_path = dataset_root / sample[event_path_key]
        events = np.fromfile(event_path, dtype=EVENT_DTYPE)
        repetition_windows = get_ordered_repetition_windows(chunk_info)

        data_dir_name = Path(sample["data_dir"]).name
        chunk_dir_name = Path(sample["chunk_dir"]).name
        output_dir = repetitions_root / data_dir_name / chunk_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[process sample {sample_index}/{total_samples}] "
            f"{sample['sample_name']} -> {len(repetition_windows)} repetitions "
            f"from {event_path}"
        )

        times_ns = events["witnessed_utc_ns"] if events.size else None

        for window in repetition_windows:
            if events.size == 0:
                rep_events = events
            else:
                mask = (
                    (times_ns >= window["start_ns"])
                    & (times_ns <= window["end_ns"])
                )
                rep_events = events[mask]

            if rep_events.size == 0:
                print(
                    f"  [skip repetition] {window['repetition_name']} has no "
                    "events after clipping"
                )
                skipped.append(
                    {
                        "sample_name": sample["sample_name"],
                        "repetition_name": window["repetition_name"],
                        "reason": "empty_repetition_after_clipping",
                    }
                )
                continue

            features = summarize_repetition(
                rep_events=rep_events,
                start_ns=window["start_ns"],
                end_ns=window["end_ns"],
                num_bins=num_bins,
            )

            output_path = output_dir / f"{window['repetition_name']}.npy"
            np.save(output_path, features)
            nonzero_bins = int((features.sum(axis=1) != 0).sum())
            print(
                f"  [saved] {output_path} "
                f"events={int(rep_events.size)} nonzero_bins={nonzero_bins}"
            )

            manifest_entries.append(
                {
                    "source_sample_name": sample["sample_name"],
                    "sequence": sample["sequence"],
                    "datastring": sample["datastring"],
                    "data_dir": data_dir_name,
                    "chunk_dir": chunk_dir_name,
                    "chunk_name": sample["chunk_name"],
                    "repetition_name": window["repetition_name"],
                    "repetition_index": window["repetition_index"],
                    "source_window_key": window["source_key"],
                    "source_stream": stream_name,
                    "source_event_path": sample[event_path_key],
                    "processed_path": str(output_path.relative_to(output_root)),
                    "shape": [num_bins, len(FEATURE_NAMES)],
                    "feature_names": FEATURE_NAMES,
                    "transmitted_bits": sample["transmitted_bits"],
                    "start_utc_ns": window["start_ns"],
                    "end_utc_ns": window["end_ns"],
                    "raw_event_count": int(rep_events.size),
                }
            )

    manifest = {
        "format": "processed_repetition_dataset_v1",
        "source_dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "source_stream": stream_name,
        "num_bins": num_bins,
        "feature_names": FEATURE_NAMES,
        "excluded_datastrings": sorted(excluded_datastrings),
        "excluded_samples": sorted(excluded_samples),
        "summary": {
            "processed_repetitions": len(manifest_entries),
            "skipped_entries": len(skipped),
        },
        "repetitions": manifest_entries,
        "skipped": skipped,
    }

    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote processed manifest to {manifest_path}")
    return manifest_path, manifest


def main():
    args = parse_args()
    config = load_config(args.config)
    manifest_path, manifest = build_processed_dataset(
        config=config,
        output_root=args.output_root,
        num_bins=args.num_bins,
        limit_samples=args.limit_samples,
    )
    print(f"Wrote manifest: {manifest_path}")
    print(
        "Processed repetitions:",
        manifest["summary"]["processed_repetitions"],
    )
    print(
        "Skipped entries:",
        manifest["summary"]["skipped_entries"],
    )


if __name__ == "__main__":
    main()
