import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import multiprocessing
from pathlib import Path
import sys

import numpy as np

from datasets.formats import EVENT_DTYPE
from recurrent_event_decoder.config import RecurrentEventConfig


@dataclass(frozen=True)
class PrepareOptions:
    dataset_root: Path
    output_root: Path
    window_size: int
    drop_partial: bool
    include_warnings: bool
    include_empty: bool
    max_windows: int | None
    time_target: str
    overwrite: bool
    dry_run: bool


class ProgressBar:
    def __init__(self, total, enabled=True, label="episodes"):
        self.total = total
        self.enabled = enabled
        self.label = label
        self.current = 0
        self.render()

    def update(self, increment=1):
        self.current += increment
        self.render()

    def render(self):
        if not self.enabled:
            return
        width = 30
        ratio = self.current / max(self.total, 1)
        filled = min(width, int(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r{self.label}: [{bar}] {self.current}/{self.total}",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def close(self):
        if self.enabled:
            print(file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Dataset_label chunk incoming_events.bin files as recurrent "
            "event-decoder episodes."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Dataset_label (1)"),
        help="Root containing the transmissions/ directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/recurrent_event_processed"),
        help="Directory where prepared episode arrays and manifest are written.",
    )
    parser.add_argument("--window-size", type=int, default=7500)
    parser.add_argument(
        "--drop-partial",
        action="store_true",
        help="Drop the final partial window instead of zero-padding it.",
    )
    parser.add_argument(
        "--include-warnings",
        action="store_true",
        help="Include chunks whose chunk_info.json contains warnings.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include chunks with zero incoming events.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap for windows per episode. Defaults to the full chunk.",
    )
    parser.add_argument(
        "--time-target",
        choices=("end_from_first_event", "end_from_window_start", "duration"),
        default="end_from_first_event",
        help=(
            "Value stored in target[71]. end_from_first_event is nominal end "
            "time minus first witnessed incoming event time. end_from_window_start uses "
            "requested_window_start_utc_ns. duration is nominal end minus nominal start."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Scan at most this many chunk_info.json files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing episode arrays.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report what would be prepared without writing arrays.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of chunk-preparation workers. Keep low for spinning disks or "
            "limited disk bandwidth."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar.",
    )
    return parser.parse_args()


def iter_chunk_infos(dataset_root):
    transmissions_root = dataset_root / "transmissions"
    yield from sorted(transmissions_root.glob("data_*/chunk_*/chunk_info.json"))


def build_target_vector(chunk_info, time_target_mode, first_event_utc_ns):
    config = RecurrentEventConfig()
    label = chunk_info["label"]
    bits = label["transmitted_bits"]
    if len(bits) != config.bit_count:
        raise ValueError(
            f"Expected {config.bit_count} transmitted bits, got {len(bits)} "
            f"for {chunk_info['sample_name']}"
        )

    target = np.zeros(config.output_dim, dtype=np.float32)
    target[: config.bit_count] = np.fromiter((int(bit) for bit in bits), dtype=np.float32)
    target[config.continue_index] = config.stop_threshold

    timing = chunk_info["timing"]
    nominal_end_micro = float(timing["nominal_end_time_micro"])
    if time_target_mode == "duration":
        target_time_micro = nominal_end_micro - float(timing["nominal_start_time_micro"])
    elif time_target_mode == "end_from_window_start":
        target_time_micro = nominal_end_micro - (
            float(timing["requested_window_start_utc_ns"]) / 1000.0
        )
    else:
        target_time_micro = nominal_end_micro - (float(first_event_utc_ns) / 1000.0)

    target[config.time_index] = target_time_micro
    return target


def event_count_from_file(event_path):
    size = event_path.stat().st_size
    if size % EVENT_DTYPE.itemsize != 0:
        raise ValueError(
            f"{event_path} has {size} bytes, which is not divisible by "
            f"{EVENT_DTYPE.itemsize}"
        )
    return size // EVENT_DTYPE.itemsize


def make_episode_paths(output_root, chunk_info):
    label = chunk_info["label"]
    data_name = f"data_{int(label['sequence']):06d}"
    chunk_name = chunk_info["chunk_dir"].split("/")[-1]
    episode_dir = output_root / "episodes" / data_name / chunk_name
    return {
        "episode_dir": episode_dir,
        "windows": episode_dir / "windows.npy",
        "valid_event_counts": episode_dir / "valid_event_counts.npy",
    }


def write_windows(
    event_path,
    windows_path,
    valid_counts_path,
    window_size,
    keep_partial,
    max_windows=None,
):
    event_count = event_count_from_file(event_path)
    if event_count == 0:
        raise ValueError(f"{event_path} contains no events")

    if keep_partial:
        window_count = math.ceil(event_count / window_size)
    else:
        window_count = event_count // window_size
    if window_count <= 0:
        raise ValueError(
            f"{event_path} has {event_count} events, not enough for one full window"
        )
    if max_windows is not None:
        window_count = min(window_count, max_windows)

    windows_path.parent.mkdir(parents=True, exist_ok=True)
    events = np.memmap(event_path, dtype=EVENT_DTYPE, mode="r")
    windows = np.lib.format.open_memmap(
        windows_path,
        mode="w+",
        dtype=np.float32,
        shape=(window_count, window_size, 4),
    )
    valid_counts = np.zeros(window_count, dtype=np.int32)

    first_event_utc_ns = int(events[0]["witnessed_utc_ns"])
    for window_index in range(window_count):
        start = window_index * window_size
        end = min(start + window_size, event_count)
        event_slice = events[start:end]
        valid_count = end - start
        valid_counts[window_index] = valid_count

        window = windows[window_index]
        window.fill(0.0)
        window[:valid_count, 0] = event_slice["x"].astype(np.float32)
        window[:valid_count, 1] = event_slice["y"].astype(np.float32)
        window[:valid_count, 2] = event_slice["polarity"].astype(np.float32)
        window[:valid_count, 3] = (
            event_slice["witnessed_utc_ns"].astype(np.float64)
            - float(first_event_utc_ns)
        ).astype(np.float32) / 1000.0

    windows.flush()
    np.save(valid_counts_path, valid_counts, allow_pickle=False)
    return {
        "event_count": int(event_count),
        "window_count": int(window_count),
        "first_event_utc_ns": int(first_event_utc_ns),
    }


def prepare_episode(
    options,
    chunk_info_path,
):
    dataset_root = options.dataset_root
    output_root = options.output_root
    with chunk_info_path.open("r", encoding="utf-8") as handle:
        chunk_info = json.load(handle)

    warnings = chunk_info.get("warnings", [])
    if warnings and not options.include_warnings:
        return None, f"skipped warning chunk: {chunk_info['sample_name']} {warnings}"

    event_path = dataset_root / chunk_info["incoming_events"]["path"]
    if not event_path.exists():
        return None, f"missing incoming events: {event_path}"

    file_event_count = event_count_from_file(event_path)
    if file_event_count == 0 and not options.include_empty:
        return None, f"skipped empty chunk: {chunk_info['sample_name']}"

    keep_partial = not options.drop_partial
    window_count = (
        math.ceil(file_event_count / options.window_size)
        if keep_partial
        else file_event_count // options.window_size
    )
    if options.max_windows is not None:
        window_count = min(window_count, options.max_windows)
    if window_count <= 0:
        return None, f"skipped chunk with no prepared windows: {chunk_info['sample_name']}"

    paths = make_episode_paths(output_root, chunk_info)
    if paths["windows"].exists() and not options.overwrite:
        metadata = {
            "event_count": file_event_count,
            "window_count": int(np.load(paths["valid_event_counts"]).shape[0]),
            "first_event_utc_ns": int(chunk_info["incoming_events"]["first_event_utc_ns"]),
        }
    elif options.dry_run:
        metadata = {
            "event_count": file_event_count,
            "window_count": int(window_count),
            "first_event_utc_ns": int(chunk_info["incoming_events"]["first_event_utc_ns"]),
        }
    else:
        metadata = write_windows(
            event_path=event_path,
            windows_path=paths["windows"],
            valid_counts_path=paths["valid_event_counts"],
            window_size=options.window_size,
            keep_partial=keep_partial,
            max_windows=options.max_windows,
        )

    target = build_target_vector(
        chunk_info,
        options.time_target,
        metadata["first_event_utc_ns"],
    )
    rel_windows_path = paths["windows"].relative_to(output_root)
    rel_counts_path = paths["valid_event_counts"].relative_to(output_root)

    episode = {
        "sample_name": chunk_info["sample_name"],
        "source_chunk_info": str(chunk_info_path.relative_to(dataset_root)),
        "source_incoming_events": chunk_info["incoming_events"]["path"],
        "windows_path": str(rel_windows_path),
        "valid_event_counts_path": str(rel_counts_path),
        "event_count": metadata["event_count"],
        "window_count": metadata["window_count"],
        "window_size": options.window_size,
        "frequency": chunk_info["label"]["frequency"],
        "repetitions": chunk_info["label"]["repetitions"],
        "transmitted_bits": chunk_info["label"]["transmitted_bits"],
        "target": target.tolist(),
        "target_time_mode": options.time_target,
        "warnings": warnings,
    }
    return episode, None


def prepare_episode_worker(index_and_path, options):
    index, chunk_info_path = index_and_path
    try:
        episode, skip_reason = prepare_episode(options, chunk_info_path)
        return {
            "index": index,
            "episode": episode,
            "skip_reason": skip_reason,
            "error": None,
        }
    except Exception as error:
        return {
            "index": index,
            "episode": None,
            "skip_reason": None,
            "error": f"{chunk_info_path}: {error}",
        }


def run_sequential(chunk_info_paths, options, limit, progress):
    episodes = []
    skipped = []
    errors = []
    for index, chunk_info_path in enumerate(chunk_info_paths):
        if limit is not None and len(episodes) >= limit:
            break
        result = prepare_episode_worker((index, chunk_info_path), options)
        if result["error"] is not None:
            errors.append(result["error"])
        elif result["episode"] is None:
            skipped.append(result["skip_reason"])
        else:
            episodes.append(result["episode"])
        progress.update()
    return episodes, skipped, errors


def run_parallel(chunk_info_paths, options, progress, num_workers):
    episodes = []
    skipped = []
    errors = []
    results = []
    indexed_paths = list(enumerate(chunk_info_paths))

    multiprocessing_context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=multiprocessing_context,
    ) as executor:
        future_to_index = {
            executor.submit(prepare_episode_worker, indexed_path, options): indexed_path[0]
            for indexed_path in indexed_paths
        }
        for future in as_completed(future_to_index):
            result = future.result()
            results.append(result)
            progress.update()

    for result in sorted(results, key=lambda item: item["index"]):
        if result["error"] is not None:
            errors.append(result["error"])
        elif result["episode"] is None:
            skipped.append(result["skip_reason"])
        else:
            episodes.append(result["episode"])

    return episodes, skipped, errors


def main():
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")

    options = PrepareOptions(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        window_size=args.window_size,
        drop_partial=args.drop_partial,
        include_warnings=args.include_warnings,
        include_empty=args.include_empty,
        max_windows=args.max_windows,
        time_target=args.time_target,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    dataset_root = options.dataset_root
    output_root = options.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    chunk_info_paths = list(iter_chunk_infos(dataset_root))
    if args.limit is not None:
        chunk_info_paths = chunk_info_paths[: args.limit]

    progress_total = len(chunk_info_paths)
    progress = ProgressBar(
        total=progress_total,
        enabled=not args.no_progress,
        label="chunks",
    )
    if not args.no_progress:
        print(
            f"\nPreparing {len(chunk_info_paths)} chunks with "
            f"{args.num_workers} worker(s)...",
            file=sys.stderr,
            flush=True,
        )
    try:
        if args.num_workers == 1:
            episodes, skipped, errors = run_sequential(
                chunk_info_paths,
                options,
                args.limit,
                progress,
            )
        else:
            episodes, skipped, errors = run_parallel(
                chunk_info_paths,
                options,
                progress,
                args.num_workers,
            )
    finally:
        progress.close()

    if errors:
        error_preview = "\n".join(errors[:10])
        raise RuntimeError(
            f"{len(errors)} chunks failed during preparation. First errors:\n"
            f"{error_preview}"
        )

    manifest = {
        "dataset_root": str(dataset_root),
        "window_size": options.window_size,
        "event_source": "incoming_events",
        "feature_order": ["x", "y", "polarity", "relative_time_us"],
        "time_reference": (
            "All relative_time_us features are witnessed_utc_ns minus the first "
            "witnessed incoming event in the episode, divided by 1000."
        ),
        "target_layout": {
            "bits": [0, 69],
            "continue_index": 70,
            "time_index": 71,
            "time_mode": options.time_target,
            "default_time_mode_description": (
                "end_from_first_event stores nominal_end_time_micro minus the first "
                "witnessed incoming event time in microseconds."
            ),
        },
        "num_workers": args.num_workers,
        "episode_count": len(episodes),
        "skipped_count": len(skipped),
        "episodes": episodes,
        "skipped": skipped,
    }

    manifest_path = output_root / "manifest.json"
    if not args.dry_run:
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    print(
        f"Prepared {len(episodes)} episodes; skipped {len(skipped)}; "
        f"manifest={manifest_path if not args.dry_run else '[dry-run]'}"
    )
    for reason in skipped[:10]:
        print(f"skip: {reason}")


if __name__ == "__main__":
    main()
