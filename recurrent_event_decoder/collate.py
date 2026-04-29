import torch

from recurrent_event_decoder.types import RecurrentEventBatch


def recurrent_event_collate(samples):
    max_windows = max(sample.windows.shape[0] for sample in samples)
    window_size = samples[0].windows.shape[1]
    event_dim = samples[0].windows.shape[2]

    windows = torch.zeros(
        len(samples),
        max_windows,
        window_size,
        event_dim,
        dtype=samples[0].windows.dtype,
    )
    valid_window_mask = torch.zeros(len(samples), max_windows, dtype=torch.bool)
    targets = torch.stack([sample.target for sample in samples], dim=0)
    metadata = torch.stack(
        [
            torch.stack(
                [
                    sample.num_repetitions.reshape(()),
                    sample.transmission_frequency.reshape(()),
                ]
            )
            for sample in samples
        ],
        dim=0,
    ).to(dtype=torch.float32)
    source_ids = [sample.source_id for sample in samples]

    for batch_index, sample in enumerate(samples):
        window_count = sample.windows.shape[0]
        windows[batch_index, :window_count] = sample.windows
        valid_window_mask[batch_index, :window_count] = sample.valid_window_mask

    return RecurrentEventBatch(
        windows=windows,
        metadata=metadata,
        targets=targets,
        valid_window_mask=valid_window_mask,
        source_ids=source_ids,
    )

