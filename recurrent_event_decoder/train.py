import torch
from torch.utils.data import DataLoader

from recurrent_event_decoder.collate import recurrent_event_collate
from recurrent_event_decoder.config import RecurrentEventConfig
from recurrent_event_decoder.dataset import PreparedRecurrentEventDataset
from recurrent_event_decoder.losses import recurrent_event_vector_loss
from recurrent_event_decoder.metrics import compute_recurrent_metrics
from recurrent_event_decoder.model import RecurrentEventDecoder


def move_batch_to_device(batch, device):
    batch.windows = batch.windows.to(device)
    batch.metadata = batch.metadata.to(device)
    batch.targets = batch.targets.to(device)
    batch.valid_window_mask = batch.valid_window_mask.to(device)
    return batch


def train_recurrent_event_decoder(config=None, train_dataset=None, val_dataset=None):
    config = config or RecurrentEventConfig()
    if train_dataset is None or val_dataset is None:
        if config.prepared_manifest.exists():
            if train_dataset is not None or val_dataset is not None:
                raise ValueError(
                    "Provide both train_dataset and val_dataset, or neither."
                )
            prepared_dataset = PreparedRecurrentEventDataset(
                config.prepared_manifest,
                config=config,
            )
            if len(prepared_dataset) < 2:
                train_dataset = prepared_dataset
                val_dataset = prepared_dataset
            else:
                split_index = max(1, int(len(prepared_dataset) * 0.9))
                split_index = min(split_index, len(prepared_dataset) - 1)
                train_dataset, val_dataset = torch.utils.data.random_split(
                    prepared_dataset,
                    [split_index, len(prepared_dataset) - split_index],
                    generator=torch.Generator().manual_seed(42),
                )
        else:
            raise FileNotFoundError(
                "Prepared recurrent event manifest not found: "
                f"{config.prepared_manifest}. Run "
                "`conda run -n DL python -m recurrent_event_decoder.prepare_incoming_events "
                "--dataset-root 'data/Dataset_label (1)' "
                "--output-root data/recurrent_event_processed` first, "
                "or pass train_dataset and val_dataset explicitly."
            )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=recurrent_event_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=recurrent_event_collate,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentEventDecoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = []
    for _ in range(config.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config)
        val_metrics = evaluate(model, val_loader, device, config)
        history.append({"train": train_metrics, "val": val_metrics})
    return model, history


def train_one_epoch(model, loader, optimizer, device, config=None):
    config = config or RecurrentEventConfig()
    model.train()
    total_loss = 0.0
    batch_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch.windows, batch.metadata, batch.valid_window_mask)
        loss = recurrent_event_vector_loss(
            outputs,
            batch.targets,
            batch.valid_window_mask,
            config,
        )

        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    return {"loss": total_loss / max(batch_count, 1)}


def evaluate(model, loader, device, config=None):
    config = config or RecurrentEventConfig()
    model.eval()
    total_loss = 0.0
    metric_sums = {}
    batch_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch.windows, batch.metadata, batch.valid_window_mask)
            loss = recurrent_event_vector_loss(
                outputs,
                batch.targets,
                batch.valid_window_mask,
                config,
            )
            metrics = compute_recurrent_metrics(
                outputs,
                batch.targets,
                batch.valid_window_mask,
                config,
            )

            total_loss += loss.item()
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            batch_count += 1

    averaged = {key: value / max(batch_count, 1) for key, value in metric_sums.items()}
    averaged["loss"] = total_loss / max(batch_count, 1)
    return averaged
