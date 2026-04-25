from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from datasets.collate import ctc_collate
from datasets.dataset import EventDataset
from model.model import LSTMdecoder, MambaDecoder, TransformerDecoder


CONFIG_PATH = Path("config/hyperparameter.yaml")


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_model_config(config):
    shared_model_config = dict(config["model"])
    model_type = shared_model_config["model_type"].strip().lower()
    per_model_config = config["models"][model_type]
    return model_type, {**shared_model_config, **per_model_config}


def build_model(model_type, model_config):
    model_map = {
        "lstm": LSTMdecoder,
        "mamba": MambaDecoder,
        "transformer": TransformerDecoder,
    }
    if model_type not in model_map:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model_cls = model_map[model_type]
    return model_cls(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        output_dim=model_config["output_dim"],
    )


def validate_model_runtime(model_config, device):
    model_type = model_config["model_type"].strip().lower()
    if model_type == "mamba" and device.type != "cuda":
        raise RuntimeError("Mamba does not work on CPU. Use CUDA or choose a different model_type.")


def train():
    config = load_config()
    model_type, model_config = resolve_model_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_model_runtime(model_config, device)

    dataset = EventDataset(
        events_dir=config["data"]["events_dir"],
        labels_dir=config["data"]["labels_dir"],
    )

    loader = DataLoader(
        dataset,
        batch_size=model_config["batch_size"],
        shuffle=config["training"]["shuffle"],
        collate_fn=ctc_collate,
    )

    model = build_model(model_type, model_config).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_config["learning_rate"],
    )

    for epoch in range(model_config["epochs"]):
        model.train()
        total_loss = 0.0

        for batch in loader:
            inputs, targets, input_lengths, target_lengths = batch
            inputs, targets, input_lengths, target_lengths = [
                b.to(device)
                for b in (inputs, targets, input_lengths, target_lengths)
            ]

            logits = model(inputs, input_lengths)
            log_probs = nn.functional.log_softmax(logits, dim=-1)

            log_probs = log_probs.permute(1, 0, 2)

            loss = criterion(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: loss = {total_loss / len(loader):.4f}")


if __name__ == "__main__":
    train()
