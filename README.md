Event Decoding - DL final project

# Mamba Installation
python -m pip install mamba-ssm --no-build-isolation

# Current Status
- Right now, each repetition is considered as data sample.
- Usage of `continue` flag and `end` flag needs to be added.

# Dataset
- Put the extracted raw dataset under `data/`:

```bash
data/
├── Dataset_label/
│   ├── master.json
│   └── transmissions/
└── transmission_log.json
```

- Build processed repetition data:

```bash
python scripts/process_repetition.py
```
