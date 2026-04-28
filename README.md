Event Decoding - DL final project

# Mamba Installation
python -m pip install mamba-ssm --no-build-isolation

# Current Status
- Right now, each repetition is considered as data sample.
- Usage of `continue` flag and `end` flag needs to be added.

# Dataset
- Put the extracted dataset_label under `data/`.
- Current data folder structure:

```bash
data/
├── dataset_processed/
│   ├── manifest.json
│   └── repetitions/
├── Dataset_label/
│   ├── master.json
│   └── transmissions/
└── transmission_log.json
```

- Build repetition data by running this command , it creates bins from the data:

```bash
python scripts/process_repetition.py
```
