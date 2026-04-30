Event Decoding - DL final project


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
python event_transformer/process_repetition.py
```

- Train the transformer decoder with:

```bash
python event_transformer/train.py
```

# Bin Features
- Each repetition is split into equal-time bins.
- Each bin currently has 13 `float32` features:

```text
1. log_pos = log1p(pos_count)
2. log_neg = log1p(neg_count)
3. log_total = log1p(pos_count + neg_count)
4. polarity_ratio = (pos_count - neg_count) / (pos_count + neg_count + 1e-6)
5. x_mean_norm = mean(x events in bin) / image_width
6. y_mean_norm = mean(y events in bin) / image_height
7. x_std_norm = std(x events in bin) / image_width
8. y_std_norm = std(y events in bin) / image_height
9. flicker_score = 2 * min(pos_count, neg_count) / (pos_count + neg_count + 1e-6)
10. highpass_signal = log_total - moving_average(log_total)
11. delta_highpass_signal[0] = 0, delta_highpass_signal[t] = highpass_signal[t] - highpass_signal[t-1]
12. rising_edge_score = max(delta_highpass_signal, 0)
13. falling_edge_score = max(-delta_highpass_signal, 0)
```

- Previous 8-feature bin version:

```text
1. log_pos = log1p(pos_count)
2. log_neg = log1p(neg_count)
3. log_total = log1p(pos_count + neg_count)
4. polarity_ratio = (pos_count - neg_count) / (pos_count + neg_count + 1e-6)
5. highpass_signal = log_total - moving_average(log_total)
6. delta_highpass_signal[0] = 0, delta_highpass_signal[t] = highpass_signal[t] - highpass_signal[t-1]
7. rising_edge_score = max(delta_highpass_signal, 0)
8. falling_edge_score = max(-delta_highpass_signal, 0)
```

- Previous 4-feature bin version:

```text
1. pos_count = number of positive-polarity events in the bin
2. neg_count = number of negative-polarity events in the bin
3. duration_norm = duration covered by events in the bin / repetition_duration
4. std_time_norm = std(event times in the bin) / repetition_duration
```
