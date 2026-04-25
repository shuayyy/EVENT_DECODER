import torch
from torch.nn.utils.rnn import pad_sequence

def ctc_collate(batch):

    inputs = [items[0] for items in batch]
    targets = [items[1] for items in batch]

    input_lengths = torch.tensor(
        [len(input) for input in inputs], dtype=torch.long
    )

    target_lengths = torch.tensor(
        [len(target) for target in targets], dtype=torch.long
    )

    padded_inputs = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=0.0,
    )  

    flat_targets = torch.cat(targets, dim=0)

    return padded_inputs, flat_targets, input_lengths, target_lengths