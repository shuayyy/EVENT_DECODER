import torch
from torch.nn.utils.rnn import pad_sequence


def ctc_collate(batch):
    inputs = [items[0] for items in batch]
    targets = [items[1] for items in batch]
    metadata = [items[2] for items in batch] if len(batch[0]) > 2 else None

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

    if metadata is not None:
        return padded_inputs, flat_targets, input_lengths, target_lengths, metadata

    return padded_inputs, flat_targets, input_lengths, target_lengths


def fixed_bits_collate(batch, target_bit_length=70):
    inputs = [items[0] for items in batch]
    targets = [items[1] for items in batch]
    metadata = [items[2] for items in batch] if len(batch[0]) > 2 else None

    input_lengths = torch.tensor(
        [len(input) for input in inputs], dtype=torch.long
    )

    padded_inputs = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=0.0,
    )

    for target in targets:
        if len(target) != target_bit_length:
            raise AssertionError(
                f"fixed_bits_collate expects target length {target_bit_length}, got {len(target)}"
            )
        if not torch.all((target == 1) | (target == 2)):
            raise AssertionError("fixed_bits_collate expects tokenizer IDs 1/2 only")

    fixed_targets = torch.stack(
        [(target - 1).to(dtype=torch.long) for target in targets],
        dim=0,
    )

    if metadata is not None:
        return padded_inputs, fixed_targets, input_lengths, metadata

    return padded_inputs, fixed_targets, input_lengths
