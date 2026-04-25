import torch

class Tokenizer:

    def __init__(self):
        self.blank_id = 0
        self.zero_id = 1
        self.one_id = 2

    def encode(self, bitstream: str) -> torch.Tensor:
        ids =[]

        for bit in bitstream.strip():
            if bit == '0':
                ids.append(self.zero_id)
            elif bit == '1':
                ids.append(self.one_id)

        return torch.tensor(ids, dtype=torch.long)
