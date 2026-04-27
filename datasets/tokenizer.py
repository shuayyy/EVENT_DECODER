import torch

class Tokenizer:

    def __init__(self):
        self.blank_id = 0
        self.zero_id = 1
        self.one_id = 2
        self.id_to_symbol = {
            self.zero_id: "0",
            self.one_id: "1",
        }

    def encode(self, bitstream: str) -> torch.Tensor:
        ids =[]

        for bit in bitstream.strip():
            if bit == '0':
                ids.append(self.zero_id)
            elif bit == '1':
                ids.append(self.one_id)

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids) -> str:
        symbols = []
        for token_id in ids:
            symbol = self.id_to_symbol.get(int(token_id))
            if symbol is not None:
                symbols.append(symbol)
        return "".join(symbols)

    def decode_ctc(self, ids) -> str:
        decoded = []
        previous = None

        for token_id in ids:
            token_id = int(token_id)
            if token_id != self.blank_id and token_id != previous:
                decoded.append(token_id)
            previous = token_id

        return self.decode(decoded)
