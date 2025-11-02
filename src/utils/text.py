import torch, re

class TextEncoder:
    def __init__(self, alphabet="abcdefghijklmnopqrstuvwxyz' "):
        self.alphabet = alphabet
        self.blank_id = len(alphabet)
        self.c2i = {c:i for i,c in enumerate(alphabet)}

    def normalize(self, s: str):
        s = s.lower()
        s = re.sub("[^a-z' ]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def encode(self, s: str):
        s = self.normalize(s)
        return torch.tensor([self.c2i[c] for c in s if c in self.c2i], dtype=torch.long)

    def encode_batch(self, arr):
        seqs = [self.encode(s) for s in arr]
        lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True), lens