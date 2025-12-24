import torch

def pad_1d(x, max_len):
    return torch.nn.functional.pad(x, (0, max_len - x.size(0)))

def pad_2d(x, max_len):
    return torch.nn.functional.pad(x, (0, max_len - x.size(1)))

def collate_fn(batch):
    audios = [b["audio"] for b in batch]
    mels = [b["mel"] for b in batch]
    ids = [b["id"] for b in batch]

    max_audio = max(a.size(0) for a in audios)
    max_mel = max(m.size(1) for m in mels)

    audios = torch.stack([pad_1d(a, max_audio) for a in audios], dim=0)
    mels = torch.stack([pad_2d(m, max_mel) for m in mels], dim=0)

    return {"audio": audios, "mel": mels, "id": ids}
