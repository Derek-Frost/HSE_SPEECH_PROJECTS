from torch.utils.data import Dataset
import torch, torchaudio, csv
import soundfile as sf
import numpy as np

class LibriSpeechDataset(Dataset):
    def __init__(self, manifest, sample_rate=16000, text_normalizer=None):
        self.items = []
        with open(manifest, "r", encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append({"audio": row["audio_path"], "text": row["text"]})
        self.sr = sample_rate
        self.text_norm = text_normalizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        wav_np, sr = sf.read(it["audio"], dtype="float32")
        if wav_np.ndim == 2:  # если стерео -> моно
            wav_np = wav_np.mean(axis=1)
        wav = torch.from_numpy(wav_np)
        # ресемпл при необходимости
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
            sr = self.sr
        text = it["text"]
        if self.text_norm:
            text = self.text_norm(text)
        return wav, text

class BucketingCollator:
    def __init__(self, featurizer, text_encoder):
        self.featurizer = featurizer
        self.text_encoder = text_encoder

    def __call__(self, batch):
        xs, ys = zip(*batch)
        feats = [self.featurizer(x) for x in xs]
        feat_lens = torch.tensor([f.shape[-1] for f in feats], dtype=torch.long)
        feats = torch.nn.utils.rnn.pad_sequence(
            [f.transpose(0,1) for f in feats], batch_first=True
        ).transpose(1,2).contiguous()
        targets, target_lens = self.text_encoder.encode_batch(list(ys))
        return feats, feat_lens, targets, target_lens