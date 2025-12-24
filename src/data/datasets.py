import os
import torch
from torch.utils.data import Dataset
from src.utils.audio import load_wav, mel_spectrogram

class LJSpeechDataset(Dataset):
    def __init__(self, root, segment_size, sr, n_fft, win_length, hop_length, n_mels, fmin, fmax):
        self.root = root
        self.wav_dir = os.path.join(root, "wavs")
        meta_path = os.path.join(root, "metadata.csv")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.items = [line.strip().split("|")[0] for line in f.readlines()]

        self.segment_size = segment_size
        self.sr = sr
        self.mel_cfg = dict(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fid = self.items[idx]
        wav_path = os.path.join(self.wav_dir, f"{fid}.wav")
        y = load_wav(wav_path, self.sr)

        if y.numel() >= self.segment_size:
            start = torch.randint(0, y.numel() - self.segment_size + 1, (1,)).item()
            y_seg = y[start:start + self.segment_size]
        else:
            y_seg = torch.nn.functional.pad(y, (0, self.segment_size - y.numel()))

        mel = mel_spectrogram(y_seg.unsqueeze(0), **self.mel_cfg).squeeze(0)

        return {
            "audio": y_seg,         
            "mel": mel,             
            "id": fid,
        }
