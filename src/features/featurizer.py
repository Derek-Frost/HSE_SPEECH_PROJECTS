import torchaudio

class LogMelSpec:
    def __init__(self, sr=16000, n_mels=80, win_ms=25, hop_ms=10, specaug=None):
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=int(sr*win_ms/1000),
            hop_length=int(sr*hop_ms/1000),
            n_mels=n_mels,
            f_min=20,
            f_max=sr//2,
            power=1.0,
        )
        self.amplog = torchaudio.transforms.AmplitudeToDB()
        self.specaug = specaug

    def __call__(self, wav):
        spec = self.amplog(self.melspec(wav))
        if self.specaug:
            spec = self.specaug(spec)
        return spec

class SpecAugment:
    def __init__(self, F=15, T=50, freq_mask=2, time_mask=2):
        self.F, self.T, self.fm, self.tm = F, T, freq_mask, time_mask

    def __call__(self, x):
        import random
        M, T = x.shape
        for _ in range(self.fm):
            f0 = random.randint(0, max(0, M - self.F))
            x[f0:f0+self.F] = x.mean()
        for _ in range(self.tm):
            t0 = random.randint(0, max(0, T - self.T))
            x[:, t0:t0+self.T] = x.mean()
        return x