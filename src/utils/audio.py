import torch
import torch.nn.functional as F
import torchaudio

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)

def mel_spectrogram(
    y: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    center: bool = False,
    pad_mode: str = "reflect",
):
    """
    y: (B, T) float32 in [-1, 1]
    return: (B, n_mels, frames)
    """
    if y.dim() == 1:
        y = y.unsqueeze(0)

    pad = (n_fft - hop_length) // 2
    y = F.pad(y, (pad, pad), mode=pad_mode)

    window = torch.hann_window(win_length, device=y.device)

    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.abs(spec)

    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=fmin,
        f_max=fmax,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm="slaney",
        mel_scale="slaney",
    )
    
    mel_fb = mel_fb.to(device=y.device, dtype=spec.dtype)


    mel = torch.matmul(spec.transpose(1, 2), mel_fb).transpose(1, 2)
    mel = spectral_normalize_torch(mel)
    return mel

def load_wav(path: str, sr: int):
    wav, file_sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)  # mono
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    wav = wav.squeeze(0)
    wav = wav.clamp(-1.0, 1.0)
    return wav
