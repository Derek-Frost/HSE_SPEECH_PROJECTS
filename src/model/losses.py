import torch
import torch.nn.functional as F
from src.utils.audio import mel_spectrogram

def discriminator_loss(d_real, d_fake):
    loss = 0
    for dr, df in zip(d_real, d_fake):
        loss += torch.mean((1 - dr) ** 2) + torch.mean(df ** 2)
    return loss

def generator_loss(d_fake):
    loss = 0
    for df in d_fake:
        loss += torch.mean((1 - df) ** 2)
    return loss

def feature_matching_loss(fmap_real, fmap_fake):
    loss = 0
    for fr_list, ff_list in zip(fmap_real, fmap_fake):
        for fr, ff in zip(fr_list, ff_list):
            loss += torch.mean(torch.abs(fr - ff))
    return loss

def mel_l1_loss(y, y_hat, mel_cfg):
    mel = mel_spectrogram(y, **mel_cfg)
    mel_hat = mel_spectrogram(y_hat, **mel_cfg)
    return F.l1_loss(mel_hat, mel)
