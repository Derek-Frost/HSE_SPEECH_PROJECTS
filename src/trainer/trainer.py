import torch
from tqdm import tqdm
from src.model.losses import (
    discriminator_loss, generator_loss, feature_matching_loss, mel_l1_loss
)

class Trainer:
    def __init__(self, gen, mpd, msd, opt_g, opt_d, mel_cfg, device, lambda_fm=2.0, lambda_mel=45.0):
        self.gen = gen
        self.mpd = mpd
        self.msd = msd
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.mel_cfg = mel_cfg
        self.device = device
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel

    def train_epoch(self, loader):
        self.gen.train(); self.mpd.train(); self.msd.train()
        logs = {"loss_g": 0.0, "loss_d": 0.0}
        n = 0

        for batch in tqdm(loader, desc="train", leave=False):
            y = batch["audio"].to(self.device)         
            mel = batch["mel"].to(self.device)     

            y = y.unsqueeze(1)                        
            y_hat = self.gen(mel)                     

            t = min(y.size(-1), y_hat.size(-1))
            y = y[..., :t]
            y_hat = y_hat[..., :t]

            self.opt_d.zero_grad(set_to_none=True)

            dr_mpd, fr_mpd = self.mpd(y)
            df_mpd, ff_mpd = self.mpd(y_hat.detach())
            dr_msd, fr_msd = self.msd(y)
            df_msd, ff_msd = self.msd(y_hat.detach())

            fr_mpd = [[t.detach() for t in fmap] for fmap in fr_mpd]
            fr_msd = [[t.detach() for t in fmap] for fmap in fr_msd]

            loss_d = discriminator_loss(dr_mpd, df_mpd) + discriminator_loss(dr_msd, df_msd)
            loss_d.backward()
            self.opt_d.step()

            self.opt_g.zero_grad(set_to_none=True)

            df_mpd_g, ff_mpd_g = self.mpd(y_hat)
            df_msd_g, ff_msd_g = self.msd(y_hat)

            loss_adv = generator_loss(df_mpd_g) + generator_loss(df_msd_g)
            loss_fm = feature_matching_loss(fr_mpd, ff_mpd_g) + feature_matching_loss(fr_msd, ff_msd_g)
            loss_mel = mel_l1_loss(y.squeeze(1), y_hat.squeeze(1), self.mel_cfg)

            loss_g = loss_adv + self.lambda_fm * loss_fm + self.lambda_mel * loss_mel
            loss_g.backward()
            self.opt_g.step()

            logs["loss_g"] += loss_g.item()
            logs["loss_d"] += loss_d.item()
            n += 1

        for k in logs:
            logs[k] /= max(n, 1)
        return logs
