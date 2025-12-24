import os
import yaml
import torch
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.data.datasets import LJSpeechDataset
from src.data.collate import collate_fn
from src.model.hifigan import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from src.trainer.trainer import Trainer
from src.utils.checkpoint import save_ckpt

def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    mel_cfg = cfg["audio"]
    ds = LJSpeechDataset(
        root=cfg["data"]["root"],
        segment_size=cfg["data"]["segment_size"],
        sr=mel_cfg["sample_rate"],
        n_fft=mel_cfg["n_fft"],
        win_length=mel_cfg["win_length"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg["fmin"],
        fmax=mel_cfg["fmax"],
    )
    dl = DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    mcfg = cfg["model"]
    gen = Generator(
        n_mels=mel_cfg["n_mels"],
        upsample_rates=mcfg["upsample_rates"],
        upsample_kernels=mcfg["upsample_kernels"],
        resblock_kernel_sizes=mcfg["resblock_kernel_sizes"],
        resblock_dilations=mcfg["resblock_dilations"],
        channels=mcfg["channels"],
    ).to(device)

    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    opt_g = torch.optim.AdamW(gen.parameters(), lr=cfg["train"]["lr_g"], betas=tuple(cfg["train"]["betas"]))
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(msd.parameters()), lr=cfg["train"]["lr_d"], betas=tuple(cfg["train"]["betas"]))

    trainer = Trainer(
        gen, mpd, msd, opt_g, opt_d,
        mel_cfg=mel_cfg,
        device=device,
        lambda_fm=cfg["train"]["lambda_fm"],
        lambda_mel=cfg["train"]["lambda_mel"],
    )

    ckpt_dir = cfg["train"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        logs = trainer.train_epoch(dl)
        print(f"Epoch {epoch}: {logs}")

        if epoch % 5 == 0:
            save_ckpt(
                path=os.path.join(ckpt_dir, f"epoch_{epoch}.pt"),
                gen=gen, mpd=mpd, msd=msd, opt_g=opt_g, opt_d=opt_d, epoch=epoch, cfg=cfg
            )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
