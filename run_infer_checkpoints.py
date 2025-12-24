import os
import glob
import yaml
import torch
import soundfile as sf

from src.model.hifigan import Generator
from src.utils.audio import load_wav, mel_spectrogram


@torch.no_grad()
def load_generator(cfg, ckpt_path, device):
    mel_cfg = cfg["audio"]
    mcfg = cfg["model"]

    gen = Generator(
        n_mels=mel_cfg["n_mels"],
        upsample_rates=mcfg["upsample_rates"],
        upsample_kernels=mcfg["upsample_kernels"],
        resblock_kernel_sizes=mcfg["resblock_kernel_sizes"],
        resblock_dilations=mcfg["resblock_dilations"],
        channels=mcfg["channels"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    gen.load_state_dict(ckpt["gen"])
    gen.eval()
    return gen


@torch.no_grad()
def main():
    cfg_path = "configs/hifigan_ljspeech.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    wav_in = r"W:\SPEECH\hifigan_project\LJSpeech-1.1\wavs\LJ001-0001.wav"

    mel_cfg = cfg["audio"]

    y = load_wav(wav_in, mel_cfg["sample_rate"]).to(device)

    mel = mel_spectrogram(y.unsqueeze(0), **mel_cfg)

    out_dir = "samples"
    os.makedirs(out_dir, exist_ok=True)

    sf.write(
        os.path.join(out_dir, "gt.wav"),
        y.detach().cpu().numpy(),
        mel_cfg["sample_rate"],
    )

    ckpt_dir = cfg["train"]["checkpoint_dir"]
    ckpt_dir = ckpt_dir.replace("/", "\\")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))

    if not ckpts:
        raise RuntimeError(f"Чекпоинты не найдены в {ckpt_dir}")

    print(f"Найдено чекпоинтов: {len(ckpts)}")

    for ckpt_path in ckpts:
        name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print("Инференс:", name)

        gen = load_generator(cfg, ckpt_path, device)

        y_hat = gen(mel).squeeze(0).squeeze(0)

        T = min(y_hat.numel(), y.numel())
        y_hat = y_hat[:T].detach().cpu().numpy()

        out_path = os.path.join(out_dir, f"gen_{name}.wav")
        sf.write(out_path, y_hat, mel_cfg["sample_rate"])

        print("  сохранён:", out_path)


if __name__ == "__main__":
    main()
