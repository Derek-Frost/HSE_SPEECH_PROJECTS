import yaml
import torch
import soundfile as sf

from src.model.hifigan import Generator
from src.utils.audio import load_wav, mel_spectrogram

@torch.no_grad()
def main(cfg_path, ckpt_path, wav_in, wav_out):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
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

    y = load_wav(wav_in, mel_cfg["sample_rate"]).to(device)
    mel = mel_spectrogram(y.unsqueeze(0), **mel_cfg)
    y_hat = gen(mel).squeeze(0).squeeze(0).cpu().numpy()

    sf.write(wav_out, y_hat, mel_cfg["sample_rate"])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav_in", required=True)
    ap.add_argument("--wav_out", required=True)
    args = ap.parse_args()
    main(args.config, args.ckpt, args.wav_in, args.wav_out)
