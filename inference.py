import os
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm

from src.data.librispeech import LibriSpeechDataset, BucketingCollator
from src.features.featurizer import LogMelSpec
from src.models.deepspeech import DS2
from src.utils.text import TextEncoder
from src.utils.metrics import compute_wer_cer

# beam search (+ LM через KenLM)
try:
    from src.decoding.beam_kenlm import make_decoder, decode_batch
    _HAS_BEAM = True
except Exception:
    _HAS_BEAM = False


def load_ckpt(path: str, model: torch.nn.Module):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)


def greedy_decode(logits: torch.Tensor, alphabet: str, blank_id: int):
    ids = logits.argmax(-1)
    hyps = []
    for b in range(ids.size(0)):
        seq, prev = [], -1
        for t in ids[b].tolist():
            if t == blank_id:
                prev = -1
                continue
            if t != prev:
                seq.append(t)
                prev = t
        hyp = "".join(alphabet[i] for i in seq).strip()
        hyps.append(" ".join(hyp.split()))
    return hyps


@hydra.main(version_base=None, config_path="src/configs/asr", config_name="experiment_small")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text/alphabet
    text_enc = TextEncoder(cfg.text.alphabet)
    alphabet = text_enc.alphabet
    num_classes = len(alphabet) + 1  # + blank (в конце)

    # model
    model = DS2(
        n_mels=cfg.n_mels,
        rnn_layers=cfg.model.rnn.layers,
        hidden=cfg.model.rnn.hidden,
        num_classes=num_classes,
    ).to(device)
    model.eval()

    # load weights
    ckpt_path = os.path.join(cfg.output_dir, "best.ckpt")
    if os.path.exists(ckpt_path):
        load_ckpt(ckpt_path, model)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("WARNING: best.ckpt not found, using random weights")

    # features & dataloader
    feat = LogMelSpec(cfg.sample_rate, cfg.n_mels, cfg.win_ms, cfg.hop_ms, specaug=None)
    collate = BucketingCollator(feat, text_enc)
    ds = LibriSpeechDataset(cfg.manifests.test, cfg.sample_rate, text_enc.normalize)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=(cfg.train.batch_size if "batch_size" in cfg.train else 8),
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        collate_fn=collate,
    )

    # beam (+ LM)
    use_beam = bool(getattr(cfg.beam, "use", False)) and _HAS_BEAM
    decoder = None
    if use_beam:
        lm_path = None
        if hasattr(cfg.beam, "lm_path") and str(cfg.beam.lm_path).lower() != "null":
            lm_path = cfg.beam.lm_path
        elif hasattr(cfg.beam, "kenlm_bin") and str(cfg.beam.kenlm_bin).lower() != "null":
            lm_path = cfg.beam.kenlm_bin

        try:
            decoder = make_decoder(
                alphabet=alphabet,
                kenlm_path=lm_path,
                alpha=float(cfg.beam.alpha),
                beta=float(cfg.beam.beta),
                unigrams_path=getattr(cfg.beam, "unigrams_path", None),
            )
            print("[decode] beam search enabled", "(with LM)" if lm_path else "(no LM)")
        except ImportError as e:
            print(f"[decode] pyctcdecode not installed ({e}). Falling back to greedy.")
            use_beam = False
        except Exception as e:
            print(f"[decode] beam init failed ({e}). Falling back to greedy.")
            use_beam = False

    hyps, refs = [], []
    blank_id = num_classes - 1

    with torch.no_grad():
        for feats, feat_lens, targets, target_lens in tqdm(dl, desc="test"):
            feats, feat_lens = feats.to(device), feat_lens.to(device)
            logits = model(feats, feat_lens)

            if use_beam and decoder is not None:
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                # длины выхода: conv stride 2x по времени → примерно //4; дополнительно ограничил фактической длиной T
                T_out = log_probs.size(1)
                logit_lens = torch.clamp(feat_lens // 4, max=T_out).cpu().numpy()
                lp = log_probs.detach().cpu().numpy().astype("float32")
                hyps.extend(
                    decode_batch(
                        decoder,
                        lp,
                        logit_lens=logit_lens,
                        beam_size=int(cfg.beam.beam_size),
                    )
                )
            else:
                hyps.extend(greedy_decode(logits, alphabet, blank_id))

            # refs
            for b in range(targets.size(0)):
                L = int(target_lens[b].item())
                ref = "".join(alphabet[i] for i in targets[b][:L].tolist())
                refs.append(ref)

    wer, cer = compute_wer_cer(hyps, refs)
    print(f"TEST: WER={wer:.3f}, CER={cer:.3f}")

    # пара примеров
    for i in range(min(5, len(hyps))):
        print(f"[{i}] REF: {refs[i]}")
        print(f"[{i}] HYP: {hyps[i]}")
        print("---")


if __name__ == "__main__":
    main()
