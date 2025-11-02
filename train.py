import os
import random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm
from torch import amp

from src.decoding.beam_kenlm import make_decoder, decode_batch
from src.data.librispeech import LibriSpeechDataset, BucketingCollator
from src.features.featurizer import LogMelSpec, SpecAugment
from src.models.deepspeech import DS2
from src.utils.text import TextEncoder
from src.utils.metrics import compute_wer_cer
from src.utils.io import save_checkpoint


# -------------------- SEEDING (детерминизм) --------------------
def set_seed_everywhere(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN: детерминизм и одинаковые алгоритмы между запусками
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """Чтобы у каждого DataLoader worker был предсказуемый сид."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# --------------------------------------------------------------


def build_dataloaders(cfg: DictConfig, text_enc: TextEncoder):
    specaug = SpecAugment(**cfg.specaugment) if cfg.specaugment else None
    feat = LogMelSpec(cfg.sample_rate, cfg.n_mels, cfg.win_ms, cfg.hop_ms, specaug)
    collate = BucketingCollator(feat, text_enc)

    tr = LibriSpeechDataset(cfg.manifests.train, cfg.sample_rate, text_enc.normalize)
    dv = LibriSpeechDataset(cfg.manifests.dev,   cfg.sample_rate, text_enc.normalize)

    # общий генератор для шифла и всех воркеров
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        tr,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        prefetch_factor=cfg.train.prefetch_factor if "prefetch_factor" in cfg.train else 2,
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(cfg.train.num_workers > 0),
    )

    dev_loader = DataLoader(
        dv,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        prefetch_factor=cfg.train.prefetch_factor if "prefetch_factor" in cfg.train else 2,
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(cfg.train.num_workers > 0),
    )

    return train_loader, dev_loader


def ctc_loss_from_logits(logits, targets, feat_lens, target_lens, blank_id):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
    T_out = logits.size(1)
    input_lengths = torch.clamp(feat_lens // 4, max=T_out)
    return nn.functional.ctc_loss(
        log_probs, targets, input_lengths, target_lens,
        blank=blank_id, zero_infinity=True
    )


@hydra.main(version_base=None, config_path="src/configs/asr", config_name="experiment_small")
def main(cfg: DictConfig):
    # фиксируем сиды
    set_seed_everywhere(cfg.seed)

    print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_enc = TextEncoder(cfg.text.alphabet)

    train_loader, dev_loader = build_dataloaders(cfg, text_enc)

    num_classes = len(text_enc.alphabet) + 1  # + blank
    model = DS2(
        n_mels=cfg.n_mels,
        rnn_layers=cfg.model.rnn.layers,
        hidden=cfg.model.rnn.hidden,
        num_classes=num_classes
    ).to(device)
    model.num_classes = num_classes

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )
    scaler = amp.GradScaler('cuda', enabled=cfg.train.amp)

    global_step = 0
    best_wer = 1e9

    for epoch in range(cfg.train.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for feats, feat_lens, targets, target_lens in pbar:
            feats, feat_lens = feats.to(device, non_blocking=True), feat_lens.to(device)
            targets, target_lens = targets.to(device), target_lens.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.train.amp):
                logits = model(feats, feat_lens)
                loss = ctc_loss_from_logits(
                    logits, targets, feat_lens, target_lens,
                    blank_id=num_classes - 1
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()

            global_step += 1
            if global_step % cfg.train.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        # -------------------- DEV EVAL --------------------
        model.eval()
        hyps, refs = [], []

        decoder = None
        if cfg.beam.use:
            try:
                lm_path = None
                if "lm_path" in cfg.beam and str(cfg.beam.lm_path).lower() != "null":
                    lm_path = cfg.beam.lm_path
                elif "kenlm_bin" in cfg.beam and str(cfg.beam.kenlm_bin).lower() != "null":
                    lm_path = cfg.beam.kenlm_bin

                decoder = make_decoder(
                    alphabet=text_enc.alphabet,
                    kenlm_path=lm_path,
                    alpha=cfg.beam.alpha,
                    beta=cfg.beam.beta,
                    unigrams_path=getattr(cfg.beam, "unigrams_path", None),
                )
                print("[decode] beam search enabled", "with LM" if cfg.beam.kenlm_bin else "(no LM)")
            except ImportError as e:
                print("[decode] pyctcdecode not installed, falling back to greedy:", e)

        with torch.no_grad():
            for feats, feat_lens, targets, target_lens in tqdm(dev_loader, desc="dev"):
                feats, feat_lens = feats.to(device), feat_lens.to(device)
                logits = model(feats, feat_lens)  # [B,T,C]

                if decoder is not None:
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    logit_lens = torch.clamp(feat_lens // 4, max=log_probs.size(1)).cpu().numpy()
                    from numpy import float32 as _f32
                    hyps.extend(
                        decode_batch(
                            decoder,
                            log_probs.detach().cpu().numpy().astype(_f32),
                            logit_lens=logit_lens,
                            beam_size=cfg.beam.beam_size,
                        )
                    )
                else:
                    ids = logits.argmax(-1)  # [B,T]
                    for b in range(ids.size(0)):
                        seq, prev = [], -1
                        for t in ids[b].tolist():
                            if t == num_classes - 1:
                                prev = -1
                                continue
                            if t != prev:
                                seq.append(t)
                                prev = t
                        hyp = ''.join(text_enc.alphabet[i] for i in seq).strip()
                        hyps.append(' '.join(hyp.split()))

                for b in range(targets.size(0)):
                    l = target_lens[b].item()
                    ref = ''.join(text_enc.alphabet[i] for i in targets[b][:l].tolist())
                    refs.append(ref)

        wer, cer = compute_wer_cer(hyps, refs)
        print(f"epoch {epoch}: dev WER={wer:.3f}, CER={cer:.3f}")

        if wer < best_wer:
            best_wer = wer
            save_checkpoint(
                {"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg)},
                os.path.join(cfg.output_dir, "best.ckpt")
            )


if __name__ == '__main__':
    main()
