from typing import Optional, List
import numpy as np

try:
    from pyctcdecode import build_ctcdecoder
except Exception:
    build_ctcdecoder = None


def make_decoder(
    alphabet: str,
    kenlm_path: Optional[str] = None,
    alpha: float = 1.6,
    beta: float = 1.0,
    unigrams_path: Optional[str] = None,
):
    if build_ctcdecoder is None:
        raise ImportError("pyctcdecode is not installed. Run: pip install pyctcdecode")

    labels = list(alphabet)

    # загрузка униграмм
    unigrams = None
    if unigrams_path is not None and str(unigrams_path).lower() != "null":
        with open(unigrams_path, "r", encoding="utf-8") as f:
            unigrams = [w.strip() for w in f if w.strip()]

    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=kenlm_path,
        alpha=alpha,
        beta=beta,
        unigrams=unigrams,
    )
    return decoder


def decode_batch(decoder, log_probs: np.ndarray, logit_lens=None, beam_size: int = 50) -> List[str]:
    B, T, C = log_probs.shape
    hyps: List[str] = []
    for i in range(B):
        T_i = int(logit_lens[i]) if logit_lens is not None else T
        hyp = decoder.decode(log_probs[i, :T_i, :], beam_width=beam_size)
        hyps.append(hyp.strip())
    return hyps
