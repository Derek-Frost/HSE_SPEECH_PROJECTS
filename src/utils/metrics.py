import editdistance

def _split_words(s):
    return s.strip().split()

def compute_wer_cer(hypotheses, references):
    assert len(hypotheses) == len(references)
    total_w, total_w_err = 0, 0
    total_c, total_c_err = 0, 0
    for h, r in zip(hypotheses, references):
        hw, rw = _split_words(h), _split_words(r)
        total_w += len(rw)
        total_w_err += editdistance.eval(hw, rw)
        total_c += len(r)
        total_c_err += editdistance.eval(h, r)
    wer = total_w_err / max(1, total_w)
    cer = total_c_err / max(1, total_c)
    return wer, cer