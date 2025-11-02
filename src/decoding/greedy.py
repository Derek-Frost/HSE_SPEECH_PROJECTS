import torch

def ctc_greedy_ids(logits, blank_id):
    ids = logits.argmax(-1)
    outs = []
    for row in ids.tolist():
        prev = -1
        seq = []
        for t in row:
            if t == blank_id: prev = -1; continue
            if t != prev:
                seq.append(t); prev = t
        outs.append(seq)
    return outs