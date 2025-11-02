import os, csv, random

ROOT = "./train-clean-100"
OUT  = os.path.join(ROOT, "manifests")
os.makedirs(OUT, exist_ok=True)

pairs = []
for spk in os.listdir(ROOT):
    spk_dir = os.path.join(ROOT, spk)
    if not os.path.isdir(spk_dir): 
        continue
    for chap in os.listdir(spk_dir):
        chap_dir = os.path.join(spk_dir, chap)
        trans = os.path.join(chap_dir, f"{spk}-{chap}.trans.txt")
        if not os.path.isfile(trans):
            continue
        with open(trans, "r", encoding="utf-8") as f:
            for line in f:
                utt, txt = line.strip().split(" ", 1)
                flac = os.path.join(chap_dir, f"{utt}.flac")
                if os.path.isfile(flac):
                    pairs.append((os.path.abspath(flac), txt))

random.shuffle(pairs)
n = len(pairs)
n_dev = max(500, int(0.02 * n))
n_test = max(500, int(0.02 * n))
dev = pairs[:n_dev]
test = pairs[n_dev:n_dev+n_test]
train = pairs[n_dev+n_test:]

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_path","text"])
        for p, t in rows:
            w.writerow([p, t])

write_csv(os.path.join(OUT, "train.csv"), train)
write_csv(os.path.join(OUT, "dev.csv"), dev)
write_csv(os.path.join(OUT, "test.csv"), test)

print(f"Done. train={len(train)}, dev={len(dev)}, test={len(test)}")
