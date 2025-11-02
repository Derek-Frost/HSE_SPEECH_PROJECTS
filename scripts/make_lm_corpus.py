import csv
import argparse
import os
import re


def normalize_text(text: str):
    """
    Упрощённая нормализация 
    """
    text = text.lower()
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Extract text corpus for KenLM")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to train.csv manifest (audio,text)")
    parser.add_argument("--out", type=str, required=True,
                        help="Path to output text file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    texts = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txt = normalize_text(row["text"])
            if len(txt) > 0:
                texts.append(txt)

    with open(args.out, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")

    print(f"Saved {len(texts)} lines to {args.out}")


if __name__ == "__main__":
    main()
