import csv, os

def write_manifest(pairs, out_csv):
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "text"])
        for path, text in pairs:
            w.writerow([os.path.abspath(path), text])