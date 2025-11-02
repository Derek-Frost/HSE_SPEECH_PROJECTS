#!/usr/bin/env bash
set -e
# генерация CSV из готовых wav и транскриптов.
ROOT=${DATA_ROOT:-$PWD/data}
OUT_DIR="$ROOT/manifests"
mkdir -p "$OUT_DIR"

# ищем все txt и парсим
echo "audio_path,text" > "$OUT_DIR/train.csv"
echo "audio_path,text" > "$OUT_DIR/dev.csv"
echo "audio_path,text" > "$OUT_DIR/test.csv"

echo "[!] Заполни манифесты согласно своим путям к аудио и текстам" >&2