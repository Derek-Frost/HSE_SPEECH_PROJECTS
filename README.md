# ASR (DeepSpeech-style, CTC)

## 📖 Описание проекта

Данный проект реализует **полный пайплайн распознавания речи (ASR)** на основе архитектуры **DeepSpeech2**.  
Система поддерживает:
- 🧠 обучение с **CTC Loss** (Connectionist Temporal Classification),
- 🔤 декодирование через **greedy**, **beam search**, и **beam + языковую модель (KenLM)**,
- 🗂️ автоматическую подготовку датасета **LibriSpeech**,
- 📈 вычисление метрик **WER** и **CER**,
- ⚙️ гибкую систему конфигураций **Hydra**.

## 🏗️ Структура проекта

ASR_project/
├── src/
│ ├── configs/asr/
│ │ ├── experiment_small.yaml
│ │ ├── data.yaml
│ │ ├── model_small_5080.yaml
│ │ ├── train.yaml
│ │ └── decode_cpu.yaml
│ ├── data/librispeech.py
│ ├── features/featurizer.py
│ ├── models/deepspeech.py
│ ├── decoding/{beam_kenlm.py, greedy.py}
│ ├── utils/{text.py, metrics.py, io.py}
│ └── ...
│
├── scripts/
│ ├── prepare_librispeech.sh # скачивание и подготовка датасета
│ ├── make_manifests.py # создание CSV-манифестов
│ └── make_lm_corpus.py # генерация корпуса для языковой модели
│
├── train.py # обучение модели
├── inference.py # инференс и оценка
├── requirements.txt
└── README.md


## ⚙️ Установка окружения

```bash
conda create -n asr python=3.12
conda activate asr
pip install -r requirements.txt


sudo apt install build-essential cmake libboost-all-dev
git clone https://github.com/kpu/kenlm.git
cd kenlm && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

🧩 Скрипты

| Скрипт                   | Назначение                          | Выход                              |
| :----------------------- | :---------------------------------- | :--------------------------------- |
| `prepare_librispeech.sh` | Загрузка и подготовка датасета      | структура данных + манифесты       |
| `make_manifests.py`      | Формирование CSV для train/dev/test | `train.csv`, `dev.csv`, `test.csv` |
| `make_lm_corpus.py`      | Подготовка текста для LM            | `data/text_train.norm.txt`         |


### Обучение модели

python train.py -cn=experiment_small

### Инференс

⚡ Greedy decoding:

python inference.py -cn=experiment_small \
  beam.use=false \
  hydra.job.chdir=false

🔎 Beam search (без LM)

python inference.py -cn=experiment_small \
  beam.use=true \
  beam.lm_path=null \
  hydra.job.chdir=false

🧠 Beam search + KenLM

python inference.py -cn=experiment_small \
  beam.use=true \
  beam.beam_size=64 \
  beam.lm_path=./models/librispeech_4gram.arpa.bin \
  beam.unigrams_path=./data/unigrams.txt \
  beam.alpha=1.0 beam.beta=1.0 \
  hydra.job.chdir=false

  
📈 Результаты (на dev/test)

| Декодер       |     WER ↓ |     CER ↓ | Примечание                |
| :------------ | --------: | --------: | :------------------------ |
| Greedy        |     0.424 |     0.136 | базовый вариант           |
| Beam (без LM) |     0.411 |     0.131 | незначительное улучшение  |
| Beam + LM     | **0.253** | **0.114** | заметный прирост точности |





