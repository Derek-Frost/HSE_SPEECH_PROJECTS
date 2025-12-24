# HiFi-GAN Vocoder (Mel → Waveform)

## 📖 Описание проекта

Данный проект реализует **нейросетевой вокодер HiFi-GAN**, предназначенный для преобразования **мел-спектрограмм в аудиосигнал (waveform)**.  
Проект выполнен в рамках учебного задания по TTS и охватывает **полный цикл обучения и инференса вокодера**.

В реализации:
- используется архитектура **HiFi-GAN** (Generator + Adversarial Discriminators),
- обучение проводится на датасете **LJSpeech**,
- применяется **единый код извлечения mel-спектрограмм** для обучения и инференса,
- качество оценивается **субъективно**, по аудио-примерам на разных эпохах обучения.

Проект фокусируется именно на **вокодере**, а не на полной TTS-системе.

---

## 🧠 Архитектура

Используется архитектура **HiFi-GAN**, включающая:

- **Generator**
  - принимает mel-спектрограмму,
  - использует каскад upsampling-блоков,
  - residual-блоки с дилатациями,
  - генерирует waveform во временной области.

- **Discriminators**
  - **Multi-Period Discriminator (MPD)** — анализ периодической структуры сигнала,
  - **Multi-Scale Discriminator (MSD)** — анализ сигнала на разных временных масштабах.

Обучение проводится в **GAN-постановке** с дополнительными регуляризирующими потерями.

---

## 🏗️ Структура проекта

```text
hifigan_project/
│
├── src/
│   ├── model/
│   │   ├── hifigan.py              # Generator, MPD, MSD
│   │   ├── modules.py              # ResBlocks, Upsample-блоки
│   │   └── losses.py               # adversarial, feature matching, mel loss
│   │
│   ├── data/
│   │   ├── datasets.py             # LJSpeechDataset
│   │   └── collate.py              # паддинг и батчинг
│   │
│   ├── trainer/
│   │   └── trainer.py              # цикл обучения
│   │
│   └── utils/
│       ├── audio.py                # загрузка wav + mel_spectrogram
│       ├── checkpoint.py           # сохранение моделей
│       └── seed.py                 # фиксация сидов
│
├── configs/
│   └── hifigan_ljspeech.yaml       # конфигурация эксперимента
│
├── train.py                        # обучение HiFi-GAN
├── infer.py                        # инференс одного чекпоинта
├── run_infer_checkpoints.py        # инференс всех чекпоинтов
├── requirements.txt
└── README.md
