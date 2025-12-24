<h1 align="center">🤖 ML & AI Projects Collection</h1>

<p align="center">
  <b>Набор моих проектов в области машинного обучения, речи и обработки естественного языка</b><br>
  <i>Каждый проект — в отдельной ветке. Здесь собраны описания, ссылки и ключевые результаты.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch">
  <img src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## 📚 Оглавление

1. 🎧 DeepSpeech2 ASR — Распознавание речи (CTC + KenLM)

## 🎧 DeepSpeech2 ASR

> **End-to-End система автоматического распознавания речи**  
> Архитектура: DeepSpeech2 (Conv + BiGRU + CTC)  
> Поддержка Beam Search и внешней языковой модели KenLM

**Основные возможности:**
- Обучение на LibriSpeech (`train-clean-100`)
- Поддержка `greedy`, `beam`, `beam + LM`
- Интеграция с KenLM для улучшения WER
- Гибкая настройка через Hydra
- Метрики: WER / CER

**Лучшие результаты (test):**
| Decoder | WER ↓ | CER ↓ |
|:--|--:|--:|
| Greedy | 0.424 | 0.136 |
| Beam (no LM) | 0.411 | 0.131 |
| Beam + LM | **0.253** | **0.114** |

🔗 **Проект:** [Открыть ветку `ASR`](https://github.com/Derek-Frost/HSE_SPEECH_PROJECTS/tree/ASR)

🧩 **Ключевые технологии:** `PyTorch`, `CTC`, `KenLM`, `Hydra`, `LibriSpeech`


2. 🔊 HiFi-GAN Vocoder — Нейросетевой вокодер для синтеза речи

## 🔊 HiFi-GAN Vocoder

> **Нейросетевой вокодер для преобразования мел-спектрограмм в аудиосигнал**  
> Архитектура: HiFi-GAN (Generator + Multi-Period Discriminator + Multi-Scale Discriminator)  
> Реализация выполнена с нуля в рамках учебного проекта

**Основные возможности:**
- Обучение HiFi-GAN на датасете **LJSpeech**
- Генерация waveform из mel-спектрограмм (mel → wav)
- Использование **единого кода извлечения mel-признаков** для train и inference
- Adversarial обучение с Multi-Period и Multi-Scale Discriminators
- Поддержка сохранения чекпоинтов и отслеживания прогресса обучения
- Инференс и сравнение качества синтеза на разных эпохах обучения

**Детали обучения:**
- Датасет: `LJSpeech-1.1`
- Частота дискретизации: 22 050 Hz
- Mel-спектрограммы: 80 mel-фильтров
- Потери:
  - Adversarial loss
  - Feature Matching loss
  - Mel L1 loss
- Чекпоинты сохраняются каждые **5 эпох**

**Оценка качества:**
- Основная оценка проводилась **субъективно**, по аудио-примерам
- Использовался инференс на фиксированной mel-спектрограмме
- Наблюдается постепенное улучшение качества синтеза по мере роста числа эпох:
  - ранние эпохи — шум и зачатки речи
  - поздние эпохи — разборчивая речь с меньшим числом артефактов

**Примеры результатов:**
- `gt.wav` — оригинальное аудио из LJSpeech
- `gen_epoch_5.wav`
- `gen_epoch_10.wav`
- `gen_epoch_15.wav`
- `gen_epoch_20.wav`

🔗 **Проект:** [Открыть ветку `HiFiGAN`](https://github.com/Derek-Frost/HSE_SPEECH_PROJECTS/tree/HiFi_GAN)

🧩 **Ключевые технологии:** `PyTorch`, `HiFi-GAN`, `GAN`, `TTS`, `Vocoder`, `LJSpeech`




---
