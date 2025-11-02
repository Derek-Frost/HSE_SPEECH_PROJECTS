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

🧩 **Ключевые технологии:** `PyTorch`, `CTC`, `KenLM`, `Hydra`, `LibriSpeech`, `SpecAugment`



---
