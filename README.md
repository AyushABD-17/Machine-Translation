<!-- ========================================================= -->
<!--                  NEURAL MACHINE TRANSLATION               -->
<!-- ========================================================= -->

<h1 align="center">📚 Neural Machine Translation Research</h1>

<p align="center">
<b>Multimodal AI System for English → French Translation</b><br/>
Computer Vision • Neural Machine Translation • Speech Synthesis
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![NLP](https://img.shields.io/badge/Field-NLP-green)
![Computer Vision](https://img.shields.io/badge/Field-Computer%20Vision-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</p>

---

# 🚀 Project Overview

This repository implements a **research-oriented Neural Machine Translation (NMT) system** and extends it into a **real-world AI application** capable of translating text captured directly from a camera.

The system integrates multiple AI domains:

- 📷 **Computer Vision** — Extract text from images using OCR  
- 🤖 **Neural Machine Translation** — Translate English → French  
- 🔊 **Speech Synthesis** — Convert translated text to speech  

The goal is to demonstrate a **complete multimodal AI pipeline** combining **vision, language, and speech**.

---

# 🧠 Multimodal Translation Pipeline

<p align="center">
Camera Input
│
▼
Image Capture
│
▼
OCR (Text Extraction)
│
▼
Neural Machine Translation
│
▼
French Translation
│
▼
Text-to-Speech Output

</p>

### Workflow

1️⃣ Capture image from camera  
2️⃣ Extract English text using OCR  
3️⃣ Translate using Neural Machine Translation model  
4️⃣ Generate spoken French translation  

---

# 🤖 Implemented Translation Models

| Model | Description |
|------|-------------|
| **RNN (GRU)** | Baseline sequence-to-sequence model |
| **RNN + Embedding** | Improves semantic representation |
| **Attention Seq2Seq** | Enhances alignment between languages |
| **Transformer (From Scratch)** | Modern self-attention architecture |

---

# ⚙️ Transformer Architecture

Key components implemented:

- Scaled Dot-Product Attention  
- Multi-Head Attention  
- Positional Encoding  
- Masked Decoder Attention  

<p align="center">
Input Embedding
│
▼
Positional Encoding
│
▼
Multi-Head Attention
│
▼
Feed Forward Network
│
▼
Decoder
│
▼
Softmax Output


</p>

---

# 🧪 Training Techniques

The project incorporates modern training strategies:

| Technique | Purpose |
|------|------|
Beam Search | Improves translation decoding |
Label Smoothing | Prevents model overconfidence |
Scheduled Sampling | Reduces exposure bias |
Mixed Precision | Accelerates GPU training |
Ablation Studies | Evaluates architecture components |

---

# 📊 Evaluation Metrics

Models are evaluated using standard machine translation metrics.

| Metric | Description |
|------|-------------|
BLEU Score | Measures translation quality |
Perplexity | Language modeling capability |
Token Accuracy | Correct token predictions |

### Example Results

| Model | BLEU |
|------|------|
RNN | 21.4 |
Attention | 27.9 |
Transformer | **32.5** |

---

# 🏗️ System Architecture

<p align="center">

       User
        │
        ▼
   Camera Input
        │
        ▼
    OCR Engine
        │
        ▼
   Neural Translation Model
        │
        ▼
   French Translation
        │
        ▼
   Speech Output

</p>

---

# 📁 Project Structure
 neural-machine-translation
│
├── configs
├── data
├── notebooks
│
├── src
│ ├── datasets
│ ├── models
│ ├── training
│ ├── decoding
│ ├── evaluation
│ ├── inference
│ ├── utils
│ │
│ ├── vision
│ │ ├── camera_capture.py
│ │ ├── image_preprocessing.py
│ │ └── ocr_engine.py
│ │
│ ├── speech
│ │ └── tts_engine.py
│ │
│ └── pipeline
│ └── camera_translation_pipeline.py
│
├── api
├── apps
├── deployment
├── scripts
└── tests

---

# ⚡ Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
python src/experiments/run_experiment.py --config configs/transformer_config.yaml
python api/fastapi_server.py
streamlit run apps/camera_translation_app.py
