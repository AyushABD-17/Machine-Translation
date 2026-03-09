# Machine-Translation
Machine Translation using Neural Networks
Neural Machine Translation Research Project
Comparative Study of Sequence-to-Sequence Architectures for Low-Resource Neural Machine Translation

This project implements and analyzes multiple Neural Machine Translation (NMT) architectures for translating English → French. The goal is to study the effectiveness of different sequence-to-sequence models and modern transformer-based architectures in a controlled experimental environment.

## Multimodal Application Extension

This project includes a fully integrated Camera + OCR + Text-to-Speech pipeline.

### Architecture
```text
Camera
 ↓ (Capture)
OCR (EasyOCR)
 ↓ (English Text)
Translation Model (Transformer)
 ↓ (French Text)
Voice Output (gTTS)
```

### Running the Application

1. **Start Backend API**
```bash
python api/fastapi_server.py
```

2. **Start the Multimodal UI**
```bash
streamlit run apps/camera_translation_app.py
```

## Running the Project
### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train Model
```bash
python src/experiments/run_experiment.py --config configs/transformer_config.yaml
```

The repository is structured to resemble real-world machine learning research projects developed in industry and academia.
