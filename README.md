---
title: Kemet RAG
emoji: 𓂀
colorFrom: gold
colorTo: black
sdk: streamlit
app_file: app.py
pinned: false
---

# 𓂀 KEMET RAG
### Knowledge of the Ancient Scrolls — AI-Powered Egyptian Tour Guide

**Kemet RAG** is a professional-grade Retrieval-Augmented Generation (RAG) application designed to provide an immersive, educational experience for tourists exploring ancient Egypt. It combines multi-modal AI (Vision + Voice) with a curated library of historical texts to act as an "Oracle" that knows the secrets of the pyramids.

---

## ✨ Key Features

*   **𓂀 The Sacred Library (RAG)**: Index your own PDF and TXT scrolls. The app uses semantic search (ChromaDB) to retrieve factual information.
*   **🎙 The Oracle's Voice (TTS)**: Realistic, emotive speech synthesis via ElevenLabs, blended with atmospheric Egyptian background music.
*   **🖼 The Eye of Horus (Vision)**: Upload photos of landmarks. The app uses a **Swin Transformer (Swin-T)** classifier and LLaMA-3 Vision to identify and describe them.
*   **🌐 Universal Understanding**: Automatic language detection and translation — speak to the oracle in any language.
*   **📱 Modern Aesthetic**: A premium, dark-themed Streamlit UI inspired by ancient Egyptian art and gold.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Python 3.10+** and **ffmpeg** installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Configuration
1.  Copy `.env.example` to a new file named `.env`.
2.  Add your **Groq** and **ElevenLabs** API keys.

### 4. Prepare the Oracle (Optional)
If you have new landmark data, convert it for the classifier:
```bash
python tools/add.py
```

### 5. Launch the Experience
```bash
streamlit run app.py
```

---

## 📂 Project Structure

- `app.py`: Main entry point and UI.
- `core/`: The modular RAG engine (Voice, NLP, Indexing, LLM, TTS).
- `tools/`: Diagnostics and model conversion utilities.
- `assets/`: Background music and visual assets.
- `data/`: Your document library for RAG.
- `model/`: AI model weights and checkpoints.

---

## 🛡 Security
This project uses centralized configuration in `core/config.py`. API keys are managed via environment variables to keep your credentials safe from public exposure.

---
*Created for the travelers of the Nile.*
