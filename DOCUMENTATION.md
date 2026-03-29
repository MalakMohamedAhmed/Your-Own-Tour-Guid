# 𓂀 KEMET RAG — Technical Documentation

Welcome to the internal scrolls of the **Kemet RAG** project. This document provides a complete map of the codebase, explaining the function of every file and the logic within.

---

## 📂 Project Architecture

The application follows a modular **Pipeline Architecture**. Each step of the user interaction (Voice → NLP → Retrieval → LLM → TTS) is isolated into its own script within the `/core/` directory.

### 1. Main Application (`app.py`)
**Role**: The central nervous system of the project.
- **Functions**:
  - `new_session()`: Manages chat history and unique IDs.
  - `active_sess()`: A helper to get the current open conversation.
- **Logic**: It coordinates the "Process Send" block, which calls every pipeline in sequence whenever you press Enter or record a voice note. It also contains the **Theming (CSS)** that gives the app its ancient Egyptian aesthetic.

---

## ⚙️ The Core RAG Engine (`/core/`)

### 📜 `pipeline_1_voice.py` (The Ear)
- **Function**: `transcribe_audio(audio_bytes, extension)`
- **How it works**: Uses the **Groq Whisper API** to turn recorded voice (`.webm`, `.wav`, etc.) into text. It includes robust error handling to clean up temporary files after processing.

### 📜 `pipeline_2_nlp.py` (The Translator)
- **Function**: `full_text_pipeline(text)`
- **How it works**: 
  - Detects the language (Arabic, English, French, etc.) using `langdetect`.
  - Translates non-English text to English so the RAG search works better.
  - Uses `nltk` to clean the text (removing symbols) so searching the database is more accurate.

### 📜 `pipeline_3_indexing.py` (The Librarian)
- **Functions**: `scan_and_index()`, `get_chroma_collection()`
- **How it works**: 
  - Scans the `/data/` folder for `.txt`, `.md`, and `.pdf` files.
  - Splits long documents into small "chunks."
  - Turns text into math (Vectors) using `SentenceTransformers`.
  - Saves everything into the **ChromaDB** database (`.kemet_chroma_db`).

### 📜 `pipeline_4_retrieval.py` (The Searcher)
- **Function**: `retrieve(query)`, `build_context(results)`
- **How it works**: Performs a **Semantic Search**. It finds the exact parts of your documents that "resonate" with the user's question and formats them into a "Context" block for the AI.

### 📜 `pipeline_5_llm.py` (The Brain)
- **Functions**: `groq_chat()`, `describe_image()`
- **How it works**: 
  - `groq_chat`: Sends the question + context to the LLaMA-3 or Mixtral models hosted on Groq.
  - `describe_image`: Uses a vision model to look at your attached photos and describe what it sees in the context of the Egyptian tour.

### 📜 `pipeline_6_tts.py` (The Mouth)
- **Function**: `run_tts_pipeline(text)`
- **How it works**: 
  - Calls **ElevenLabs** to generate the oracle's voice.
  - Uses `pydub` to blend that voice with the **Background Music** from `/assets/egyptian_crop.mp3`.
  - Adds professional audio mastering (normalization and compression).

### 📜 `pipeline_7_classifier.py` (The Eye)
- **Function**: `classify_image(image_bytes)`
- **How it works**: Uses a **Swin Transformer (Swin-T)** neural network trained on 20 specific Egyptian landmarks. If the confidence is high enough (>45%), it identifies the landmark automatically.

---

## 🛠 Developer Tools (`/tools/`)

- **`add.py`**: A conversion script. It takes the raw `data.pkl` file and turns it into `model.pth` (the weights file Swin-T needs to run).
- **`debug_tts.py`**: A diagnostic app. Run this if the voice stops working to check your API keys and your computer's `ffmpeg` settings.
- **`test_voice.py`**: A small script to test if the microphone transcription is working independently.

---

## 📦 Data & Resources

- **`/data/`**: Put any documents here. The app will automatically read them next time you hit "Reload Sacred Texts."
- **`/model/`**: Stores the heavy AI weights and knowledge bases for the image classifier.
- **`/assets/`**: Contains the atmospheric background music.
- **`.kemet_chroma_db/`**: The hidden folder where all your indexed documents are stored as searchable data.

---

## 📜 Dependencies (`requirements.txt`)
Lists every library needed to run the app. Install them all at once using:
`pip install -r requirements.txt`
