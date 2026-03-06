# 🌍 AI Tour Guide: Your Personal Storyteller

An intelligent, multimodal travel companion designed to replace traditional human tour guides by leveraging **Generative AI** and **RAG (Retrieval-Augmented Generation)**.

---

## 📌 Table of Contents
* [💡 Project Idea](#-project-idea)
* [⚙️ How it Works](#-how-it-works)
* [🛠 Tools & Tech Stack](#-tools--tech-stack)
* [📓 Notebooks & Data](#-notebooks--data)
* [🚀 Deployment & Presentation](#-deployment--presentation)
* [🏗 System Architecture](#-system-architecture)

---

## 💡 Project Idea
The **AI Tour Guide** addresses the accessibility gap in global tourism. While professional guides can be expensive or unavailable in specific languages, this AI-driven solution provides an immersive, storytelling-driven historical context on demand. 

By simply uploading a photo, speaking a name, or typing a landmark in your **native language**, the app identifies the location and generates a rich, factual narrative. It goes beyond providing dry facts by using an LLM to deliver information in a **compelling story format**, available as both text and audio.

---

## ⚙️ How it Works
The system is divided into three primary functional pipelines:

1.  **Input Phase:** * **Image:** Classified via a Vision Model to identify the landmark name.
    * **Speech:** Converted to text using Speech-to-Text (STT) models.
    * **Translation:** All inputs (Voice/Text) are detected and converted to English for processing.
2.  **Data Injection Pipe:** * Real-time data scraping via the **Wiki API**.
    * **Preprocessing:** Text cleaning, chunking, and lemmatization.
    * **Vectorization:** Converting text into embeddings for storage in a **Vector DB**.
3.  **Retrieval Pipe (RAG):** * Context is retrieved from the Vector DB and passed to the **LLM** with a storytelling prompt.
    * The final output is translated back to the user's original language and delivered via **Text-to-Speech (TTS)** or on-screen text.

---

## 🛠 Tools & Tech Stack
* **Frontend:** `Streamlit` (Web Deployment)
* **LLM Framework:** `LangChain` / `OpenAI` (Storytelling & Reasoning)
* **Vector Database:** `ChromaDB` (Efficient Context Retrieval)
* **NLP Tools:** `NLTK`, `Spacy` (Text Preprocessing & Lemmatization)
* **Computer Vision:** `CNN` / `ResNet` (Image Classification)
* **Audio Processing:** `Whisper` (STT) & `ElevenLabs` (TTS)
* **External API:** `Wikipedia API` (Source Knowledge)

---

## 📓 Notebooks & Data

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Preprocessing Notebook** | Data cleaning, chunking, and embedding generation. | [Open Notebook]([INSERT_LINK_HERE]) |
| **RAG Pipeline Notebook** | Retrieval logic and LLM prompt engineering. | [Open Notebook]([INSERT_LINK_HERE]) |
| **Image Classification** | Training the model to recognize landmarks. | [Open Notebook]([INSERT_LINK_HERE]) |
| **Main Dataset** | Landmark images and Wiki-text corpus. | [View Dataset]([INSERT_LINK_HERE]) |

---

## 🚀 Deployment & Presentation

* **Live App:** [Launch Streamlit Demo]([INSERT_LINK_HERE])
* **Project Presentation:** [View Slides/PDF]([INSERT_LINK_HERE])
* **Video Walkthrough:** [Watch Demo Video]([INSERT_LINK_HERE])

---

## 🏗 System Architecture
The following flowchart illustrates the end-to-end logic of the project, from user input to the final storyteller output.

![System Flowchart](https://github.com/YOUR_USERNAME/YOUR_REPO/blob/main/Screenshot%202026-03-06%20013454.png?raw=true)

---

### 🤝 Contributing
I’m always looking to improve the accuracy and storytelling capabilities of this guide. Feel free to fork the repo and submit a pull request!
