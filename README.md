**Multimodal AI Personal Assistant**
A privacy-first, locally running AI assistant that combines vision, voice, and memory for real-time intelligent interaction. This project brings together face recognition, speech-to-text, object detection, semantic memory, sentiment analysis, and LLM integration — creating a robust desktop assistant capable of perceiving and interacting with the world around it.

Built with the goal of making AI interaction more human-like, context-aware, and privacy-respecting.

---

## Table of Contents

* [Features](#features)
* [Demo](#demo)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Tech Stack](#tech-stack)
* [Contributing](#contributing)
* [License](#license)
* [Author](#author)

---

## Features

* Voice Input: Recognizes your commands using Whisper or SpeechRecognition.
* Semantic Memory: Retrieves and stores meaningful interactions with FAISS indexing.
* Face Recognition: Identifies users and adapts conversations using DeepFace and face\_recognition.
* Emotion Detection: Understands facial expressions for context-aware replies.
* Sentiment Analysis: Analyzes emotional tone in speech/text.
* Object Detection: Detects surroundings using YOLOv8 and camera feed.
* TTS (Text-to-Speech): Speaks back using Microsoft Edge TTS or fallback local TTS.
* LLM Integration: Uses local LLM (Mistral, LLaMA, etc.) for responses.
* GUI Interface: User-friendly interface using Tkinter for simple use.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/multimodal-ai-assistant.git
cd multimodal-ai-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Download models

* Whisper (STT): [`openai/whisper`](https://github.com/openai/whisper) or use `faster-whisper`
* LLM (e.g. Mistral/LLaMA): Use GGUF format with `llama-cpp-python`
* YOLOv8: Download pretrained model from Ultralytics
* DeepFace models: Handled automatically on first run

---

## Usage

### To launch the assistant:

```bash
python main.py
```

Make sure your webcam and microphone are connected. The GUI will launch, and you can interact using text or speech.

---

## Project Structure

```
multimodal-ai-assistant/
│
├── gui.py                 # GUI interface (Tkinter)
├── voice_input.py         # Voice capture and STT
├── sentiment.py           # Sentiment & emotion analysis
├── speech.py              # Text-to-Speech module
├── object_detection.py    # LLM, memory & object processing
├── camera.py              # Camera feed and face/emotion detection
├── face_recognition.py    # Face recognition logic
├── memory/                # FAISS indexing & JSON memory
├── models/                # LLM weights, Whisper, YOLO, etc.
├── requirements.txt
└── README.md
```

---

## Tech Stack

* **Language**: Python
* **Libraries**: OpenCV, DeepFace, MediaPipe, Transformers, SpeechRecognition, VADER, edge-tts, Tkinter
* **Models**: Mistral/LLaMA (local LLM), Whisper, YOLOv8
* **Memory**: FAISS for vector search, JSON for structured memory
* **GUI**: Tkinter (desktop interface)

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## License

MIT License. See `LICENSE` for more information.

---


