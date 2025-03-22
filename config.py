# API Keys (Replace with your actual keys)
GEMINI_API_KEY = "AIzaSyDgP8I6ssQ4Zdl7mEr_5QEiOCXAO0yckjI"

# Vector Database Configuration (FAISS)
VECTOR_DB_FILE = "data/memory_index.faiss"
VECTOR_DB_DIMENSION = 512  # Adjusted for FAISS-based memory storage

# Memory Storage
MEMORY_DB_FILE = "data/memory.json"

# Personality Storage
PERSONALITY_FILE = "data/personality.json"

# Face Recognition Model
FACE_MODEL = "cnn"  # Options: 'hog' (CPU) or 'cnn' (GPU)

# Object Detection Model (YOLOv8)
YOLO_MODEL = "yolov8m"  # Change to 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', or 'yolov8x' as needed

# Text-to-Speech Configuration
TTS_VOICE = "en-GB-Wavenet-D"  # British accent for Bagley
