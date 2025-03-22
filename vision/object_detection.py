import torch
import google.generativeai as genai
import config
from memory.memory import AIMemory
import cv2
import numpy as np
import urllib.request

# Configure Gemini API
genai.configure(api_key=config.GEMINI_API_KEY)

# Path to YOLO model
MODEL_URL = "https://ssd.mathworks.com/supportfiles/vision/deeplearning/models/yolov8/yolov8m.onnx"
MODEL_PATH = "vision/yolov8m.onnx"

class ChatBot:
    def __init__(self):
        """Initialize ChatBot with AI memory."""
        self.memory = AIMemory()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = genai.GenerativeModel("gemini-1.5-pro")

    def embed_text(self, text):
        """Convert text into an embedding using a random tensor."""
        return torch.rand(512, device=self.device)

    def chat(self, user_input):
        """Generates a response using Gemini API, incorporating memory."""
        if not isinstance(user_input, str) or not user_input.strip():
            return "Error: Invalid input. Please enter a valid message."

        query_embedding = self.embed_text(user_input)
        past_memories = self.memory.search_memory(query_embedding.cpu().numpy())
        memory_context = "\n".join(past_memories) if past_memories else "No previous context."

        prompt = f"""
        AI Memory:
        {memory_context}
        
        User: {user_input}
        AI:
        """

        try:
            response = self.model.generate_content(prompt)
            ai_response = response.text.strip() if response and hasattr(response, "text") else "Error: No response received."
        except Exception as e:
            ai_response = f"Error generating response: {str(e)}"

        self.memory.add_memory(user_input, query_embedding.cpu().numpy())
        return ai_response

    def analyze_emotion(self, detected_emotion):
        """Store detected emotion for AI awareness without responding."""
        self.memory.add_memory(f"Detected emotion: {detected_emotion}", torch.rand(512, device=self.device).cpu().numpy())

    def store_detected_objects(self, objects):
        """Store detected objects in AI memory for awareness."""
        if objects:
            self.memory.add_memory(f"Detected objects: {', '.join(objects)}", torch.rand(512, device=self.device).cpu().numpy())

# Download the YOLO model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading YOLOv8 model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")

# Load YOLO model
def load_model():
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    return net

def detect_objects(bot):
    download_model()
    net = load_model()
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    detected_objects = set()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        for detection in detections[0]:  # Adjust based on actual YOLOv8 output structure
            confidence = detection[4]  # Confidence score
            if confidence > 0.25:
                x1, y1, x2, y2 = map(int, detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]))
                label = "Object"  # Placeholder (should be mapped to actual class labels)
                detected_objects.add(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        bot.store_detected_objects(list(detected_objects))
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    bot = ChatBot()
    detect_objects(bot)
