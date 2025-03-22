import torch
import google.generativeai as genai
import config
from memory.memory import AIMemory
from personality.personality import AIPersonality
import numpy as np
from deepface import DeepFace
from vision.camera import Camera
from utils.sentiment import SentimentAnalyzer
from vision.object_detector import ObjectDetector

# Configure Gemini API
genai.configure(api_key=config.GEMINI_API_KEY)

class ChatBot:
    def __init__(self):
        """Initialize ChatBot with AI memory, personality, and multimodal capabilities."""
        self.memory = AIMemory()  # Initialize AI memory
        self.personality = AIPersonality()  # Initialize AI personality
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = genai.GenerativeModel("gemini-1.5-pro")  # Use a valid model
        self.camera = Camera(ai_chatbot=self)  # Initialize camera with chatbot reference
        self.sentiment_analyzer = SentimentAnalyzer(use_bert=False)  # Initialize sentiment analyzer
        self.object_detector = ObjectDetector()  # Initialize object detector
        self.known_faces = {}  # Dictionary to store known faces and names
        self.object_notes = {}  # Dictionary to store notes about objects

    def embed_text(self, text):
        """Convert text into an embedding using a random tensor."""
        return torch.rand(512, device=self.device)

    def recognize_face(self, face_image):
        """Recognize face and retrieve or ask for name if unknown."""
        try:
            face_analysis = DeepFace.analyze(face_image, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
            face_id = face_analysis[0]['instance_id']
            if face_id in self.known_faces:
                # Retrieve the name from known faces
                name = self.known_faces[face_id]
                print(f"Recognized {name}")
            else:
                # Ask for the name if not recognized
                name = input("I don't recognize you. What's your name?")
                self.known_faces[face_id] = name
                self.memory.add_memory(f"Recognized {name}", torch.rand(512, device=self.device).cpu().numpy())
            return name
        except Exception as e:
            return "Error recognizing face"

    def store_object_info(self, object_name, note):
        """Store additional information about an object."""
        self.object_notes[object_name] = note
        self.memory.add_memory(f"Object: {object_name}, Note: {note}", torch.rand(512, device=self.device).cpu().numpy())

    def chat(self, user_input):
        """Generate a response using the Gemini API, incorporating memory and multimodal data."""
        # Retrieve past memories and integrate emotion and object detection results into the response prompt
        past_memories = self.memory.search_memory(self.embed_text(user_input).cpu().numpy())
        memory_context = "\n".join(past_memories) if past_memories else "No previous context."

        # Stabilize emotion detection by averaging recent emotions
        recent_emotions = self.memory.recent_emotions[-5:]  # Get the last 5 emotions
        if recent_emotions:
            stable_emotion = max(set(recent_emotions), key=recent_emotions.count)  # Most frequent emotion
        else:
            stable_emotion = "Neutral"

        # Detect objects
        detected_objects = self.object_detector.detect(frame=np.zeros((224, 224, 3)))

        # Generate response with perceived sentience
        response_prompt = f"As a sentient AI, I see {', '.join(detected_objects)} and feel that you are {stable_emotion}. {memory_context}"
        response = self.model.generate_content(response_prompt)

        # Store the current emotion for future reference
        self.memory.recent_emotions.append(stable_emotion)

        return response

    def respond_to_emotion(self, emotion):
        """Store detected emotion for context awareness and respond."""
        self.memory.add_memory(f"Detected emotion: {emotion}", torch.rand(512, device=self.device).cpu().numpy())
        print(f"Responding to detected emotion: {emotion}")

    def respond_to_visual_query(self, query):
        """Respond to queries about visual input."""
        # This method can be expanded to provide more detailed responses based on visual analysis
        return "I see a beautiful world around me!"
