import os
import time
from llm.chatbot import ChatBot
from memory.memory import AIMemory
from vision.camera import Camera
from vision.face_recognition import FaceRecognition
from vision.object_detection import ObjectDetector
from personality.personality import AIPersonality
from utils.speech import SpeechSynthesizer
from utils.sentiment import SentimentAnalyzer
from utils.voice_input import VoiceInput  # New voice input module

class BagleyAI:
    def __init__(self, use_voice=True):
        """Initialize Bagley AI with all modules and optional voice input."""
        print("[Bagley AI] Initializing...")
        self.chatbot = ChatBot()
        self.memory = AIMemory()
        self.camera = Camera()
        self.face_recognition = FaceRecognition()
        self.object_detector = ObjectDetector()
        self.personality = AIPersonality()
        self.speech = SpeechSynthesizer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.voice_input = VoiceInput() if use_voice else None  # Enable voice input

    def start(self):
        """Start Bagley AI conversation loop (supports voice & text)."""
        print("[Bagley AI] Ready! Say 'Hey Bagley' or type 'exit' to quit.")

        while True:
            if self.voice_input:
                self.voice_input.wait_for_wake_word()  # Wait for "Hey Bagley"
                user_input = self.voice_input.listen()
            else:
                user_input = input("You: ")

            if not user_input or user_input.lower() in ["exit", "quit"]:
                print("[Bagley AI] Goodbye!")
                break

            # Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze_sentiment(user_input)
            emotion = self.sentiment_analyzer.interpret_score(sentiment)
            print(f"[Emotion] Detected: {emotion}")

            # Retrieve memory and generate response
            relevant_memory = self.memory.retrieve_memory(user_input)
            response = self.chatbot.generate_response(user_input, relevant_memory)

            # Adjust personality based on sentiment
            self.personality.adjust_based_on_interaction(user_input, response, emotion)

            print(f"Bagley AI: {response}")
            self.speech.synthesize(response)

if __name__ == "__main__":
    bagley = BagleyAI(use_voice=True)  # Enable voice input
    bagley.start()
