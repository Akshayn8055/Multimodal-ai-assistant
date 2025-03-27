import tkinter as tk
from tkinter import scrolledtext
import threading
from llm.chatbot import ChatBot  # Import ChatBot class
from memory.memory import AIMemory
from vision.camera import Camera
from personality.personality import AIPersonality
from utils.speech import SpeechSynthesizer
from utils.sentiment import SentimentAnalyzer
from utils.voice_input import VoiceInput

class BagleyGUI:
    def __init__(self):
        """Initialize Bagley AI GUI"""
        self.window = tk.Tk()
        self.window.title("Bagley AI")
        self.window.geometry("600x500")

        # Chat Display
        self.chat_display = scrolledtext.ScrolledText(self.window, wrap=tk.WORD, height=20, width=65)
        self.chat_display.pack(padx=10, pady=10)
        self.chat_display.config(state=tk.DISABLED)

        # User Input Box
        self.user_input = tk.Entry(self.window, width=50)
        self.user_input.pack(pady=5)
        self.user_input.bind("<Return>", self.process_input)  # Enter key triggers response

        # Buttons
        self.send_button = tk.Button(self.window, text="Send", command=self.process_input)
        self.send_button.pack(pady=5)

        self.voice_button = tk.Button(self.window, text="🎤 Voice Input", command=self.listen_voice)
        self.voice_button.pack(pady=5)

        self.camera_button = tk.Button(self.window, text="📷 Open Camera", command=self.open_camera)
        self.camera_button.pack(pady=5)

        # AI Modules
        self.chatbot = ChatBot()  # Instantiate ChatBot
        self.memory = AIMemory()
        self.camera = Camera(self.chatbot)  # ✅ Pass chatbot to Camera
        self.personality = AIPersonality()
        self.speech = SpeechSynthesizer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.voice_input = VoiceInput()

    def update_chat(self, sender, message):
        """Update chat display with user & AI messages"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}: {message}\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.yview(tk.END)  # Auto-scroll

    def process_input(self, event=None):
        """Handle text input from user"""
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        
        self.update_chat("You", user_text)
        self.user_input.delete(0, tk.END)

        threading.Thread(target=self.generate_response, args=(user_text,)).start()  # Avoid UI freeze

    def generate_response(self, user_text):
        """Generate AI response incorporating memory and personality."""

        # ✅ Convert user input to embedding
        query_embedding = self.chatbot.embed_text(user_text)

        # ✅ Use the correct embedding format in memory search
        relevant_memory = self.memory.search_memory(query_embedding.cpu().numpy())

        memory_context = "\n".join(relevant_memory) if relevant_memory else "No previous context."

        # ✅ Get sentiment analysis results
        sentiment_result = self.sentiment_analyzer.analyze(user_text)  # Corrected variable
        emotion = sentiment_result["facial_emotion"] if sentiment_result["facial_emotion"] != "Neutral" else sentiment_result["text_emotion"]

        # ✅ Store detected emotion in memory (without responding)
        self.chatbot.analyze_emotion(emotion)

        # ✅ Generate AI response
        response = self.chatbot.chat(user_text)

        # ✅ Adjust AI personality based on sentiment
        self.personality.adjust_based_on_interaction(user_text, response, emotion)

        # ✅ Display AI response
        self.update_chat("Bagley AI", response)
        
        # ✅ Use speech synthesis if available
        if hasattr(self.speech, "speak"):
            self.speech.speak(response)
        else:
            print("Error: SpeechSynthesizer has no speak method!")

    def listen_voice(self):
        """Handle voice input"""
        self.update_chat("Bagley AI", "Listening...")
        user_text = self.voice_input.listen()
        if user_text:
            self.update_chat("You", user_text)
            threading.Thread(target=self.generate_response, args=(user_text,)).start()

    def open_camera(self):
        """Open camera for face/object detection"""
        self.update_chat("Bagley AI", "Opening Camera...")
        threading.Thread(target=self.camera.start_camera).start()

    def run(self):
        """Run the GUI"""
        self.window.mainloop()

if __name__ == "__main__":
    gui = BagleyGUI()
    gui.run()
