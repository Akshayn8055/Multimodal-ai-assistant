import torch
import google.generativeai as genai
import config
from memory.memory import AIMemory

# Configure Gemini API
genai.configure(api_key=config.GEMINI_API_KEY)

class ChatBot:
    def __init__(self):
        """Initialize ChatBot with AI memory."""
        self.memory = AIMemory()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = genai.GenerativeModel("gemini-1.5-pro")  # Use a valid model

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
            response = self.model.generate_content(prompt)  # Correct API call
            ai_response = response.text.strip() if response and hasattr(response, "text") else "Error: No response received."
        except Exception as e:
            ai_response = f"Error generating response: {str(e)}"

        self.memory.add_memory(user_input, query_embedding.cpu().numpy())
        return ai_response

    def analyze_emotion(self, detected_emotion):
        """Store detected emotion for context awareness without responding."""
        # Just store the detected emotion in memory for awareness
        self.memory.add_memory(f"Detected emotion: {detected_emotion}", torch.rand(512, device=self.device).cpu().numpy())

    def respond_to_emotion(self, emotion):
        """Dummy function to prevent errors."""
        return "Emotion response not implemented."

    def shutdown(self):
        """Save all data and perform cleanup before shutting down."""
        self.memory.save_memory()  # Assuming there's a method to save memory
        # Additional cleanup code can go here
        print("Shutting down the application...")
        exit(0)  # Graceful exit
