import json
import os

PERSONALITY_FILE = "data/personality.json"

DEFAULT_PERSONALITY = {
    "openness": 0.5,
    "agreeableness": 0.5,
    "emotional_stability": 0.5,
    "humor": 0.5,
    "logic": 0.5
}

class AIPersonality:
    def __init__(self):
        """Initialize AI personality by loading or creating personality traits."""
        self.personality = self.load_personality()

    def load_personality(self):
        """Load personality traits from a JSON file, or create defaults if missing."""
        if not os.path.exists("data"):
            os.makedirs("data")  # Create directory if missing

        if not os.path.exists(PERSONALITY_FILE):
            print("Creating default personality.json file...")
            self.save_personality(DEFAULT_PERSONALITY)  # Save default personality

        with open(PERSONALITY_FILE, "r") as file:
            return json.load(file)

    def save_personality(self, personality_data=None):
        """Save updated personality traits to the JSON file."""
        if personality_data is None:
            personality_data = self.personality
        with open(PERSONALITY_FILE, "w") as file:
            json.dump(personality_data, file, indent=4)

    def adjust_based_on_interaction(self, user_input, ai_response, emotion):
        """Modify personality traits based on user interactions."""
        user_input = user_input.lower()

        if "thank you" in user_input or emotion == "positive":
            self.personality["agreeableness"] = min(1.0, self.personality["agreeableness"] + 0.05)
            self.personality["emotional_stability"] = min(1.0, self.personality["emotional_stability"] + 0.03)
        elif "angry" in emotion or "frustrated" in emotion:
            self.personality["agreeableness"] = max(0.0, self.personality["agreeableness"] - 0.05)
            self.personality["emotional_stability"] = max(0.0, self.personality["emotional_stability"] - 0.03)
        elif "debate" in user_input or "logical" in ai_response:
            self.personality["logic"] = min(1.0, self.personality["logic"] + 0.05)
            self.personality["openness"] = min(1.0, self.personality["openness"] + 0.03)
        elif "joke" in user_input or "funny" in ai_response:
            self.personality["humor"] = min(1.0, self.personality["humor"] + 0.04)

        self.save_personality()

    def get_personality_prompt(self):
        """Generate a string that describes the AI's personality traits for the LLM."""
        return (
            f"My personality traits: Openness={self.personality['openness']}, "
            f"Agreeableness={self.personality['agreeableness']}, "
            f"Emotional Stability={self.personality['emotional_stability']}, "
            f"Humor={self.personality['humor']}, Logic={self.personality['logic']}."
        )
