import asyncio
import edge_tts
import pygame.mixer
import io
import re

class SpeechSynthesizer:
    def __init__(self, voice="default"):
        """Initialize the TTS engine."""
        self.voice = "en-GB-RyanNeural" if voice == "male" or voice == "default" else "en-GB-SoniaNeural"
        self.rate = "-2%"  # Slightly slower for more natural speech
        self.volume = "+0%"
        pygame.mixer.init()
        self.is_speaking = False

    def stop(self):
        """Stop current speech"""
        self.is_speaking = False
        pygame.mixer.music.stop()

    async def _speak_async(self, text):
        """Internal async method to handle TTS."""
        self.is_speaking = True
        communicate = edge_tts.Communicate(
            text, 
            self.voice, 
            rate=self.rate, 
            volume=self.volume,
            pitch="+0Hz"
        )
        audio_data = io.BytesIO()
        
        async for chunk in communicate.stream():
            if not self.is_speaking:
                break
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        
        if self.is_speaking:  # Only play if not stopped
            audio_data.seek(0)
            pygame.mixer.music.load(audio_data)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and self.is_speaking:
                pygame.time.Clock().tick(10)

    def speak(self, text):
        """Convert text to speech with natural conversation patterns."""
        if text.strip():
            # Process text for more natural speech
            processed_text = self._process_text_for_natural_speech(text)
            asyncio.run(self._speak_async(processed_text))

    def _process_text_for_natural_speech(self, text):
        # Simple text cleanup
        text = text.strip()
        
        # Basic punctuation handling
        sentences = text.split('.')
        processed_text = '. '.join(sentence.strip() for sentence in sentences if sentence.strip())
        
        return processed_text

# Example Usage
if __name__ == "__main__":
    tts = SpeechSynthesizer()  # Will now use male voice by default
    tts.speak("Hello! I am Bagley AI. How can I assist you today? Let me know if you need anything!")
