import speech_recognition as sr

class VoiceInput:
    def __init__(self):
        """Initialize the speech recognizer"""
        self.recognizer = sr.Recognizer()

    def listen(self):
        """Listen to microphone and convert speech to text"""
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)

        try:
            text = self.recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError:
            print("Speech recognition service error")
            return None

# Example Usage
if __name__ == "__main__":
    voice = VoiceInput()
    voice.listen()
