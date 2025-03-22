from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import cv2
import mediapipe as mp

class SentimentAnalyzer:
    def __init__(self, use_bert=False):
        """Initialize sentiment analysis with VADER, optional BERT, and facial emotion detection."""
        self.vader = SentimentIntensityAnalyzer()
        self.use_bert = use_bert
        self.bert_analyzer = pipeline("sentiment-analysis") if use_bert else None

        # Initialize MediaPipe FaceMesh for emotion detection
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
        self.mp_drawing = mp.solutions.drawing_utils

    def analyze_text(self, text):
        """Analyze sentiment using VADER and optionally BERT."""
        if not isinstance(text, str) or not text.strip():
            return {"error": "Invalid input", "vader": 0.0}

        vader_score = self.vader.polarity_scores(text)["compound"]
        bert_result = None
        if self.use_bert and self.bert_analyzer:
            bert_analysis = self.bert_analyzer(text)[0]
            bert_result = {"label": bert_analysis["label"], "confidence": bert_analysis["score"]}

        return {"vader": vader_score, "bert": bert_result}

    def interpret_score(self, sentiment):
        """Convert numerical sentiment score to human-readable emotion."""
        vader_score = sentiment.get("vader", 0.0)
        if vader_score >= 0.05:
            return "Positive"
        elif vader_score <= -0.05:
            return "Negative"
        return "Neutral"

    def analyze_facial_expression(self, frame):
        """Detect emotions based on facial landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            return "Engaged"  # Placeholder for facial emotion analysis
        return "Neutral"

    def analyze(self, text, frame=None):
        """Combine text sentiment and facial emotion detection."""
        text_sentiment = self.analyze_text(text)  # ✅ Returns a dictionary
        text_emotion = self.interpret_score(text_sentiment)  # ✅ Returns a string

        facial_emotion = "Neutral"
        if frame is not None:
            facial_emotion = self.analyze_facial_expression(frame)  # ✅ Returns a string

        return {  # ✅ Now returns a dictionary instead of a string
            "text_sentiment": text_sentiment,
            "text_emotion": text_emotion,
            "facial_emotion": facial_emotion
        }

# Example Usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer(use_bert=False)
    text = "I love talking to Bagley AI!"
    sentiment = analyzer.analyze(text)
    print(f"Sentiment Analysis: {sentiment}")
