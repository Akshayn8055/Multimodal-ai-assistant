from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import cv2
import mediapipe as mp
import face_recognition
import pickle
import os

# Path to store known faces
KNOWN_FACES_DIR = "vision/known_faces"
FACE_ENCODINGS_FILE = "vision/face_encodings.pkl"

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
        text_sentiment = self.analyze_text(text)
        text_emotion = self.interpret_score(text_sentiment)
        
        facial_emotion = "Neutral"
        if frame is not None:
            facial_emotion = self.analyze_facial_expression(frame)

        return facial_emotion if facial_emotion != "Neutral" else text_emotion

# Load known faces from storage
def load_known_faces():
    """Loads stored face encodings and names."""
    if os.path.exists(FACE_ENCODINGS_FILE):
        with open(FACE_ENCODINGS_FILE, "rb") as file:
            known_faces = pickle.load(file)
    else:
        known_faces = {"encodings": [], "names": []}
    return known_faces

def save_known_faces(known_faces):
    """Saves face encodings to file."""
    with open(FACE_ENCODINGS_FILE, "wb") as file:
        pickle.dump(known_faces, file)

def add_new_face(image_path, name):
    """Encodes a new face from an image and saves it."""
    known_faces = load_known_faces()
    
    # Load the image and encode the face
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_faces["encodings"].append(encodings[0])
        known_faces["names"].append(name)
        save_known_faces(known_faces)
        print(f"Face added for {name}.")
    else:
        print("No face detected in the image.")

def recognize_faces():
    """Detects and recognizes faces in real-time from the webcam."""
    video_capture = cv2.VideoCapture(0)
    known_faces = load_known_faces()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces["encodings"], face_encoding)
            name = "Unknown"

            # Find the best match
            if True in matches:
                match_index = matches.index(True)
                name = known_faces["names"][match_index]

            # Draw a box around the face and label it
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the live feed
        cv2.imshow("Face Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
