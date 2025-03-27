import cv2
import threading
import torch
import numpy as np
from deepface import DeepFace
import mediapipe as mp

class Camera:
    def __init__(self, ai_chatbot):
        """Initialize the camera with face, emotion, and object detection."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")

        self.running = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.ai_chatbot = ai_chatbot  # Link to AI chatbot for conversation

    def start_camera(self):
        """Start the camera feed in a separate thread."""
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        """Capture video frames and process them with GPU acceleration."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Convert frame to RGB for DeepFace & MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces using MediaPipe
            results = self.face_detector.process(rgb_frame)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, c = frame.shape
                    x, y, w, h = (int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Extract face ROI for emotion detection
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size > 0:
                        try:
                            emotion_result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                            dominant_emotion = emotion_result[0]['dominant_emotion']
                            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            print(f"Detected Emotion: {dominant_emotion}")
                            
                            # Trigger AI conversation if an emotion is detected
                            self.ai_chatbot.respond_to_emotion(dominant_emotion)
                        except Exception as e:
                            print("Emotion Detection Error:", e)

            cv2.imshow("Bagley AI Vision (Face & Emotion Detection)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()

    def stop(self):
        """Stop the camera feed."""
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

# Example Usage
if __name__ == "__main__":
    class DummyAI:
        def respond_to_emotion(self, emotion):
            print(f"AI Responding to Emotion: {emotion}")
    
    cam = Camera(ai_chatbot=DummyAI())
    cam.start_camera()