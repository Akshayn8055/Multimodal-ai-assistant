import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

class ObjectDetector:
    def __init__(self):
        self.model = MobileNetV2(weights='imagenet')
        self.target_size = (224, 224)

    def detect(self, frame):
        """Detect objects in the given frame, focusing on specific regions and using a confidence threshold."""
        # Preprocess the frame
        img = cv2.resize(frame, self.target_size)
        img = img.astype(np.uint8)  # Convert to appropriate depth
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Predict objects
        predictions = self.model.predict(img)
        results = decode_predictions(predictions, top=3)[0]

        # Filter results based on confidence threshold
        confidence_threshold = 0.5
        filtered_results = [result[1] for result in results if result[2] >= confidence_threshold]

        # Return filtered detected objects
        return filtered_results