import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import mediapipe as mp

class GestureRecognitionModel:
    def __init__(self, data_dir='comprehensive_3d_hand_data'):
        """
        Initialize the gesture recognition model training pipeline
        
        Args:
            data_dir (str): Directory containing collected gesture data
        """
        self.data_dir = data_dir
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=2, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7
        )
        
        # Landmark Names (21 landmarks)
        self.landmark_names = [
            "WRIST",
            "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
            "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]

    def load_collected_data(self):
        """
        Load and preprocess collected gesture data
        
        Returns:
            tuple: (X, y) where X is feature matrix and y is label vector
        """
        X = []
        y = []
        
        # Collect all .npy landmark files
        landmark_files = [f for f in os.listdir(self.data_dir) if f.endswith('_landmarks.npy')]
        
        for file in landmark_files:
            # Extract gesture name from filename
            gesture_name = file.split('_')[0]
            
            # Load landmark data
            landmarks_data = np.load(os.path.join(self.data_dir, file), allow_pickle=True)
            
            # Process each recording (assumes 3 seconds of data)
            for recording in landmarks_data:
                # Flatten 3D coordinates of all landmarks into a single vector
                flattened_landmarks = []
                for landmark in recording['landmarks']:
                    # Extract x, y, z coordinates and visibility
                    flattened_landmarks.extend([
                        landmark['x'], 
                        landmark['y'], 
                        landmark['z'], 
                        landmark['visibility']
                    ])
                
                X.append(flattened_landmarks)
                y.append(gesture_name)
        
        return np.array(X), np.array(y)

    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Prepare data for model training
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Label vector
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
        
        Returns:
            tuple: Prepared training and testing datasets
        """
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test, label_encoder

    def create_model(self, input_shape, num_classes):
        """
        Create a neural network model for gesture recognition
        
        Args:
            input_shape (tuple): Shape of input features
            num_classes (int): Number of gesture classes
        
        Returns:
            tf.keras.Model: Compiled neural network model
        """
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train_model(self):
        """
        Complete model training pipeline
        """
        # Load and preprocess data
        X, y = self.load_collected_data()
        
        # Prepare data
        X_train, X_test, y_train, y_test, label_encoder = self.prepare_data(X, y)
        
        # Create model
        model = self.create_model(
            input_shape=(X_train.shape[1],), 
            num_classes=len(np.unique(y))
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
        
        # Save model and label encoder
        model.save('gesture_recognition_model.h5')
        import joblib
        joblib.dump(label_encoder, 'gesture_label_encoder.pkl')
        
        return model, label_encoder

    def real_time_prediction(self, model_path='gesture_recognition_model.h5', 
                              encoder_path='gesture_label_encoder.pkl'):
        """
        Real-time gesture prediction using trained model
        
        Args:
            model_path (str): Path to saved model
            encoder_path (str): Path to saved label encoder
        """
        # Load model and label encoder
        model = keras.models.load_model(model_path)
        import joblib
        label_encoder = joblib.load(encoder_path)
        
        # Open camera
        camera = cv2.VideoCapture(0)
        
        while True:
            # Capture frame
            ret, frame = camera.read()
            if not ret:
                break
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands
            results = self.hands.process(frameRGB)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([
                            landmark.x, 
                            landmark.y, 
                            landmark.z, 
                            landmark.visibility
                        ])
                    
                    # Predict gesture
                    landmarks_array = np.array(landmarks).reshape(1, -1)
                    prediction = model.predict(landmarks_array)
                    predicted_class = label_encoder.inverse_transform(
                        [np.argmax(prediction)]
                    )[0]
                    confidence = np.max(prediction)
                    
                    # Display prediction
                    cv2.putText(
                        frame, 
                        f"Gesture: {predicted_class} (Conf: {confidence:.2f})", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2
                    )
            
            # Show frame
            cv2.imshow('Gesture Recognition', frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        camera.release()
        cv2.destroyAllWindows()

def main():
    # Create gesture recognition pipeline
    gesture_ml = GestureRecognitionModel()
    
    # Train the model
    print("Training Gesture Recognition Model...")
    model, label_encoder = gesture_ml.train_model()
    
    # Real-time prediction
    print("Starting Real-Time Gesture Recognition...")
    gesture_ml.real_time_prediction()

if __name__ == "__main__":
    main()