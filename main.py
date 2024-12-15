import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import os
import time
import json
from threading import Thread

class PersistentGestureTracker:
    def __init__(self):
        # MediaPipe Hands Setup
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

        # Data Collection Setup
        self.output_dir = "comprehensive_3d_hand_data"
        os.makedirs(self.output_dir, exist_ok=True)

        # Recording Variables
        self.gesture_name = None
        self.is_recording = False
        self.recording_data = {
            'metadata': {},
            'landmarks': [],
            'raw_frames': []
        }

        # Depth Feedback Variables
        self.depth_threshold_near = -0.05  # Adjust these values based on your camera
        self.depth_threshold_far = 0.05

    def get_gesture_name(self):
        """
        Open a dialog to input gesture name
        """
        def on_submit():
            nonlocal submitted
            self.gesture_name = entry.get().strip()
            if self.gesture_name:
                submitted = True
                root.quit()
        
        submitted = False
        root = tk.Tk()
        root.title("3D Hand Tracking - Gesture Name")
        root.geometry("300x200")
        
        label = tk.Label(root, text="Enter Gesture Name:", font=("Arial", 12))
        label.pack(padx=20, pady=10)
        
        entry = tk.Entry(root, font=("Arial", 12), width=20)
        entry.pack(padx=20, pady=10)
        
        submit_button = tk.Button(root, text="Start Recording", command=on_submit, font=("Arial", 12))
        submit_button.pack(pady=10)
        
        root.mainloop()
        
        return self.gesture_name if submitted else None

    def get_landmark_color(self, z_value):
        """
        Generate color based on depth (z-coordinate)
        Green: Near, Red: Far, Blue: Neutral
        """
        if z_value < self.depth_threshold_near:
            # Green for very close
            return (0, 255, 0)
        elif z_value > self.depth_threshold_far:
            # Red for very far
            return (0, 0, 255)
        else:
            # Blue for neutral/default position
            return (255, 0, 0)

    def extract_comprehensive_hand_data(self, hand_landmarks, frame_shape):
        """
        Extract detailed 3D hand tracking data with depth information
        """
        h, w, _ = frame_shape
        hand_data = {
            'landmarks': [],
            'additional_metrics': {
                'hand_size': None,
                'palm_orientation': None,
                'hand_area': None
            }
        }

        # Extract 3D coordinates with additional metrics
        for idx, landmark in enumerate(hand_landmarks.landmark):
            # 3D Coordinates
            landmark_data = {
                'name': self.landmark_names[idx],
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility,
                'presence': landmark.presence,
                'pixel_coordinates': {
                    'x': int(landmark.x * w),
                    'y': int(landmark.y * h)
                }
            }
            hand_data['landmarks'].append(landmark_data)

        # Calculate hand size (distance between wrist and middle finger tip)
        wrist = hand_landmarks.landmark[0]
        middle_tip = hand_landmarks.landmark[12]
        hand_data['additional_metrics']['hand_size'] = np.sqrt(
            (wrist.x - middle_tip.x)**2 + 
            (wrist.y - middle_tip.y)**2 + 
            (wrist.z - middle_tip.z)**2
        )

        # Rough palm orientation (using wrist and middle finger MCP)
        middle_mcp = hand_landmarks.landmark[9]
        palm_vector = [
            middle_mcp.x - wrist.x,
            middle_mcp.y - wrist.y,
            middle_mcp.z - wrist.z
        ]
        hand_data['additional_metrics']['palm_orientation'] = palm_vector

        return hand_data

    def capture_3d_hand_data(self):
        """
        Main method to capture comprehensive 3D hand tracking data with depth feedback
        """
        camera = cv2.VideoCapture(0)
        
        # Prompt for initial gesture name
        if not self.gesture_name:
            gesture_name_thread = Thread(target=self.get_gesture_name)
            gesture_name_thread.start()
            gesture_name_thread.join()

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            # Flip and convert frame
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hands
            results = self.hands.process(frameRGB)

            # Visualization
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Extract comprehensive hand data
                    hand_data = self.extract_comprehensive_hand_data(hand_landmarks, frame.shape)

                    # Recording logic
                    if self.is_recording:
                        # Store frame and landmark data
                        self.recording_data['landmarks'].append(hand_data)
                        self.recording_data['raw_frames'].append(frame)

                    # Visualization of landmarks with depth-based color
                    for landmark in hand_landmarks.landmark:
                        h, w, _ = frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        
                        # Get color based on depth
                        color = self.get_landmark_color(landmark.z)
                        
                        # Draw landmark with depth-based color
                        cv2.circle(frame, (cx, cy), 5, color, cv2.FILLED)

                    # Optional: Depth zone visualization
                    cv2.putText(frame, 
                        f"Depth Zones: Green (Near), Blue (Neutral), Red (Far)", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display recording status
            cv2.putText(frame, 
                f"Gesture: {self.gesture_name}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, 
                f"Recording: {'Yes' if self.is_recording else 'No'}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, 
                "Press 'f' to change gesture | 's' to record | 'q' to quit", 
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('3D Hand Tracking', frame)

            # Key controls (unchanged from previous version)
            key = cv2.waitKey(1)
            
            # Change gesture name (F key)
            if key == ord('f'):
                gesture_name_thread = Thread(target=self.get_gesture_name)
                gesture_name_thread.start()
                gesture_name_thread.join()

            # Start recording (S key)
            if key == ord('s'):
                if not self.is_recording and self.gesture_name:
                    # Reset recording data
                    self.recording_data = {
                        'metadata': {
                            'gesture_name': self.gesture_name,
                            'timestamp': time.time(),
                            'recording_duration': 3  # seconds
                        },
                        'landmarks': [],
                        'raw_frames': []
                    }
                    self.is_recording = True
                    print(f"Recording gesture: {self.gesture_name}")

            # Stop recording automatically
            if self.is_recording:
                if len(self.recording_data['landmarks']) >= 60:  # 3 seconds at 30 fps
                    # Save recorded data
                    filename = os.path.join(
                        self.output_dir, 
                        f"{self.gesture_name}_{int(time.time())}"
                    )
                    
                    # Save landmark data
                    np.save(f"{filename}_landmarks.npy", 
                            np.array(self.recording_data['landmarks'], dtype=object))
                    
                    # Optionally save raw frames if needed
                    np.save(f"{filename}_frames.npy", 
                            np.array(self.recording_data['raw_frames']))
                    
                    # Save metadata
                    with open(f"{filename}_metadata.json", 'w') as f:
                        json.dump(self.recording_data['metadata'], f)

                    print(f"Gesture data saved: {filename}")
                    
                    # Reset recording state
                    self.is_recording = False

            # Quit
            if key == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

    def run(self):
        """
        Start the comprehensive 3D hand tracking data collection
        """
        self.capture_3d_hand_data()

# Main execution
if __name__ == "__main__":
    tracker = PersistentGestureTracker()
    tracker.run()