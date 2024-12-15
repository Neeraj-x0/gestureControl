import cv2
import numpy as np

# Returns action text, thumb and index tip coordinates, overall distance
def detect_action(id, lm, frame, x, y, thumb_tip, index_tip, finger_array):
    lmx = int(lm.x * x)
    lmy = int(lm.y * y)
    action = ""
    
    if id == 4:  # Thumb tip
        thumb_tip = (lmx, lmy)
        cv2.circle(frame, thumb_tip, 10, (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, "Thumb", (lmx - 20, lmy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if id == 8:  # Index finger tip
        index_tip = (lmx, lmy)
        cv2.circle(frame, index_tip, 10, (255, 0, 0), cv2.FILLED)
        cv2.putText(frame, "Index", (lmx - 20, lmy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Add the landmark coordinates to the array
    finger_array.append((lmx, lmy))

    # Calculate the distance between thumb and index finger
    ThumbIndex_distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
    #print(ThumbIndex_distance)

    # Check if the thumb and index fingers are close
    if ThumbIndex_distance < 40:
        action = "Hold"
    
    # Calculate the overall distance between all landmarks
    overall_distance = get_overall_distance(finger_array)
    
    if overall_distance < 850:
        action = "Crush"
    
    return action, ThumbIndex_distance, thumb_tip, index_tip, overall_distance

# Function to calculate the overall distance between all hand landmarks
def get_overall_distance(finger_array):
    total_distance = 0
    # Calculate the distance between consecutive landmarks
    for i in range(1, len(finger_array)):
        total_distance += np.linalg.norm(np.array(finger_array[i]) - np.array(finger_array[i-1]))
    return total_distance


import numpy as np

def prepare_landmarks_for_prediction(landmarks):
    # Flatten the input landmarks (already flattened)
    flattened = landmarks.flatten()

    # Expected shape (1, 10, 63) for the model
    # Create a zero-filled array with the expected shape
    padded_landmarks = np.zeros((1, 10, 63), dtype=np.float32)
    
    # Assuming flattened array needs to be filled into 10 data points with 63 features each
    # Copy the flattened landmarks into the padded_landmarks
    # This part may need to be adjusted depending on your actual data and model's requirements
    # For example, filling the flattened array into padded_landmarks[0, 0, :63] and so on
    padded_landmarks[0, 0, :flattened.size] = flattened
    
    return padded_landmarks
