import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Adjusted thresholds and constants for punch detection and ready stance
MOVEMENT_THRESHOLD = 0.15  # More strict on movement for ready stance
PUNCH_SPEED_THRESHOLD = 3.0  # Higher punch speed threshold for less sensitivity to minor movements
FINISH_SPEED_THRESHOLD = 0.3  # Lower sensitivity to small movements for punch finishing
UPPERCUT_ELBOW_THRESHOLD = 92  # Elbow angle threshold for uppercuts and hooks
READY_ELBOW_THRESHOLD = 85  # Elbow angle threshold for "Ready" stance
VISIBILITY_THRESHOLD = 0.6  # Threshold for checking if keypoints are visible in the frame

# Initialize the video capture from the default camera (0)
cap = cv2.VideoCapture(0)

# Initialize previous landmarks and punch state tracking
previous_landmarks = None
punch_history = []  # To store all punches detected
punch_active = False  # Track whether a punch is currently active
waiting_for_ready = False  # Flag to track if we are waiting for the user to return to the "Ready" stance

# Function to calculate the angle between three points (used for elbow angle)
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (e.g., shoulder)
    b = np.array(b)  # Middle point (e.g., elbow)
    c = np.array(c)  # Third point (e.g., wrist)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

# Calculate punch speed based on hand movement
def calculate_punch_speed(start_landmark, end_landmark, start_time, end_time):
    distance = np.sqrt(
        (end_landmark.x - start_landmark.x) ** 2 +
        (end_landmark.y - start_landmark.y) ** 2 +
        (end_landmark.z - start_landmark.z) ** 2
    )
    time_diff = end_time - start_time
    if time_diff == 0:  # Avoid division by zero
        return 0
    return distance / time_diff

# Function to detect if hands are in a "Ready" position (both hands above shoulders, elbows bent < 85 degrees)
def in_ready_position(pose_landmarks):
    left_hand = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_hand = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow = pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Check if hands are above the shoulders
    hands_above_shoulders = left_hand.y < left_shoulder.y and right_hand.y < right_shoulder.y

    # Calculate elbow angles
    left_elbow_angle = calculate_angle([left_shoulder.x, left_shoulder.y], 
                                       [left_elbow.x, left_elbow.y], 
                                       [left_hand.x, left_hand.y])
    
    right_elbow_angle = calculate_angle([right_shoulder.x, right_shoulder.y], 
                                        [right_elbow.x, right_elbow.y], 
                                        [right_hand.x, right_hand.y])

    # Check if elbows are bent less than 85 degrees
    elbows_bent = left_elbow_angle < READY_ELBOW_THRESHOLD and right_elbow_angle < READY_ELBOW_THRESHOLD

    return hands_above_shoulders and elbows_bent

# Function to check if hips and shoulders are visible in the frame
def is_user_in_frame(pose_landmarks):
    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    
    if (left_shoulder.visibility < VISIBILITY_THRESHOLD or
        right_shoulder.visibility < VISIBILITY_THRESHOLD or
        left_hip.visibility < VISIBILITY_THRESHOLD or
        right_hip.visibility < VISIBILITY_THRESHOLD):
        return False
    return True

# Refined punch detection logic for jab, cross, hook, and uppercut
def detect_punch(pose_landmarks, previous_landmarks, start_time, end_time):
    left_hand = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_hand = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow = pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Calculate the angle of the right elbow for hooks and uppercuts
    elbow_angle = calculate_angle([right_shoulder.x, right_shoulder.y],
                                  [right_elbow.x, right_elbow.y],
                                  [right_hand.x, right_hand.y])

    right_hand_speed = calculate_punch_speed(previous_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST], right_hand, start_time, end_time)
    left_hand_speed = calculate_punch_speed(previous_landmarks[mp_pose.PoseLandmark.LEFT_WRIST], left_hand, start_time, end_time)

    # Detect punches based on movement and elbow angle
    if left_hand_speed > PUNCH_SPEED_THRESHOLD:
        # Jab: Left hand extends forward with minimal lateral movement
        if left_hand.x > left_shoulder.x and abs(left_hand.y - left_shoulder.y) < MOVEMENT_THRESHOLD:
            return "jab"
        
    if right_hand_speed > PUNCH_SPEED_THRESHOLD:
        # Cross: Right hand extends forward with minimal lateral movement
        if right_hand.x > right_shoulder.x and abs(right_hand.y - right_shoulder.y) < MOVEMENT_THRESHOLD:
            return "cross"
        # Hook: Elbow angle < 92 degrees, hand crosses the body centerline
        elif elbow_angle < UPPERCUT_ELBOW_THRESHOLD and right_hand.x < right_shoulder.x:
            return "hook"
        # Uppercut: Elbow angle < 92 degrees, punch starts below chest and ends above
        elif elbow_angle < UPPERCUT_ELBOW_THRESHOLD and right_hand.y > right_hip.y and right_hand.y < right_shoulder.y:
            return "uppercut"

    return "unknown"

# Main loop
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        punch_detected = "unknown"
        
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Check if the user is in frame (hips and shoulders visible)
            if not is_user_in_frame(result.pose_landmarks.landmark):
                cv2.putText(frame, "Please step back, hips not in frame", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Shadow-Boxing', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                continue  # Skip punch detection if user is not in frame

            # Initialize landmarks for the first frame
            if previous_landmarks is None:
                previous_landmarks = result.pose_landmarks.landmark
                start_time = time.time()

            # Check if the user is in "Ready" stance before detecting a new punch
            if waiting_for_ready:
                if in_ready_position(result.pose_landmarks.landmark):
                    waiting_for_ready = False  # User is ready again
                    cv2.putText(frame, "Ready", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Return to Ready Stance", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Shadow-Boxing', frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                    continue  # Wait until the user returns to the ready stance

            # Detect punch or elbow
            end_time = time.time()
            punch_detected = detect_punch(result.pose_landmarks.landmark, previous_landmarks, start_time, end_time)

            # Update landmarks for the next iteration
            previous_landmarks = result.pose_landmarks.landmark
            start_time = end_time

            # Determine if a punch is active or finished based on speed
            if punch_detected != "unknown":
                punch_active = True
                punch_history.append(punch_detected)
                waiting_for_ready = True  # Set flag to wait for the user to return to ready stance
            elif punch_active:
                # Check if punch is finished (hands return to guard or speed drops)
                right_hand_speed = calculate_punch_speed(previous_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST], 
                                                         result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST], 
                                                         start_time, end_time)
                left_hand_speed = calculate_punch_speed(previous_landmarks[mp_pose.PoseLandmark.LEFT_WRIST], 
                                                        result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST], 
                                                        start_time, end_time)

                if right_hand_speed < FINISH_SPEED_THRESHOLD and left_hand_speed < FINISH_SPEED_THRESHOLD and in_ready_position(result.pose_landmarks.landmark):
                    punch_active = False
                    print("Punch finished!")

            # Display punch history on the screen
            for i, punch in enumerate(punch_history[-10:]):  # Show the last 10 punches
                cv2.putText(frame, punch, (30, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Shadow-Boxing', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
