import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    if angle < 0:
        angle += 360
    return angle

# Start webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for pose estimation
    results = pose.process(rgb_frame)

    # Get frame dimensions
    height, width, _ = frame.shape

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        )

        # Extract key points for pose classification
        landmarks = results.pose_landmarks.landmark

        def get_coords(index):
            return [landmarks[index].x * width, landmarks[index].y * height]

        left_shoulder = get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_elbow = get_coords(mp_pose.PoseLandmark.LEFT_ELBOW)
        right_elbow = get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW)
        left_wrist = get_coords(mp_pose.PoseLandmark.LEFT_WRIST)
        right_wrist = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)
        left_hip = get_coords(mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = get_coords(mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee = get_coords(mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = get_coords(mp_pose.PoseLandmark.RIGHT_KNEE)
        left_ankle = get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)

        # Calculate angles for pose detection
        angle_left_elbow = calculate_angle(left_shoulder, left_elbow, left_wrist)
        angle_right_elbow = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Initialize pose name
        pose_name = "Unknown Pose"

        # Pose Detection Logic

        # Detect T-Pose (Arms outstretched horizontally)
        if (angle_left_elbow > 160 and angle_right_elbow > 160 and 
            abs(left_shoulder[1] - right_shoulder[1]) < 50):
            pose_name = "T-Pose"

        # Detect Namaste Pose (Hands together in front)
        elif (angle_left_elbow < 60 and angle_right_elbow < 60 and
              abs(left_wrist[0] - right_wrist[0]) < 50 and 
              abs(left_wrist[1] - right_wrist[1]) < 50):
            pose_name = "Namaste"

        # Detect Wave Hand (One hand raised above head)
        elif (left_wrist[1] < left_shoulder[1] and angle_left_elbow > 140):
            pose_name = "Wave Hand (Left)"
        elif (right_wrist[1] < right_shoulder[1] and angle_right_elbow > 140):
            pose_name = "Wave Hand (Right)"

        # Detect Chair Pose (Knees bent, arms raised)
        elif (left_knee[1] > left_hip[1] and right_knee[1] > right_hip[1] and
              angle_left_elbow > 160 and angle_right_elbow > 160 and 
              left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]):
            pose_name = "Chair Pose"

        # Detect Plank Pose (Straight body, arms extended)
        elif (abs(left_hip[1] - right_hip[1]) < 50 and
              abs(left_shoulder[1] - right_shoulder[1]) < 50 and
              angle_left_elbow > 160 and angle_right_elbow > 160):
            pose_name = "Plank Pose"

        # Detect Warrior Pose (One arm extended, other bent)
        elif (angle_left_elbow < 90 and right_wrist[1] < right_shoulder[1]):
            pose_name = "Warrior Pose"

        # Detect Superman Pose (Arms and legs stretched forward)
        elif (angle_left_elbow > 160 and angle_right_elbow > 160 and
              left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1] and
              left_ankle[1] < left_knee[1] and right_ankle[1] < right_knee[1]):
            pose_name = "Superman Pose"

        # Display the pose name on the frame
        cv2.putText(
            frame,
            pose_name,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # Display the frame
    cv2.imshow('Pose Detection', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

