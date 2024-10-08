import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Create output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# Process each video in the input directory
input_dir = r'C:\Users\Davie\Desktop\pytone\pose_estimator\input'
output_dir = r'C:\Users\Davie\Desktop\pytone\pose_estimator\output'

for video_file in os.listdir(input_dir):
    if video_file.endswith('.mp4'):
        cap = cv2.VideoCapture(os.path.join(input_dir, video_file))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        out = cv2.VideoWriter(os.path.join(output_dir, video_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Set up the Pose model
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
              
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    # Calculate angle
                    angle = calculate_angle(shoulder, elbow, wrist)
                    
                    # Visualize angle
                    cv2.putText(image, str(int(angle)), 
                                tuple(np.multiply(elbow, [width, height]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                    
                    # Add indicators for left hand, right hand, and head
                    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
                    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
                    head = landmarks[mp_pose.PoseLandmark.NOSE.value]

                    def draw_indicator(landmark, label):
                        pos = tuple(np.multiply([landmark.x, landmark.y], [width, height]).astype(int))
                        cv2.circle(image, pos, 8, (0, 200, 0), -1)
                        cv2.putText(image, label, (pos[0]+10, pos[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 205, 0), 1, cv2.LINE_AA)

                    draw_indicator(left_hand, "Left Hand")
                    draw_indicator(right_hand, "Right Hand")
                    draw_indicator(head, "Head")
                                       
                except:
                    pass
                
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                         )               

                out.write(image)

        cap.release()
        out.release()

cv2.destroyAllWindows()
