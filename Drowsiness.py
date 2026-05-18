import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import winsound
import threading

print("Driver Drowsiness Detection System")

# Create face detector
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio"""
    v1 = math.dist(eye_landmarks[1], eye_landmarks[5])
    v2 = math.dist(eye_landmarks[2], eye_landmarks[4])
    h = math.dist(eye_landmarks[0], eye_landmarks[3])
    return (v1 + v2) / (2.0 * h)

def sound_alarm_mild():
    """Mild drowsiness alarm - triple high beep"""
    for _ in range(3):
        winsound.Beep(2500, 200)
        time.sleep(0.1)

def sound_alarm_severe(stop_event):
    """Severe drowsiness alarm - continuous siren until eyes open"""
    while not stop_event.is_set():
        winsound.Beep(800, 300)
        winsound.Beep(1200, 300)
        time.sleep(0.05)

# Parameters
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 15   # Mild alarm
SEVERE_FRAMES = 60        # Severe alarm from prolonged closure
SEVERE_ALARM_COOLDOWN = 5 # Seconds to count multiple mild alarms

eye_closed_counter = 0
alarm_mild_on = False
alarm_severe_on = False
severe_alarm_thread = None
severe_stop_event = None

mild_alarm_count = 0
last_mild_alarm_time = 0

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")
print("System is monitoring...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    if results.face_landmarks:
        face_landmarks = results.face_landmarks[0]

        # Eye landmarks (MediaPipe indices)
        left_eye = [
            (face_landmarks[33].x, face_landmarks[33].y),
            (face_landmarks[160].x, face_landmarks[160].y),
            (face_landmarks[158].x, face_landmarks[158].y),
            (face_landmarks[133].x, face_landmarks[133].y),
            (face_landmarks[153].x, face_landmarks[153].y),
            (face_landmarks[144].x, face_landmarks[144].y)
        ]
        right_eye = [
            (face_landmarks[362].x, face_landmarks[362].y),
            (face_landmarks[385].x, face_landmarks[385].y),
            (face_landmarks[387].x, face_landmarks[387].y),
            (face_landmarks[263].x, face_landmarks[263].y),
            (face_landmarks[373].x, face_landmarks[373].y),
            (face_landmarks[380].x, face_landmarks[380].y)
        ]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Draw eye landmarks
        h, w = frame.shape[:2]
        for eye in [left_eye, right_eye]:
            for point in eye:
                x, y = int(point[0] * w), int(point[1] * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # ---------- Drowsiness Logic ----------
        if ear < EAR_THRESHOLD:
            eye_closed_counter += 1

            # Mild alarm trigger
            if eye_closed_counter >= CONSECUTIVE_FRAMES and not alarm_mild_on and not alarm_severe_on:
                alarm_mild_on = True
                threading.Thread(target=sound_alarm_mild).start()

                current_time = time.time()
                if current_time - last_mild_alarm_time < SEVERE_ALARM_COOLDOWN:
                    mild_alarm_count += 1
                else:
                    mild_alarm_count = 1
                last_mild_alarm_time = current_time

                # Escalate to severe after 3 mild alarms within cooldown
                if mild_alarm_count >= 3:
                    alarm_severe_on = True
                    if severe_stop_event is None:
                        severe_stop_event = threading.Event()
                        severe_alarm_thread = threading.Thread(
                            target=sound_alarm_severe, args=(severe_stop_event,)
                        )
                        severe_alarm_thread.start()

            # Severe alarm from prolonged closure
            if eye_closed_counter >= SEVERE_FRAMES and not alarm_severe_on:
                alarm_severe_on = True
                if severe_stop_event is None:
                    severe_stop_event = threading.Event()
                    severe_alarm_thread = threading.Thread(
                        target=sound_alarm_severe, args=(severe_stop_event,)
                    )
                    severe_alarm_thread.start()

        else:
            # Eyes are open → reset everything
            if eye_closed_counter > 0:
                eye_closed_counter = 0
                alarm_mild_on = False
                if alarm_severe_on:
                    alarm_severe_on = False
                    if severe_stop_event:
                        severe_stop_event.set()
                        severe_alarm_thread.join(timeout=0.5)
                        severe_stop_event = None
                        severe_alarm_thread = None

        # ---------- Display ----------
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if eye_closed_counter > 0:
            cv2.putText(frame, f"Closed: {eye_closed_counter}/{CONSECUTIVE_FRAMES}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        if alarm_mild_on and not alarm_severe_on:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if alarm_severe_on:
            cv2.putText(frame, "!!! STOP THE CAR !!!", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            cv2.putText(frame, "WAKE UP IMMEDIATELY!", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    else:
        cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        eye_closed_counter = 0
        if alarm_severe_on:
            alarm_severe_on = False
            if severe_stop_event:
                severe_stop_event.set()
                severe_alarm_thread.join(timeout=0.5)
                severe_stop_event = None
                severe_alarm_thread = None

    cv2.imshow('Driver Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
if severe_stop_event:
    severe_stop_event.set()
    if severe_alarm_thread:
        severe_alarm_thread.join(timeout=0.5)
cap.release()
cv2.destroyAllWindows()
print("\nMonitoring stopped!")