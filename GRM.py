import cv2
import mediapipe as mp
import numpy as np

# Replace with your actual camera stream URL
RTSP_URL = "rtsp://admin:Abc112233@192.168.1.110:554/cam/realmonitor?channel=1&subtype=1"

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def fingers_status(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb (check x direction)
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(True)
    else:
        fingers.append(False)

    # Other fingers (check y direction)
    for i in range(1, 5):
        if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y:
            fingers.append(True)
        else:
            fingers.append(False)

    return fingers

def classify_gesture(fingers):
    if all(fingers):
        return "open_palm"
    if not any(fingers):
        return "fist"
    if fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:
        return "peace"
    if not fingers[0] and fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
        return "three_up"
    return None


def main():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)  # Use FFMPEG backend for better RTSP handling

    if not cap.isOpened():
        print("❌ Could not open RTSP stream.")
        return

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to grab frame")
                break

            # Resize for faster processing (optional)
            frame = cv2.resize(frame, (640, 480))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            gesture = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    fingers = fingers_status(hand_landmarks)
                    gesture = classify_gesture(fingers)

            if gesture:
                cv2.putText(image, f'Gesture: {gesture}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("RTSP Gesture Recognition", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
