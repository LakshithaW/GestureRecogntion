import cv2
import numpy as np
import mediapipe as mp
from onvif import ONVIFCamera
import time

# --------- CONFIG ---------
RTSP_URL = "rtsp://admin:Abc112233@192.168.1.110:554/cam/realmonitor?channel=1&subtype=0"

# ONVIF PTZ Setup
def setup_camera():
    cam = ONVIFCamera('192.168.1.110', 80, 'admin', 'Abc112233')
    media_service = cam.create_media_service()
    ptz_service = cam.create_ptz_service()
    profile_token = media_service.GetProfiles()[0].token
    return ptz_service, profile_token

def move_camera(ptz, token, x, y):
    request = ptz.create_type('ContinuousMove')
    request.ProfileToken = token
    request.Velocity = {'PanTilt': {'x': x, 'y': y}}
    ptz.ContinuousMove(request)
    time.sleep(0.3)  # short movement
    ptz.Stop({'ProfileToken': token})

# --------- Gesture Detection Utils ---------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def fingers_status(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0]-1].x:
        fingers.append(True)
    else:
        fingers.append(False)

    # Other fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers.append(True)
        else:
            fingers.append(False)

    return fingers

def classify_gesture(fingers):
    if all(fingers):
        return 'open_palm'  # Move up
    if not any(fingers):
        return 'fist'       # Move down
    if fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:
        return 'peace'      # Move left
    if not fingers[0] and fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
        return "three_up"
    return None

# --------- Main ---------
def main():
    ptz_service, ptz_token = setup_camera()
    cap = cv2.VideoCapture(RTSP_URL)

    last_gesture = None

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            gesture = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    fingers = fingers_status(hand_landmarks)
                    gesture = classify_gesture(fingers)

                    if gesture and gesture != last_gesture:
                        last_gesture = gesture

                        # Move camera based on gesture (small step)
                        if gesture == 'open_palm':
                            move_camera(ptz_service, ptz_token, 0, 0.5)   # Up
                        elif gesture == 'fist':
                            move_camera(ptz_service, ptz_token, 0, -0.5)  # Down
                        elif gesture == 'peace':
                            move_camera(ptz_service, ptz_token, -0.5, 0)  # Left
                        elif gesture == 'three_up':
                            move_camera(ptz_service, ptz_token, 0.5, 0)   # Right
                    break  # process only first hand

            # Display gesture
            if gesture:
                cv2.putText(frame, f'Gesture: {gesture}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

            cv2.imshow('Amcrest PTZ Gesture Control', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
