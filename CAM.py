import cv2
import threading
from onvif import ONVIFCamera
from pynput import keyboard

# Camera credentials and IP
IP = '192.168.1.110'
PORT = 80
USERNAME = 'admin'
PASSWORD = 'Abc112233'

# Movement speed
SPEED = 0.5

# Connect to camera
camera = ONVIFCamera(IP, PORT, USERNAME, PASSWORD)
media_service = camera.create_media_service()
ptz_service = camera.create_ptz_service()

# Get profile
profile = media_service.GetProfiles()[0]
token = profile.token

# Stop movement
def stop_move():
    ptz_service.Stop({'ProfileToken': token})

# Move in direction
def move(x, y):
    request = ptz_service.create_type('ContinuousMove')
    request.ProfileToken = token
    request.Velocity = {'PanTilt': {'x': x, 'y': y}}
    ptz_service.ContinuousMove(request)
    threading.Timer(0.5, stop_move).start()  # Move for 0.5 seconds

# Keyboard event handler
def on_press(key):
    try:
        if key.char == 'w':
            move(0, SPEED)  # tilt up
        elif key.char == 's':
            move(0, -SPEED)  # tilt down
        elif key.char == 'a':
            move(-SPEED, 0)  # pan left
        elif key.char == 'd':
            move(SPEED, 0)  # pan right
    except AttributeError:
        pass

# Start listening to keyboard in a non-blocking way
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Open live RTSP stream in a window (optional)
RTSP_URL = f'rtsp://{USERNAME}:{PASSWORD}@{IP}:554/cam/realmonitor?channel=1&subtype=0'
cap = cv2.VideoCapture(RTSP_URL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Amcrest Camera Feed (WASD to move)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
listener.stop()
