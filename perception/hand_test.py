import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

# The 21-point hand skeleton: pairs of landmark indices to connect
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm edge
]

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
start = time.time()
prev = start

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)  # mirror so it feels natural
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int((time.time() - start) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w = frame.shape[:2]
    for hand in result.hand_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in hand]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    now = time.time()
    fps = 1.0 / max(now - prev, 1e-6)
    prev = now
    cv2.putText(frame, f"hands: {len(result.hand_landmarks)}  fps: {fps:.0f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("hand test (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()