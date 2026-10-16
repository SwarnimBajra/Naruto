import cv2
import mediapipe as mp

print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
print("Camera opened:", ok, "| frame shape:", None if not ok else frame.shape)
cap.release()