import csv
import os
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"
OUT_PATH = "data/spike_mediapipe.csv"
SEALS = ["Rat", "Ox", "Tiger", "Hare", "Dragon", "Snake",
         "Horse", "Ram", "Monkey", "Bird", "Dog", "Boar"]
COUNTDOWN_S = 3.0
RECORD_S = 5.0

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def new_stats():
    return {"frames": 0, "two": 0, "one": 0, "zero": 0,
            "drops": 0, "scores": [], "jitter": [], "t_start": time.time()}


def summarize(seal, s):
    n = max(s["frames"], 1)
    dur = max(time.time() - s["t_start"], 1e-6)
    return {
        "seal": seal,
        "frames": s["frames"],
        "fps": round(s["frames"] / dur, 1),
        "pct_two_hands": round(100 * s["two"] / n, 1),
        "pct_one_hand": round(100 * s["one"] / n, 1),
        "pct_no_hands": round(100 * s["zero"] / n, 1),
        "hand_count_changes": s["drops"],
        "mean_handedness_score": round(float(np.mean(s["scores"])), 3) if s["scores"] else 0.0,
        "mean_jitter": round(float(np.mean(s["jitter"])), 4) if s["jitter"] else 0.0,
    }


def main():
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    results = {}
    idx = 0
    state = "ready"          # ready -> countdown -> record -> ready
    t_state = time.time()
    stats = None
    prev_count = None
    prev_pts = None
    start = time.time()
    last_ts = -1

    while idx < len(SEALS):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - start) * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        res = landmarker.detect_for_video(mp_image, ts)

        hands = res.hand_landmarks
        n = len(hands)

        # draw skeletons
        for hand in hands:
            pts = [(int(p.x * w), int(p.y * h)) for p in hand]
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

        now = time.time()
        seal = SEALS[idx]

        if state == "record":
            stats["frames"] += 1
            if n == 2:
                stats["two"] += 1
            elif n == 1:
                stats["one"] += 1
            else:
                stats["zero"] += 1
            if prev_count is not None and n != prev_count:
                stats["drops"] += 1
            prev_count = n
            for hs in res.handedness:
                if hs:
                    stats["scores"].append(hs[0].score)
            if n == 2:
                ordered = sorted(hands, key=lambda hd: hd[0].x)
                cur = np.array([[p.x, p.y] for hd in ordered for p in hd])
                if prev_pts is not None:
                    stats["jitter"].append(float(np.mean(np.abs(cur - prev_pts))))
                prev_pts = cur
            else:
                prev_pts = None

            remaining = RECORD_S - (now - t_state)
            msg = f"RECORDING {seal.upper()}  {remaining:.1f}s  hands:{n}"
            color = (0, 0, 255)
            if remaining <= 0:
                results[seal] = summarize(seal, stats)
                idx += 1
                state = "ready"
                t_state = now
        elif state == "countdown":
            remaining = COUNTDOWN_S - (now - t_state)
            msg = f"GET READY: {seal.upper()}  {remaining:.1f}s"
            color = (0, 200, 255)
            if remaining <= 0:
                state = "record"
                t_state = now
                stats = new_stats()
                prev_count = None
                prev_pts = None
        else:
            msg = f"SPACE = start {seal.upper()} ({idx + 1}/12)   s=skip  b=back  q=quit"
            color = (255, 255, 0)

        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("MediaPipe spike", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and state == "ready":
            state = "countdown"
            t_state = time.time()
        if key == ord("s") and state == "ready":
            idx += 1
        if key == ord("b") and state == "ready" and idx > 0:
            idx -= 1
            results.pop(SEALS[idx], None)

    cap.release()
    cv2.destroyAllWindows()

    if results:
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        rows = list(results.values())
        with open(OUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {OUT_PATH}\n")
        print("seal,fps,two%,one%,none%,changes,handed_score,jitter")
        for r in rows:
            print(f"{r['seal']},{r['fps']},{r['pct_two_hands']},{r['pct_one_hand']},"
                  f"{r['pct_no_hands']},{r['hand_count_changes']},"
                  f"{r['mean_handedness_score']},{r['mean_jitter']}")


if __name__ == "__main__":
    main()