import re
import csv
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

REPO = "third_party/NARUTO-HandSignDetection"
sys.path.insert(0, REPO)
from model.yolox.yolox_onnx import YoloxONNX  # noqa: E402

MODEL = f"{REPO}/model/yolox/yolox_nano.onnx"
LABELS = f"{REPO}/setting/labels.csv"
OUT_PATH = "data/spike_yolox.csv"
SEALS = ["Rat", "Ox", "Tiger", "Hare", "Dragon", "Snake",
         "Horse", "Ram", "Monkey", "Bird", "Dog", "Boar"]
COUNTDOWN_S = 3.0
RECORD_S = 5.0
DEMO_THRESHOLD = 0.7  # the threshold Kazuhito's demos use by default
LABEL_OFFSET = 1

ALIASES = {
    "rat": "Rat", "ne": "Rat", "ox": "Ox", "ushi": "Ox",
    "tiger": "Tiger", "tora": "Tiger", "hare": "Hare", "u": "Hare",
    "dragon": "Dragon", "tatsu": "Dragon", "snake": "Snake", "mi": "Snake",
    "horse": "Horse", "uma": "Horse", "ram": "Ram", "hitsuji": "Ram",
    "monkey": "Monkey", "saru": "Monkey", "bird": "Bird", "tori": "Bird",
    "dog": "Dog", "inu": "Dog", "boar": "Boar", "i": "Boar",
}


def load_labels():
    names = []
    with open(LABELS, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                names.append(row[0].strip())
    return names


def canon(name):
    m = re.search(r"\((.*?)\)", name)
    return m.group(1) if m else name


def new_stats():
    return {"frames": 0, "detected": 0, "correct_any": 0, "correct_th": 0,
            "scores_correct": [], "wrong": Counter(), "inf_ms": [],
            "t_start": time.time()}


def summarize(seal, s, flip):
    n = max(s["frames"], 1)
    dur = max(time.time() - s["t_start"], 1e-6)
    wrong_total = sum(s["wrong"].values())
    top_conf = ""
    if wrong_total:
        name, cnt = s["wrong"].most_common(1)[0]
        top_conf = f"{name}:{round(100 * cnt / n)}%"
    return {
        "seal": seal,
        "flip": flip,
        "frames": s["frames"],
        "fps": round(s["frames"] / dur, 1),
        "pct_detected": round(100 * s["detected"] / n, 1),
        "pct_correct_any_score": round(100 * s["correct_any"] / n, 1),
        "pct_correct_at_0.7": round(100 * s["correct_th"] / n, 1),
        "mean_score_when_correct": round(float(np.mean(s["scores_correct"])), 3)
        if s["scores_correct"] else 0.0,
        "top_confusion": top_conf,
        "mean_inference_ms": round(float(np.mean(s["inf_ms"])), 1) if s["inf_ms"] else 0.0,
    }


def main():
    labels = load_labels()
    print("Labels loaded from labels.csv:")
    for i, name in enumerate(labels):
        print(f"  {i}: {name} -> {canon(name)}")
    print()

    det = YoloxONNX(
        model_path=MODEL,
        input_shape=(416, 416),
        class_score_th=0.3,
        nms_th=0.45,
        nms_score_th=0.1,
        providers=["CPUExecutionProvider"],
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    flip = True
    results = {}
    idx = 0
    state = "ready"
    t_state = time.time()
    stats = None

    while idx < len(SEALS):
        ok, frame = cap.read()
        if not ok:
            break
        if flip:
            frame = cv2.flip(frame, 1)

        t0 = time.time()
        bboxes, scores, class_ids = det.inference(frame)
        inf_ms = (time.time() - t0) * 1000

        top = None
        if len(scores) > 0:
            k = int(np.argmax(scores))
            cid = int(class_ids[k])
            li = cid + LABEL_OFFSET
            name = canon(labels[li]) if 0 <= li < len(labels) else f"id{cid}"
            top = (name, float(scores[k]), bboxes[k])
            x1, y1, x2, y2 = [int(v) for v in bboxes[k]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {scores[k]:.2f}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        now = time.time()
        seal = SEALS[idx]

        if state == "record":
            stats["frames"] += 1
            stats["inf_ms"].append(inf_ms)
            if top is not None:
                stats["detected"] += 1
                if top[0] == seal:
                    stats["correct_any"] += 1
                    stats["scores_correct"].append(top[1])
                    if top[1] >= DEMO_THRESHOLD:
                        stats["correct_th"] += 1
                else:
                    stats["wrong"][top[0]] += 1
            remaining = RECORD_S - (now - t_state)
            msg = f"RECORDING {seal.upper()}  {remaining:.1f}s"
            color = (0, 0, 255)
            if remaining <= 0:
                results[seal] = summarize(seal, stats, flip)
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
        else:
            msg = (f"SPACE=start {seal.upper()} ({idx + 1}/12)  s=skip b=back "
                   f"f=flip({'on' if flip else 'off'}) q=quit")
            color = (255, 255, 0)

        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.imshow("YOLOX spike", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if state == "ready":
            if key == ord(" "):
                state = "countdown"
                t_state = time.time()
            elif key == ord("s"):
                idx += 1
            elif key == ord("b") and idx > 0:
                idx -= 1
                results.pop(SEALS[idx], None)
            elif key == ord("f"):
                flip = not flip

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
        print("seal,flip,fps,detected%,correct_any%,correct@0.7%,mean_score,top_confusion,inf_ms")
        for r in rows:
            print(f"{r['seal']},{r['flip']},{r['fps']},{r['pct_detected']},"
                  f"{r['pct_correct_any_score']},{r['pct_correct_at_0.7']},"
                  f"{r['mean_score_when_correct']},{r['top_confusion']},"
                  f"{r['mean_inference_ms']}")


if __name__ == "__main__":
    main()