import csv
import os
import random
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

REPO = "third_party/NARUTO-HandSignDetection"
sys.path.insert(0, REPO)
from model.yolox.yolox_onnx import YoloxONNX  # type: ignore  # noqa: E402

DATA = "data/external/Pure Naruto Hand Sign Data"
OUT_DIR = "data/audit"
YOLOX_MODEL = f"{REPO}/model/yolox/yolox_nano.onnx"
HAND_MODEL = "models/hand_landmarker.task"
MAX_SIDE = 960          # shrink big photos to webcam-like size
DUP_BITS = 5            # dHash hamming distance counted as near-duplicate
MONTAGE_N = 8

NAME_TO_ID = {"rat": 0, "ox": 1, "tiger": 2, "hare": 3, "dragon": 4, "snake": 5,
              "horse": 6, "ram": 7, "monkey": 8, "bird": 9, "dog": 10, "boar": 11}
ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}
ID_TO_NAME.update({12: "gassho", 13: "unknown", 14: "mizunoe"})
IMG_EXT = (".jpg", ".jpeg", ".png")


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def dhash(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def thumb(img, w=200, h=150):
    s = min(w / img.shape[1], h / img.shape[0])
    t = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    canvas = np.full((h, w, 3), 90, dtype=np.uint8)
    y0, x0 = (h - t.shape[0]) // 2, (w - t.shape[1]) // 2
    canvas[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
    return canvas


def collect():
    items = []
    for split in ("train", "test"):
        sdir = os.path.join(DATA, split)
        if not os.path.isdir(sdir):
            continue
        for cls in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, cls)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                if fn.lower().endswith(IMG_EXT):
                    items.append((split, cls.lower(), os.path.join(cdir, fn)))
    return items


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    items = collect()
    print(f"Found {len(items)} images\n")

    rng = random.Random(0)
    by_class_train = defaultdict(list)
    for split, cls, p in items:
        if split == "train":
            by_class_train[cls].append(p)
    sample_paths = {}
    for cls, paths in by_class_train.items():
        for p in rng.sample(paths, min(MONTAGE_N, len(paths))):
            sample_paths[p] = cls
    thumbs = defaultdict(list)

    det = YoloxONNX(model_path=YOLOX_MODEL, input_shape=(416, 416),
                    class_score_th=0.3, nms_th=0.45, nms_score_th=0.1,
                    providers=["CPUExecutionProvider"])
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.IMAGE, num_hands=2))

    rows = []
    hashes = defaultdict(list)       # (split, cls) -> list of hash arrays
    sizes = defaultdict(list)
    t0 = time.time()
    for i, (split, cls, path) in enumerate(items):
        img = load_image(path)
        if img is None:
            print("Could not read:", path)
            continue
        sizes[cls].append(img.shape[:2])
        hashes[(split, cls)].append(dhash(img))
        if path in sample_paths:
            thumbs[cls].append(thumb(img))

        bboxes, scores, class_ids = det.inference(img)
        if len(scores) > 0:
            k = int(np.argmax(scores))
            pred = ID_TO_NAME.get(int(class_ids[k]), f"id{int(class_ids[k])}")
            score = float(scores[k])
        else:
            pred, score = "none", 0.0

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hand.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        n_hands = len(res.hand_landmarks)

        rows.append({"split": split, "cls": cls, "path": path, "yolox_pred": pred,
                     "yolox_score": round(score, 3), "mp_hands": n_hands})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(items)}  ({time.time() - t0:.0f}s)")

    with open(os.path.join(OUT_DIR, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    for cls, ts in thumbs.items():
        ts = ts + [np.full((150, 200, 3), 40, np.uint8)] * (MONTAGE_N - len(ts))
        grid = np.vstack([np.hstack(ts[:4]), np.hstack(ts[4:8])])
        cv2.imwrite(os.path.join(OUT_DIR, f"montage_{cls}.png"), grid)

    # per-class summary
    classes = sorted({r["cls"] for r in rows})
    summary = []
    for cls in classes:
        cr = [r for r in rows if r["cls"] == cls]
        n = len(cr)
        n_train = sum(r["split"] == "train" for r in cr)
        n_test = n - n_train
        if cls in NAME_TO_ID:
            ok = sum(r["yolox_pred"] == cls for r in cr)
            ok7 = sum(r["yolox_pred"] == cls and r["yolox_score"] >= 0.7 for r in cr)
            wrong = Counter(r["yolox_pred"] for r in cr if r["yolox_pred"] != cls)
            top_wrong = ""
            if wrong:
                nm, c = wrong.most_common(1)[0]
                top_wrong = f"{nm}:{round(100 * c / n)}%"
        else:   # e.g. 'zero' = expected no seal
            ok = sum(r["yolox_pred"] == "none" or r["yolox_score"] < 0.5 for r in cr)
            ok7 = ok
            top_wrong = ""
        two = sum(r["mp_hands"] == 2 for r in cr)
        one = sum(r["mp_hands"] == 1 for r in cr)

        tr = np.array(hashes.get(("train", cls), []))
        te = np.array(hashes.get(("test", cls), []))
        dup_pct, leaked = "", ""
        if len(tr) > 1:
            d = (tr[:, None, :] != tr[None, :, :]).sum(2)
            np.fill_diagonal(d, 99)
            dup_pct = round(100 * float(np.mean((d <= DUP_BITS).any(1))), 1)
        if len(tr) and len(te):
            d = (te[:, None, :] != tr[None, :, :]).sum(2)
            leaked = int((d <= DUP_BITS).any(1).sum())
        hh = np.array(sizes[cls])
        summary.append({
            "class": cls, "train": n_train, "test": n_test,
            "yolox_ok%": round(100 * ok / n, 1), "yolox_ok@0.7%": round(100 * ok7 / n, 1),
            "yolox_top_wrong": top_wrong,
            "mp_two%": round(100 * two / n, 1), "mp_one%": round(100 * one / n, 1),
            "near_dup_train%": dup_pct, "test_leaked": leaked,
            "median_hw": f"{int(np.median(hh[:, 0]))}x{int(np.median(hh[:, 1]))}",
        })

    with open(os.path.join(OUT_DIR, "audit_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)

    print("\nclass,train,test,yolox_ok%,yolox_ok@0.7%,top_wrong,mp_two%,mp_one%,"
          "near_dup_train%,test_leaked,median_hw")
    for s in summary:
        print(",".join(str(s[k]) for k in s))
    print(f"\nSaved to {OUT_DIR}/ (summary, per_image.csv, one montage per class)")


if __name__ == "__main__":
    main()