from pathlib import Path

import cv2

SRC = Path("data/external/Pure Naruto Hand Sign Data/train")
DST = Path("data/parity_samples")
PER_CLASS = 8
MAX_SIDE = 960

DST.mkdir(parents=True, exist_ok=True)
for cls_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
    files = sorted(f for f in cls_dir.iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        continue
    step = max(1, len(files) // PER_CLASS)
    picks = files[::step][:PER_CLASS]       # spread out: neighbouring files are near-duplicates
    for i, f in enumerate(picks):
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        s = MAX_SIDE / max(h, w)
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(DST / f"{cls_dir.name}_{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("saved", len(list(DST.glob("*.jpg"))), "images")