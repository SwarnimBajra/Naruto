import json
import shutil
import sys
from pathlib import Path

import cv2

REPO = "third_party/NARUTO-HandSignDetection"
sys.path.insert(0, REPO)
from model.yolox.yolox_onnx import YoloxONNX  # type: ignore  # noqa: E402

ID_TO_NAME = {0: "Rat", 1: "Ox", 2: "Tiger", 3: "Hare", 4: "Dragon", 5: "Snake",
              6: "Horse", 7: "Ram", 8: "Monkey", 9: "Bird", 10: "Dog", 11: "Boar",
              12: "Gassho", 13: "Unknown", 14: "Mizunoe"}
SRC = Path("data/parity_samples")
DST = Path("web/spike")
MODEL = Path(f"{REPO}/model/yolox/yolox_nano.onnx")

det = YoloxONNX(model_path=str(MODEL), input_shape=(416, 416), class_score_th=0.3,
                nms_th=0.45, nms_score_th=0.1, providers=["CPUExecutionProvider"])

(DST / "parity").mkdir(parents=True, exist_ok=True)
expected = {}
for p in sorted(SRC.glob("*.jpg")):
    img = cv2.imread(str(p))
    bboxes, scores, class_ids = det.inference(img)
    best = {}
    for s, c in zip(scores, class_ids):
        name = ID_TO_NAME.get(int(c), f"id{int(c)}")
        if float(s) > best.get(name, 0.0):
            best[name] = float(s)
    top = max(best, key=best.get) if best else None
    expected[p.name] = {"top": top, "top_score": round(best[top], 4) if top else 0.0}
    shutil.copy(p, DST / "parity" / p.name)

(DST / "parity_expected.json").write_text(json.dumps(expected, indent=1))
shutil.copy(MODEL, DST / "yolox_nano.onnx")
print(f"wrote {len(expected)} expected results and copied the model to {DST}")