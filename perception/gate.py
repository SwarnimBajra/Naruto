"""Pure hand-seal decision logic shared by live capture and offline replay."""

from collections import Counter, deque


ID_TO_NAME = {
    0: "Rat", 1: "Ox", 2: "Tiger", 3: "Hare", 4: "Dragon", 5: "Snake",
    6: "Horse", 7: "Ram", 8: "Monkey", 9: "Bird", 10: "Dog", 11: "Boar",
    12: "Gassho", 13: "Unknown", 14: "Mizunoe",
}
CLASS_NAMES = [ID_TO_NAME[i] for i in range(15)]
SEALS = CLASS_NAMES[:12]
SEAL_SET = set(SEALS)

DEFAULT_THRESHOLD = 0.7
FREE_THRESHOLDS = {"Snake": 0.6, "Monkey": 0.6}
EXPECT_THRESHOLDS = {"Hare": 0.5, "Ram": 0.6, "Snake": 0.6, "Monkey": 0.6}
THRESH_GRID = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
HARD = {"Hare", "Ram", "Snake"}
WINDOW = 12
NEED, HARD_NEED = 8, 8
ARM_BELOW = 0.35
ARM_FRAMES = 4


def need_for(seal):
    return HARD_NEED if seal in HARD else NEED


class StabilityGate:
    """Confirm a candidate after enough hits, with release hysteresis."""

    def __init__(self, window=WINDOW, need=NEED):
        self.hist = deque(maxlen=window)
        self.need = need
        self.release = max(2, need // 3)
        self.current = None

    def update(self, candidate):
        self.hist.append(candidate)
        counts = Counter(c for c in self.hist if c is not None)
        if self.current is not None:
            if counts.get(self.current, 0) >= self.release:
                return self.current, 1.0, False
            self.current = None
        best, n = counts.most_common(1)[0] if counts else (None, 0)
        if best and n >= self.need:
            self.current = best
            return best, 1.0, True
        hold = min(n / self.need, 1.0) if best else 0.0
        return None, hold, False


def best_per_class(scores, class_ids):
    """Return the highest post-NMS score for each model class."""
    best = {}
    for score, class_id in zip(scores, class_ids):
        name = ID_TO_NAME.get(int(class_id), f"id{int(class_id)}")
        value = float(score)
        if value > best.get(name, 0.0):
            best[name] = value
    return best


def update_armed(expect, expect_score, armed, low_run):
    """Advance the relax-first latch; caller resets it when target changes."""
    if expect is None:
        return True, 0
    if not armed:
        low_run = low_run + 1 if expect_score < ARM_BELOW else 0
        armed = low_run >= ARM_FRAMES
    return armed, low_run


def frame_decision(scores, expect, armed, threshold_overrides=None, top_hint=None,
                   free_threshold_overrides=None):
    """Return candidate, threshold, target score, top class, and top score."""
    threshold_overrides = threshold_overrides or {}
    free_threshold_overrides = free_threshold_overrides or {}
    detected = {name: float(score) for name, score in scores.items() if float(score) > 0}
    if detected:
        top_name = max(detected, key=detected.get)
        top_score = detected[top_name]
        if top_hint and top_hint[0] in detected and detected[top_hint[0]] == top_score:
            top_name = top_hint[0]
            top_score = detected[top_name]
    else:
        top_name, top_score = None, 0.0

    if expect:
        threshold = threshold_overrides.get(
            expect, EXPECT_THRESHOLDS.get(expect, DEFAULT_THRESHOLD)
        )
        expect_score = float(scores.get(expect, 0.0))
        candidate = expect if armed and expect_score >= threshold else None
    else:
        threshold = free_threshold_overrides.get(
            top_name,
            FREE_THRESHOLDS.get(top_name, DEFAULT_THRESHOLD),
        )
        expect_score = 0.0
        candidate = top_name if top_name in SEAL_SET and top_score >= threshold else None
    return candidate, float(threshold), expect_score, top_name, top_score
