"""Replay perception recordings with the live gate implementation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

try:
    from .gate import (ARM_BELOW, ARM_FRAMES, CLASS_NAMES, DEFAULT_THRESHOLD,
                       EXPECT_THRESHOLDS, FREE_THRESHOLDS, HARD_NEED, NEED, SEALS,
                       StabilityGate, frame_decision, need_for, update_armed)
except ImportError:  # python perception/replay.py
    from gate import (ARM_BELOW, ARM_FRAMES, CLASS_NAMES, DEFAULT_THRESHOLD,
                      EXPECT_THRESHOLDS, FREE_THRESHOLDS, HARD_NEED, NEED, SEALS,
                      StabilityGate, frame_decision, need_for, update_armed)


def _logged_settings(header):
    gate = header.get("gate", {})
    return {
        "window": int(gate.get("WINDOW", 12)),
        "need": int(gate.get("NEED", 8)),
        "hard_need": int(gate.get("HARD_NEED", gate.get("NEED", 8))),
        "thresholds": dict(gate.get("thresholds", {
            s: EXPECT_THRESHOLDS.get(s, DEFAULT_THRESHOLD) for s in SEALS})),
        "free_thresholds": dict(gate.get("free_thresholds", {
            s: FREE_THRESHOLDS.get(s, DEFAULT_THRESHOLD) for s in SEALS})),
        "arm_below": float(gate.get("ARM_BELOW", ARM_BELOW)),
        "arm_frames": int(gate.get("ARM_FRAMES", ARM_FRAMES)),
    }


def _override_need(settings, need):
    if need is not None:
        settings["need"] = need
        settings["hard_need"] = need


def replay_records(records, overrides=None):
    header = next((r for r in records if r.get("type") == "header"), {})
    settings = _logged_settings(header)
    overrides = overrides or {}
    if overrides.get("window") is not None:
        settings["window"] = overrides["window"]
    _override_need(settings, overrides.get("need"))
    thresholds = dict(settings["thresholds"])
    thresholds.update(overrides.get("thresholds", {}))
    target_overrides = thresholds
    frames = [r for r in records if r.get("type") == "frame"]
    events = [r for r in records if r.get("type") == "expect"]
    expect = None
    expect_since = 0.0
    last_event_idx = 0
    gate = StabilityGate(settings["window"], settings["need"])
    armed, low_run = True, 0
    replayed, mismatch_rows = [], []
    for frame in frames:
        t = float(frame.get("t", 0))
        # Event records precede the corresponding frame in the source log.
        while last_event_idx < len(events) and float(events[last_event_idx].get("t", 0)) <= t:
            next_expect = events[last_event_idx].get("seal")
            if next_expect != expect:
                expect = next_expect
                expect_since = float(events[last_event_idx].get("t", 0))
                gate = StabilityGate(settings["window"],
                                     settings["hard_need"] if expect in {"Hare", "Ram", "Snake"}
                                     else settings["need"])
                armed, low_run = (expect is None), 0
            last_event_idx += 1
        # Logs may have target values on frames without expect events (older files).
        if not events and frame.get("expect") != expect:
            expect = frame.get("expect")
            expect_since = t
            gate = StabilityGate(settings["window"], settings["hard_need"] if expect in {"Hare", "Ram", "Snake"}
                                 else settings["need"])
            armed, low_run = (expect is None), 0

        scores = frame.get("scores", {})
        # Score is independent of the latch, so compute it before advancing the
        # relax-first rule, then make the candidate with the updated latch.
        _, _, exp_score, _, _ = frame_decision(
            scores, expect, armed, target_overrides, frame.get("top"), settings["free_thresholds"])
        armed, low_run = update_armed(expect, exp_score, armed, low_run)
        candidate, _, _, _, _ = frame_decision(
            scores, expect, armed, target_overrides, frame.get("top"), settings["free_thresholds"])
        stable, hold, confirm = gate.update(candidate)
        replayed.append({"frame": frame, "expect": expect, "stable": stable,
                         "hold": hold, "confirm": confirm, "expect_since": expect_since})
        if bool(frame.get("confirm")) != confirm:
            mismatch_rows.append((frame.get("i"), bool(frame.get("confirm")), confirm))

    return {"header": header, "frames": replayed, "mismatches": mismatch_rows,
            "settings": settings}


def _attempts(result):
    frames = result["frames"]
    attempts = []
    current = None
    for item in frames:
        frame, target, t = item["frame"], item["expect"], float(item["frame"].get("t", 0))
        if target != (current["seal"] if current else None):
            if current:
                current["end"] = item["expect_since"]
                attempts.append(current)
            current = {"seal": target, "start": item["expect_since"],
                       "end": t, "confirmed": None} if target else None
        if current:
            current["end"] = t
            if item["confirm"] and item["stable"] == current["seal"] and current["confirmed"] is None:
                current["confirmed"] = t
    if current:
        current["end"] = float(frames[-1]["frame"].get("t", current["start"])) if frames else current["start"]
        attempts.append(current)
    return attempts


def _km_quantile(attempts, quantile):
    observations = sorted((max(0.0, (a["confirmed"] if a["confirmed"] is not None else a["end"]) - a["start"]),
                           a["confirmed"] is not None) for a in attempts)
    survival, at_risk, i = 1.0, len(observations), 0
    while i < len(observations):
        t = observations[i][0]
        events = 0
        while i < len(observations) and observations[i][0] == t:
            events += int(observations[i][1])
            i += 1
        if at_risk:
            survival *= (1.0 - events / at_risk)
            if survival <= 1.0 - quantile:
                return t
        # Events and censorings at this time both leave the risk set.
        at_risk = len(observations) - i
    return None


def analyze(result):
    frames, attempts = result["frames"], _attempts(result)
    if frames:
        times = [float(x["frame"].get("t", 0)) for x in frames]
        duration = max(times) - min(times)
    else:
        duration = 0.0
    duration_minutes = duration / 60.0
    false_wrong = sum(1 for x in frames if x["confirm"] and x["expect"] and x["stable"] != x["expect"])
    false_outside = sum(1 for x in frames if x["confirm"] and not x["expect"])
    per_seal = {}
    for seal in SEALS:
        group = [a for a in attempts if a["seal"] == seal]
        successes = [a for a in group if a["confirmed"] is not None]
        per_seal[seal] = {"attempts": len(group), "successes": len(successes),
                          "rate": len(successes) / len(group) if group else None,
                          "median": _km_quantile(group, 0.5), "p90": _km_quantile(group, 0.9)}
    return {"attempts": attempts, "per_seal": per_seal,
            "false_non_target": false_wrong, "false_outside": false_outside,
            "duration_minutes": duration_minutes,
            "false_per_minute": ((false_wrong + false_outside) / duration_minutes
                                 if duration_minutes > 0 else None)}


def _fmt(value):
    return f"{value:.2f}s" if value is not None else "not reached"


def print_report(report):
    print(f"Attempts: {len(report['attempts'])}")
    print(f"False confirms: non-target={report['false_non_target']}, outside attempts={report['false_outside']}, "
          f"combined/min={report['false_per_minute'] if report['false_per_minute'] is not None else 'n/a'}")
    print("Seal       attempts  success  rate   median  p90")
    for seal, row in report["per_seal"].items():
        rate = f"{row['rate']:.0%}" if row["rate"] is not None else "n/a"
        print(f"{seal:<10} {row['attempts']:>8} {row['successes']:>8} {rate:>6} "
              f"{_fmt(row['median']):>8} {_fmt(row['p90']):>8}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--need", type=int)
    ap.add_argument("--window", type=int)
    ap.add_argument("--threshold", action="append", default=[], metavar="SEAL=VALUE")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    records = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines() if line.strip()]
    threshold_overrides = {}
    for spec in args.threshold:
        seal, value = spec.split("=", 1)
        if seal not in SEALS:
            ap.error(f"unknown seal: {seal}")
        threshold_overrides[seal] = float(value)
    if args.check and (args.need is not None or args.window is not None or threshold_overrides):
        ap.error("--check uses logged settings and cannot be combined with overrides")
    result = replay_records(records, {"need": args.need, "window": args.window,
                                      "thresholds": threshold_overrides})
    print_report(analyze(result))
    if args.check:
        for idx, expected, actual in result["mismatches"]:
            print(f"Mismatch frame {idx}: logged confirm={expected}, replay confirm={actual}")
        print(f"--check mismatches: {len(result['mismatches'])}")
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["seal", "attempts", "successes", "success_rate", "median_seconds", "p90_seconds"])
            for seal, row in analyze(result)["per_seal"].items():
                writer.writerow([seal, row["attempts"], row["successes"], row["rate"], row["median"], row["p90"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

