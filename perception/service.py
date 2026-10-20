import base64
from pathlib import Path
import os
import argparse
import asyncio
import json
import hashlib
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone


import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from websockets.asyncio.server import serve, broadcast
from websockets.exceptions import ConnectionClosed

try:
    from .gate import (
        ARM_BELOW, ARM_FRAMES, CLASS_NAMES, DEFAULT_THRESHOLD, EXPECT_THRESHOLDS,
        FREE_THRESHOLDS, HARD_NEED, ID_TO_NAME, NEED, SEALS, SEAL_SET, THRESH_GRID,
        WINDOW, StabilityGate, best_per_class, frame_decision, need_for, update_armed,
    )
    from .recording import SessionRecorder
except ImportError:  # direct invocation: python perception/service.py
    from gate import (  # type: ignore[no-redef]
        ARM_BELOW, ARM_FRAMES, CLASS_NAMES, DEFAULT_THRESHOLD, EXPECT_THRESHOLDS,
        FREE_THRESHOLDS, HARD_NEED, ID_TO_NAME, NEED, SEALS, SEAL_SET, THRESH_GRID,
        WINDOW, StabilityGate, best_per_class, frame_decision, need_for, update_armed,
    )
    from recording import SessionRecorder  # type: ignore[no-redef]

REPO = "third_party/NARUTO-HandSignDetection"
sys.path.insert(0, REPO)
from model.yolox.yolox_onnx import YoloxONNX  # type: ignore  # noqa: E402

YOLOX_MODEL = f"{REPO}/model/yolox/yolox_nano.onnx"
HAND_MODEL = "models/hand_landmarker.task"

HAND_EVERY = 3            # run MediaPipe every 3rd frame
IDLE_COUNTDOWN_S = 5.0
REF_DIR = Path("shared/reference")   # relative to the repo root
REF_MIN_SCORE = 0.7                  # only save references that scored this well
REF_MAX_SIDE = 480
FRAME_EVERY = 2                      # send every 2nd camera frame to Unity
PREVIEW_W = 320

STATE = {"expect": None}
CLIENTS = set()


async def handler(ws):
    CLIENTS.add(ws)
    print(f"[ws] client connected ({len(CLIENTS)})")
    try:
        async for raw in ws:
            try:
                data = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(data, dict) and "expect" in data:
                e = data["expect"]
                STATE["expect"] = e if e in SEAL_SET else None
                print(f"[ws] expect -> {STATE['expect']}")
    except ConnectionClosed:
        pass
    finally:
        CLIENTS.discard(ws)
        print(f"[ws] client disconnected ({len(CLIENTS)})")


def run_server(port, ready, holder):
    async def server_main():
        holder["loop"] = asyncio.get_running_loop()
        async with serve(handler, "127.0.0.1", port):
            ready.set()
            await asyncio.Future()
    asyncio.run(server_main())


def box_for(name, scores, class_ids, bboxes):
    bb, bs = None, 0.0
    for s, c, b in zip(scores, class_ids, bboxes):
        if ID_TO_NAME.get(int(c)) == name and float(s) > bs:
            bb, bs = b, float(s)
    return bb


def load_ref_best():
    out = {}
    for s in SEALS:
        try:
            out[s] = float((REF_DIR / f"{s}.txt").read_text().strip())
        except (OSError, ValueError):
            pass
    return out

def save_reference(frame, seal, score, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    mx, my = (x2 - x1) * 0.25, (y2 - y1) * 0.25
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return
    crop = frame[y1:y2, x1:x2]
    s = REF_MAX_SIDE / max(crop.shape[:2])
    if s < 1:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(REF_DIR / f"{seal}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (REF_DIR / f"{seal}.txt").write_text(f"{score:.3f}")
    print(f"[ref] saved {seal} reference (score {score:.2f})")

def print_idle_report(idle, seconds):
    print(f"\nIdle test: {seconds:.0f}s with no seals made. Numbers are FALSE confirmations.")
    print(f"Free mode (any seal, default thresholds): {idle['free_count']}\n")
    print("Target mode: false confirmations per seal at each threshold")
    print("seal      " + "".join(f"{t:>6.2f}" for t in THRESH_GRID))
    for s in SEALS:
        row = "".join(f"{idle['counts'][(s, t)]:>6d}" for t in THRESH_GRID)
        print(f"{s:<10}{row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--show", action="store_true",
                    help="show the debug preview window (always on for --idle-test)")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--idle-test", type=float, default=0,
                    help="seconds of idle false-positive test, then print a report and exit")
    ap.add_argument("--record", metavar="DIR", help="write a session JSONL recording")
    ap.add_argument("--person", default="", help="pseudonymous participant ID")
    ap.add_argument("--note", default="", help="free-text session note")
    ap.add_argument("--record-video", action="store_true", help="also record raw camera frames to MP4")
    args = ap.parse_args()
    if args.record_video and not args.record:
        ap.error("--record-video requires --record DIR")
    show = args.show or args.idle_test > 0
    
    ready, holder = threading.Event(), {}
    threading.Thread(target=run_server, args=(args.port, ready, holder),
                     daemon=True).start()
    if not ready.wait(5):
        print("WebSocket server did not start (is the port already in use?)")
        return
    loop = holder["loop"]
    print(f"[ws] listening on ws://127.0.0.1:{args.port}")

    det = YoloxONNX(model_path=YOLOX_MODEL, input_shape=(416, 416),
                    class_score_th=0.3, nms_th=0.45, nms_score_th=0.1,
                    providers=["CPUExecutionProvider"])
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO, num_hands=2))

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    camera_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    gate = StabilityGate(WINDOW, NEED)
    last_expect = None
    ref_best = load_ref_best()
    armed, low_run = True, 0
    start = time.time()
    prev, last_ts, fps = start, -1, 0.0
    n_hands, frame_i, hand_ms = 0, 0, 0.0
    recorder = None
    session_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record_start = time.monotonic()
    recorded_frames = 0

    idle = None
    if args.idle_test > 0:
        t0 = time.time() + IDLE_COUNTDOWN_S
        idle = {"t_start": t0, "t_end": t0 + args.idle_test,
                "gates": {(s, t): StabilityGate(WINDOW, need_for(s))
                          for s in SEALS for t in THRESH_GRID},
                "counts": Counter(), "free_gate": StabilityGate(WINDOW, NEED),
                "free_count": 0}

    try:
        while True:
            t_c = time.time()
            ok, frame = cap.read()
            cam_ms = (time.time() - t_c) * 1000
            if not ok:
                print("Camera read failed")
                break
            raw_frame = frame
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            t_y = time.time()
            bboxes, scores, class_ids = det.inference(frame)
            yolox_ms = (time.time() - t_y) * 1000

            best = best_per_class(scores, class_ids)
            top_name = max(best, key=best.get) if best else None
            top_score = best[top_name] if top_name else 0.0
            box = bboxes[int(np.argmax(scores))] if len(scores) > 0 else None

            expect = STATE["expect"]
            if expect != last_expect:
                gate = StabilityGate(WINDOW, need_for(expect))
                last_expect = expect
                armed, low_run = (expect is None), 0
                if recorder:
                    recorder.write({"type": "expect", "t": time.monotonic() - record_start,
                                    "seal": expect})
            _, _, exp_score, _, _ = frame_decision(best, expect, armed)
            armed, low_run = update_armed(expect, exp_score, armed, low_run)
            candidate, thr, exp_score, top_name, top_score = frame_decision(best, expect, armed)
            stable, hold, new_confirm = gate.update(candidate)
            if (expect and stable == expect and exp_score >= REF_MIN_SCORE
                    and exp_score > ref_best.get(expect, 0.0)):
                rb = box_for(expect, scores, class_ids, bboxes)
                if rb is not None:
                    save_reference(frame, expect, exp_score, rb)
                    ref_best[expect] = exp_score

            if frame_i % HAND_EVERY == 0:
                t_h = time.time()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = int((time.time() - start) * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                res = hand.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                n_hands = len(res.hand_landmarks)
                hand_ms = (time.time() - t_h) * 1000
            if args.record and recorder is None:
                model_sha = hashlib.sha256()
                with open(YOLOX_MODEL, "rb") as model_file:
                    for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                        model_sha.update(chunk)
                try:
                    service_commit = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                        text=True, timeout=2).strip()
                except Exception:
                    service_commit = None
                stamp = datetime.now(timezone.utc).isoformat()
                rec_dir = Path(args.record)
                stem = f"session_{session_stamp}"
                header = {"type": "header", "service_git_commit": service_commit,
                          "model_path": YOLOX_MODEL, "model_sha256": model_sha.hexdigest(),
                          "camera_size": {"width": int(raw_frame.shape[1]), "height": int(raw_frame.shape[0])},
                          "mirror": not args.no_mirror,
                          "gate": {"WINDOW": WINDOW, "NEED": NEED, "HARD_NEED": HARD_NEED,
                                   "thresholds": {s: EXPECT_THRESHOLDS.get(s, DEFAULT_THRESHOLD) for s in SEALS},
                                   "free_thresholds": {s: FREE_THRESHOLDS.get(s, DEFAULT_THRESHOLD) for s in SEALS},
                                   "ARM_BELOW": ARM_BELOW, "ARM_FRAMES": ARM_FRAMES},
                          "person": args.person, "note": args.note, "start_utc": stamp}
                recorder = SessionRecorder(rec_dir / f"{stem}.jsonl", header,
                                           rec_dir / f"{stem}.mp4" if args.record_video else None,
                                           camera_fps)
                if last_expect is not None:
                    recorder.write({"type": "expect", "t": time.monotonic() - record_start,
                                    "seal": last_expect})
            frame_i += 1

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now

            if recorder:
                score_map = {name: float(best.get(name, 0.0)) for name in CLASS_NAMES}
                top_box = None
                if top_name:
                    selected = box_for(top_name, scores, class_ids, bboxes)
                    if selected is not None:
                        top_box = [float(v) for v in selected]
                recorder.write({"type": "frame", "i": recorded_frames,
                                "t": time.monotonic() - record_start, "expect": expect,
                                "scores": score_map, "top": [top_name, float(top_score)],
                                "box": top_box, "hands": n_hands, "armed": bool(armed),
                                "hold": float(hold), "stable": stable, "confirm": bool(new_confirm),
                                "ms": {"cam": cam_ms, "yolox": yolox_ms}},
                               raw_frame.copy() if args.record_video else None)
                recorded_frames += 1

            phase_msg = ""
            if idle:
                if now < idle["t_start"]:
                    phase_msg = f"IDLE TEST in {idle['t_start'] - now:.0f}s - relax, make NO seals"
                elif now < idle["t_end"]:
                    el = now - idle["t_start"]
                    third = args.idle_test / 3
                    phase = ("hands DOWN on the desk" if el < third else
                             "fists TOGETHER at chest" if el < 2 * third else
                             "slow random gestures, NO seals")
                    phase_msg = f"IDLE {idle['t_end'] - now:.0f}s left: {phase}"
                    for (s, t), g in idle["gates"].items():
                        if g.update(s if best.get(s, 0.0) >= t else None)[2]:
                            idle["counts"][(s, t)] += 1
                            if t == 0.7 or t == EXPECT_THRESHOLDS.get(s):
                                print(f"[idle] {now - idle['t_start']:5.1f}s  false {s} "
                                      f"at threshold {t} (score {best.get(s, 0.0):.2f})")
                                os.makedirs("data/idle_fp", exist_ok=True)
                                snap = frame.copy()
                                cv2.putText(snap, f"{s} {best.get(s, 0.0):.2f} thr {t}", (10, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                cv2.imwrite(f"data/idle_fp/{now - idle['t_start']:05.1f}s_{s}_{t}.jpg", snap)
                    
                    ft = FREE_THRESHOLDS.get(top_name, DEFAULT_THRESHOLD)
                    fc = top_name if (top_name in SEAL_SET and top_score >= ft) else None
                    if idle["free_gate"].update(fc)[2]:
                        idle["free_count"] += 1
                else:
                    print_idle_report(idle, args.idle_test)
                    break

            if CLIENTS:
                msg = json.dumps({
                    "t": round(now, 3), "stable_seal": stable, "hold": round(hold, 2),
                    "new_confirm": new_confirm, "raw": top_name,
                    "raw_score": round(top_score, 3), "expect": expect,
                    "expect_score": round(exp_score, 3), "expect_thr": thr, "armed": armed,
                    "hands": n_hands, "fps": round(fps, 1),
                })
                loop.call_soon_threadsafe(lambda m=msg: broadcast(CLIENTS, m))
                if frame_i % FRAME_EVERY == 0:
                    small = cv2.resize(frame, (PREVIEW_W, int(frame.shape[0] * PREVIEW_W / frame.shape[1])),
                                       interpolation=cv2.INTER_AREA)
                    okj, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    if okj:
                        fmsg = json.dumps({"type": "frame",
                                           "jpg": base64.b64encode(buf.tobytes()).decode("ascii")},
                                          separators=(",", ":"))
                        loop.call_soon_threadsafe(lambda m=fmsg: broadcast(CLIENTS, m))

            if show:
                if box is not None:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{top_name} {top_score:.2f}", (x1, max(y1 - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                mode = f"target {expect} {exp_score:.2f}/{thr:.2f}" if expect else "free"
                line1 = (f"{mode} | stable {stable or '-'} hold {hold:.0%} | hands {n_hands} | "
                         f"fps {fps:.0f} cam {cam_ms:.0f}ms yolox {yolox_ms:.0f}ms hand {hand_ms:.0f}ms")
                cv2.putText(frame, line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 0), 1)
                if phase_msg:
                    cv2.putText(frame, phase_msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 200, 255), 2)
                cv2.imshow("Hand Seal Dojo - perception (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if recorder:
            recorder.close()
            elapsed = max(time.monotonic() - record_start, 1e-6)
            print(f"[record] {recorded_frames / elapsed:.2f} FPS recording=on "
                  f"frames={recorded_frames} dropped={recorder.dropped} file={recorder.jsonl_path}")
        else:
            elapsed = max(time.monotonic() - record_start, 1e-6)
            print(f"[record] {frame_i / elapsed:.2f} FPS recording=off frames={frame_i}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
