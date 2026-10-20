"""Asynchronous, buffered session recording for the perception service."""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path


class SessionRecorder:
    """Queue JSONL records and optional raw video for background writing."""

    def __init__(self, jsonl_path, header, video_path=None, video_fps=30.0):
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.video_path = Path(video_path) if video_path else None
        self.video_fps = video_fps if video_fps > 0 else 30.0
        # Keep the capture loop non-blocking; if disk/video encoding falls behind,
        # queued records remain intact and are drained at shutdown.
        self.queue = queue.Queue()
        self.dropped = 0
        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()
        self.write(header)

    def write(self, record, raw_frame=None):
        self.queue.put_nowait((record, raw_frame))

    def _writer(self):
        import cv2

        writer = None
        last_flush = time.monotonic()
        try:
            with self.jsonl_path.open("w", encoding="utf-8", buffering=1024 * 1024) as stream:
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    record, raw_frame = item
                    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                    if raw_frame is not None and self.video_path:
                        if writer is None:
                            height, width = raw_frame.shape[:2]
                            writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                     self.video_fps, (width, height))
                        if writer.isOpened():
                            writer.write(raw_frame)
                    if time.monotonic() - last_flush >= 1.0:
                        stream.flush()
                        last_flush = time.monotonic()
                stream.flush()
        finally:
            if writer is not None:
                writer.release()

    def close(self):
        self.queue.put(None)
        self.thread.join()

