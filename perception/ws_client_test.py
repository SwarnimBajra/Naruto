import argparse
import asyncio
import json
import sys
import time

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

PEAK = {"score": 0.0}


async def main(expect, port):
    t_start = time.time()
    async with connect(f"ws://127.0.0.1:{port}") as ws:
        await ws.send(json.dumps({"expect": expect}))
        print(f"Connected. Target: {expect or 'any seal (free mode)'}. Ctrl+C to stop.\n")
        last = 0.0
        try:
            async for raw in ws:
                d = json.loads(raw)
                PEAK["score"] = max(PEAK["score"], d["expect_score"])
                now = time.time()
                if d["new_confirm"]:
                    sys.stdout.write("\r" + " " * 100 + "\r")
                    if d["expect"]:
                        detail = f"target score {d['expect_score']:.2f} (needs {d['expect_thr']:.2f})"
                    else:
                        detail = f"top {d['raw']} {d['raw_score']:.2f}"
                    print(f"[{now - t_start:5.1f}s] CONFIRMED {d['stable_seal']}  "
                          f"{detail}  hands={d['hands']}")
                if now - last > 0.2:
                    last = now
                    if d["expect"]:
                        info = f"{d['expect']} score={d['expect_score']:.2f}/{d['expect_thr']:.2f}"
                    else:
                        info = f"top={d['raw']} {d['raw_score']:.2f}"
                    sys.stdout.write(f"\r  {info:<30} hold={d['hold']:.2f} "
                                     f"hands={d['hands']} fps={d['fps']:.0f}   ")
                    sys.stdout.flush()
        except (asyncio.CancelledError, ConnectionClosed):
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", default=None, help="target seal, e.g. Hare")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    try:
        asyncio.run(main(a.expect, a.port))
    except KeyboardInterrupt:
        pass
    if a.expect:
        print(f"\nPeak target score this session: {PEAK['score']:.2f}")