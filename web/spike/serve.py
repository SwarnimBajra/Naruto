import argparse
import functools
import http.server
import socket
from pathlib import Path


class Handler(http.server.SimpleHTTPRequestHandler):
    isolate = False
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".mjs": "text/javascript",
                      ".js": "text/javascript", ".onnx": "application/octet-stream",
                      ".json": "application/json"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        if Handler.isolate:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--isolate", action="store_true",
                    help="send COOP/COEP headers (needed for multi-threaded WASM)")
    a = ap.parse_args()
    Handler.isolate = a.isolate
    handler = functools.partial(Handler, directory=str(Path(__file__).parent))
    print(f"Laptop:  http://localhost:{a.port}")
    print(f"Phone:   http://{lan_ip()}:{a.port}   (same Wi-Fi)")
    http.server.ThreadingHTTPServer(("0.0.0.0", a.port), handler).serve_forever()