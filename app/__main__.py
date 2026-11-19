"""Start the app:  python -m app   (start.bat does this for you).

Starts the local server and opens the browser. If the app is already running, it just opens
the browser on the running copy.
"""

from __future__ import annotations

import argparse
import socket
import threading
import webbrowser

import uvicorn

HOST = "127.0.0.1"   # this computer only; nothing is reachable from the network
PORT = 8756


def _already_running(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, port)) == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = f"http://{HOST}:{args.port}/"

    if _already_running(args.port):
        if not args.no_browser:
            webbrowser.open(url)
        print(f"The digitizer is already running at {url}")
        return

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"AI Embroidery Digitizer is running at {url}")
    print("Leave this window open while you use it. Close it to stop.")
    uvicorn.run("app.api.server:app", host=HOST, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
