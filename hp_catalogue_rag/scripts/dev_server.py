"""Start backend (FastAPI :8000) and frontend (Streamlit :8501) together. Ctrl+C stops both.

    python scripts/dev_server.py [--backend-port 8000] [--frontend-port 8501]
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-port", type=int, default=8000)
    ap.add_argument("--frontend-port", type=int, default=8501)
    a = ap.parse_args()

    env = dict(os.environ, BACKEND_URL=f"http://localhost:{a.backend_port}")
    procs = [
        subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app.main:app",
                          "--port", str(a.backend_port)], cwd=ROOT, env=env),
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py",
                          "--server.port", str(a.frontend_port),
                          "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env),
    ]
    print(f"\nAPI docs:  http://localhost:{a.backend_port}/docs\nChat UI:   http://localhost:{a.frontend_port}\n")
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
