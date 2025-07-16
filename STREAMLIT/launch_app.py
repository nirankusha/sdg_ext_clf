#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from pyngrok import ngrok, conf

# ——————— Configuration ———————
# Your ngrok auth token (optional but recommended)
NGROK_AUTH_TOKEN = "2zuQ0T2wjvEGZHTFqH94NZ7iOmy_x9ZBEKxFEU6hxXaB4qaA"
# The port your Streamlit app will listen on
STREAMLIT_PORT = 8501
# Path to your Streamlit app
APP_MODULE     = "app.py"
# Where to write the Streamlit log
LOG_FILE       = "streamlit.log"
# —————————————————————————————————

def setup_pyngrok(token: str):
    """Configure ngrok with your auth token."""
    if token:
        conf.get_default().auth_token = token

def open_tunnel(port: int) -> str:
    """Open an HTTP tunnel on the given port and return the public URL."""
    tunnel = ngrok.connect(port, bind_tls=True)
    return tunnel.public_url

def launch_streamlit(app: str, port: int, logfile: str) -> subprocess.Popen:
    """
    Launch Streamlit in a background subprocess.
    Output (stdout+stderr) is tee’d into logfile.
    """
    cmd = [
        sys.executable, "-m", "streamlit", "run", app,
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.enableCORS", "false",
        "--server.headless", "true"
    ]
    # ensure logs directory exists
    logf = open(logfile, "a")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, env=os.environ.copy())
    return proc

def main():
    # 1. Ensure your code folder is in PYTHONPATH
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + os.pathsep + project_root

    # 2. Setup and open ngrok tunnel
    setup_pyngrok(NGROK_AUTH_TOKEN)
    public_url = open_tunnel(STREAMLIT_PORT)
    print(f"🔗 Ngrok URL: {public_url}")

    # 3. Launch Streamlit
    proc = launch_streamlit(APP_MODULE, STREAMLIT_PORT, LOG_FILE)
    print(f"🚀 Streamlit launched (pid={proc.pid}), logs in ./{LOG_FILE}")

    # 4. Optional: tail the log for a few seconds to confirm startup
    time.sleep(5)
    print("\n--- Streamlit startup logs (last 20 lines) ---")
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()[-20:]
    print("".join(lines))

    # 5. Keep the script alive so the tunnel stays open
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down…")
        proc.terminate()
        ngrok.disconnect(public_url)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:55:34 2025

@author: niran
"""

