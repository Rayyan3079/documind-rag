"""Capture docs/screenshot.png: start the Streamlit UI, ask one question, screenshot the answer.

Needs a built index (python -m documind ingest), ANTHROPIC_API_KEY, and Playwright
with Chromium (pip install playwright && playwright install chromium).

Usage: python scripts/screenshot.py ["question"]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshot.png"
PORT = 8599
QUESTION = "How many GPIO pins does the Raspberry Pi Pico expose, and how many can be used as ADC inputs?"


def wait_for_server(url: str, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Streamlit did not start at {url}")


def main(question: str = QUESTION) -> int:
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless=true",
         f"--server.port={PORT}", "--browser.gatherUsageStats=false"],
        cwd=ROOT,
    )
    try:
        url = f"http://localhost:{PORT}"
        wait_for_server(url)
        with sync_playwright() as p:
            # CHROMIUM_PATH lets you reuse an already installed Chromium binary.
            browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH"))
            page = browser.new_page(viewport={"width": 1400, "height": 1000}, device_scale_factor=1)
            page.goto(url)
            box = page.get_by_test_id("stChatInputTextArea")
            box.wait_for(timeout=120_000)
            box.fill(question)
            box.press("Enter")
            # The latency/cost caption appears once the answer is rendered.
            page.get_by_text("⏱").wait_for(timeout=180_000)
            page.get_by_text("Sources (").click()
            page.wait_for_timeout(1500)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(OUT), full_page=True)
            browser.close()
        print(f"saved {OUT}")
        return 0
    finally:
        server.terminate()
        server.wait(timeout=30)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
