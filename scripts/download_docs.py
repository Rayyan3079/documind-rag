"""Download the demo corpus into data/docs/ (see data/docs/SOURCES.md).

The NIST publications are public domain and are committed to the repo; the
Raspberry Pi datasheet is fetched here instead of being redistributed.

Usage: python scripts/download_docs.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DOCS = {
    "NIST.CSWP.29.pdf": "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf",
    "NIST.SP.800-63B-4.pdf": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63B-4.pdf",
    "pico-datasheet.pdf": "https://datasheets.raspberrypi.com/pico/pico-datasheet.pdf",
}


def main(target: Path = Path("data/docs")) -> int:
    target.mkdir(parents=True, exist_ok=True)
    for name, url in DOCS.items():
        dest = target / name
        if dest.exists():
            print(f"exists  {dest}")
            continue
        print(f"fetch   {url}")
        urllib.request.urlretrieve(url, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
