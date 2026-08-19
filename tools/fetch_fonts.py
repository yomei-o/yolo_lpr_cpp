"""Fetch the fonts the synthetic generator draws with, into ./fonts (gitignored).

Fonts are NOT committed: some free-for-use Japanese fonts allow use but say nothing clear about
redistribution, and the repo should not gamble on that. Both generators (C++ and Python) read the
same files from ./fonts, and parity requires the same font set on both sides — the list below is
therefore part of the contract, in this order.

  python tools/fetch_fonts.py [--list] [--include-system]

1. GenSenRounded2-B.ttc  — SIL OFL 1.1, full Japanese coverage. Fetched from the PlateYOLO-JP
   repository, which ships it with its licence (see THIRD_PARTY_NOTICES.md).
2. Windows system fonts (--include-system): meiryo / msgothic / YuGothB are already on the machine
   and give three more shapes for free. They are NOT redistributable, so they are opt-in and the
   generator records which fonts it used.

The real plate typeface is a specific squarish design that no free font reproduces exactly; the plan
is deliberately to train across several shapes instead of betting on one (README データ戦略).
"""
import argparse
import os
import shutil
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "fonts")

REMOTE = [
    ("GenSenRounded2-B.ttc",
     "https://raw.githubusercontent.com/Kazuhito00/PlateYOLO-JP-Prototype/main/font/gensen-font/ttc/GenSenRounded2-B.ttc",
     "SIL OFL 1.1"),
    ("SIL_Open_Font_License_1.1.txt",
     "https://raw.githubusercontent.com/Kazuhito00/PlateYOLO-JP-Prototype/main/font/gensen-font/SIL_Open_Font_License_1.1.txt",
     "licence text"),
]

SYSTEM = [
    ("meiryo.ttc", r"C:\Windows\Fonts\meiryo.ttc"),
    ("msgothic.ttc", r"C:\Windows\Fonts\msgothic.ttc"),
    ("YuGothB.ttc", r"C:\Windows\Fonts\YuGothB.ttc"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show what is present and exit")
    ap.add_argument("--include-system", action="store_true",
                    help="copy Windows system fonts too (not redistributable)")
    a = ap.parse_args()

    os.makedirs(FONT_DIR, exist_ok=True)
    if a.list:
        for f in sorted(os.listdir(FONT_DIR)):
            print("  %-32s %8d bytes" % (f, os.path.getsize(os.path.join(FONT_DIR, f))))
        return 0

    for name, url, note in REMOTE:
        dst = os.path.join(FONT_DIR, name)
        if os.path.exists(dst):
            print("have %s (%s)" % (name, note))
            continue
        print("fetching %s ... " % name, end="", flush=True)
        with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
        print("%d bytes (%s)" % (os.path.getsize(dst), note))

    if a.include_system:
        for name, src in SYSTEM:
            dst = os.path.join(FONT_DIR, name)
            if os.path.exists(dst):
                print("have %s (system)" % name)
            elif os.path.exists(src):
                shutil.copyfile(src, dst)
                print("copied %s from %s (NOT redistributable)" % (name, src))
            else:
                print("skip %s (not found at %s)" % (name, src))

    print("\nfonts in %s:" % FONT_DIR)
    for f in sorted(os.listdir(FONT_DIR)):
        print("  %s" % f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
