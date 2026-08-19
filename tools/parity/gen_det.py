"""Parity test M4b: the two detection-data generators must be the same generator.

Compares, byte for byte, what the labels depend on:
  * meta.txt    — every draw per frame and per plate (scale, position, kind, text, degradation)
  * corners.txt — the 4 corners of every kept plate, normalised
and, with --images, the YOLO label files themselves. Pixels are not compared (different rasterisers).

  python tools/parity/gen_det.py [--count 300] [--seed 2024] [--bg <dir>] [--images 8]
"""
import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        print("FAILED: %s" % " ".join(cmd))
        print(r.stdout.decode("utf-8", "replace"))
        print(r.stderr.decode("utf-8", "replace"))
        sys.exit(2)
    return r.stdout.decode("utf-8", "replace")


def first_diff_line(cb, pb):
    for i, (x, y) in enumerate(zip(cb.split(b"\n"), pb.split(b"\n"))):
        if x != y:
            return i + 1, x.decode("utf-8", "replace"), y.decode("utf-8", "replace")
    return None, "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default=os.path.join(ROOT, "jlpr.exe"))
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--bg", default="")
    ap.add_argument("--images", type=int, default=0,
                    help="also render this many frames and diff the YOLO label files")
    ap.add_argument("--fonts", default=os.path.join(ROOT, "fonts"))
    ap.add_argument("--spec", default=os.path.join(ROOT, "spec", "labels.txt"))
    a = ap.parse_args()
    if not os.path.exists(a.gen):
        print("no binary at %s (EXTRA=-fopenmp sh build/gcc.sh pure/jlpr.cpp -o jlpr.exe)" % a.gen)
        return 2

    ok = True
    with tempfile.TemporaryDirectory() as td:
        base = ["--seed", str(a.seed), "--imgsz", str(a.imgsz), "--fonts", a.fonts,
                "--spec", a.spec, "--quiet"]
        if a.bg:
            base += ["--bg", a.bg]
        meta = base + ["--count", str(a.count), "--meta-only"]
        c, p = os.path.join(td, "c"), os.path.join(td, "p")
        run([a.gen, "gen-det", "--out", c] + meta)
        run([sys.executable, os.path.join(ROOT, "tools", "gen_det.py"), "--out", p] + meta)
        for f in ("meta.txt", "corners.txt"):
            cb = open(os.path.join(c, f), "rb").read()
            pb = open(os.path.join(p, f), "rb").read()
            same = cb == pb
            ok = ok and same
            print("%-12s %s  (%d bytes C++, %d bytes Python, %d frames)"
                  % (f, "MATCH" if same else "DIFFER", len(cb), len(pb), a.count))
            if not same:
                n, x, y = first_diff_line(cb, pb)
                print("  line %s:" % n)
                print("    C++    : %s" % x)
                print("    Python : %s" % y)

        if a.images:
            imgs = base + ["--count", str(a.images)]
            c2, p2 = os.path.join(td, "ci"), os.path.join(td, "pi")
            run([a.gen, "gen-det", "--out", c2] + imgs)
            run([sys.executable, os.path.join(ROOT, "tools", "gen_det.py"), "--out", p2] + imgs)
            names = sorted(os.listdir(os.path.join(c2, "labels")))
            bad = 0
            for n in names:
                cb = open(os.path.join(c2, "labels", n), "rb").read()
                pb = open(os.path.join(p2, "labels", n), "rb").read()
                bad += int(cb != pb)
            print("labels/      %s  %d/%d YOLO label files identical"
                  % ("MATCH" if bad == 0 else "DIFFER", len(names) - bad, len(names)))
            ok = ok and bad == 0

    print("M4b parity: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
