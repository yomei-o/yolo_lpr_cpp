"""Parity test M4: the two synthetic generators must be the same generator.

What must match exactly (and does):
  * labels.txt — the 11 head indices per sample
  * meta.txt   — every drawn parameter, including the chosen font
Both are byte-for-byte comparisons, which is only possible because spec/gen.md fixes the order of
every rng draw and both sides use the same splitmix64.

What cannot match, and is therefore compared loosely:
  * the pixels — stb_truetype and PIL/FreeType rasterise differently, and the blur/noise/JPEG chain
    is not bit-reproducible across implementations. With --images the test still checks the recorded
    4 corners (same projection maths, so these agree to ~0.01 px) and prints per-image mean/std so a
    gross divergence would show up.

  python tools/parity/gen.py [--count 500] [--seed 4242] [--images 24] [--gen ./gen_cli.exe]
"""
import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run(cmd, cwd=None):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    if r.returncode != 0:
        print("FAILED: %s\n%s%s" % (" ".join(cmd), r.stdout.decode("utf-8", "replace"),
                                    r.stderr.decode("utf-8", "replace")))
        sys.exit(2)
    return r.stdout.decode("utf-8", "replace")


def read_corners(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 9:
                out[p[0]] = [float(v) for v in p[1:9]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default=os.path.join(ROOT, "jlpr.exe"),
                    help="the C++ CLI; invoked as `<gen> gen ...`")
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--images", type=int, default=0, help="also render this many and compare corners")
    ap.add_argument("--fonts", default=os.path.join(ROOT, "fonts"))
    ap.add_argument("--spec", default=os.path.join(ROOT, "spec", "labels.txt"))
    a = ap.parse_args()

    if not os.path.exists(a.gen):
        print("no generator binary at %s\n  build it: EXTRA=-fopenmp sh build/gcc.sh pure/jlpr.cpp -o jlpr.exe" % a.gen)
        return 2
    if not os.path.isdir(a.fonts) or not os.listdir(a.fonts):
        print("no fonts in %s — run: python tools/fetch_fonts.py --include-system" % a.fonts)
        return 2

    ok = True
    with tempfile.TemporaryDirectory() as td:
        cdir, pdir = os.path.join(td, "c"), os.path.join(td, "p")
        common = ["--count", str(a.count), "--seed", str(a.seed), "--fonts", a.fonts,
                  "--spec", a.spec, "--quiet", "--meta-only"]
        run([a.gen, "gen", "--out", cdir] + common)
        run([sys.executable, os.path.join(ROOT, "tools", "gen.py"), "--out", pdir] + common)
        for f in ("labels.txt", "meta.txt"):
            cb = open(os.path.join(cdir, f), "rb").read()
            pb = open(os.path.join(pdir, f), "rb").read()
            same = cb == pb
            ok = ok and same
            print("%-12s %s  (%d bytes C++, %d bytes Python, %d samples)"
                  % (f, "MATCH" if same else "DIFFER", len(cb), len(pb), a.count))
            if not same:
                cl, pl = cb.split(b"\n"), pb.split(b"\n")
                for i, (x, y) in enumerate(zip(cl, pl)):
                    if x != y:
                        print("  first difference on line %d:\n    C++    : %s\n    Python : %s"
                              % (i + 1, x.decode("utf-8", "replace"), y.decode("utf-8", "replace")))
                        break

        if a.images:
            cdir2, pdir2 = os.path.join(td, "ci"), os.path.join(td, "pi")
            common = ["--count", str(a.images), "--seed", str(a.seed), "--fonts", a.fonts,
                      "--spec", a.spec, "--quiet"]
            run([a.gen, "gen", "--out", cdir2] + common)
            run([sys.executable, os.path.join(ROOT, "tools", "gen.py"), "--out", pdir2] + common)
            ca, pa = read_corners(os.path.join(cdir2, "corners.txt")), read_corners(os.path.join(pdir2, "corners.txt"))
            shared = sorted(set(ca) & set(pa))
            worst = max((max(abs(x - y) for x, y in zip(ca[k], pa[k])) for k in shared), default=0.0)
            good = worst <= 0.05
            ok = ok and good and len(shared) == a.images
            print("corners.txt  %s  worst |C++ - Python| = %.4f px over %d images"
                  % ("MATCH" if good else "DIFFER", worst, len(shared)))
            try:
                import numpy as np
                from PIL import Image
                ds = []
                for k in shared[:8]:
                    ci = np.asarray(Image.open(os.path.join(cdir2, k)).convert("RGB"), dtype=np.float32)
                    pi = np.asarray(Image.open(os.path.join(pdir2, k)).convert("RGB"), dtype=np.float32)
                    ds.append((ci.mean() - pi.mean(), ci.std() - pi.std()))
                print("pixels       not compared (different rasterisers). mean/std deltas on %d images: %s"
                      % (len(ds), ", ".join("%+.1f/%+.1f" % d for d in ds)))
            except Exception as e:                    # pillow/numpy are optional for this part
                print("pixel statistics skipped (%s)" % e)

    print("M4 parity: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
