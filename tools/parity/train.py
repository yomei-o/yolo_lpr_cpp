"""Parity test M5: the two trainers must take the same step.

Both sides are given the same ONNX file, the same data directories, the same seed and a constant
learning rate, and then asked to print the loss of every step. Because the batch sampler draws in the
same order from the same splitmix64 (set choice -> item index -> crop margin), step k sees the same
images and the same labels in both languages, so the losses are comparable numerically — not just
"similar shaped curves".

What can still differ, and why the tolerance is not 1e-7:
  * the C++ engine sums convolutions naively in float32 while PyTorch blocks and vectorises them
    (measured elsewhere: 3.3e-05 on a forward pass of this network),
  * that error is then amplified by 27 layers of backward and by Adam's normalisation, and every
    later step starts from a slightly different parameter vector.
So step 1 is compared tightly (it only needs the batch and the forward to agree — measured 1e-6) and
later steps are compared **relatively**: 3% of the loss value. Measured on the shipped 9-head graph:
step 1 diff 1e-6, steps 2-4 within 1% of the value while the loss falls 14.9 -> 7.0.

  python tools/parity/train.py [--steps 5] [--batch 4] [--synth data/synth5k] [--alpr <root>]
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LOSS_RE = re.compile(r"step\s+(\d+)\s+loss\s+([0-9.eE+-]+)")


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = r.stdout.decode("utf-8", "replace")
    if r.returncode != 0:
        print("FAILED: %s" % " ".join(cmd))
        print(out)
        print(r.stderr.decode("utf-8", "replace"))
        sys.exit(2)
    losses = {}
    for line in out.split("\n"):
        m = LOSS_RE.search(line.strip())
        if m and line.strip().startswith("step"):
            losses[int(m.group(1))] = float(m.group(2))
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jlpr", default=os.path.join(ROOT, "jlpr.exe"))
    ap.add_argument("--init", default=os.path.join(ROOT, "models", "plate_ocr.onnx"))
    ap.add_argument("--synth", default=os.path.join(ROOT, "data", "synth5k"))
    ap.add_argument("--alpr", default="")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tol-first", dest="tol_first", type=float, default=2e-3,
                    help="absolute tolerance on step 1 (batch + forward only)")
    ap.add_argument("--tol-rel", dest="tol_rel", type=float, default=0.03,
                    help="relative tolerance on later steps (optimiser drift in float32)")
    a = ap.parse_args()
    if not os.path.exists(a.jlpr):
        print("no binary at %s (EXTRA=-fopenmp sh build/gcc.sh pure/jlpr.cpp -o jlpr.exe)" % a.jlpr)
        return 2
    if not os.path.isdir(a.synth):
        print("no synthetic data at %s (jlpr gen --out %s --count 2000)" % (a.synth, a.synth))
        return 2

    common = ["--model", "ocr", "--init", a.init, "--synth", a.synth, "--steps", str(a.steps),
              "--batch", str(a.batch), "--lr", str(a.lr), "--seed", str(a.seed), "--dump-loss"]
    if a.alpr:
        common += ["--alpr", a.alpr]
    cpp = run([a.jlpr, "train"] + common)

    # match the head count of the ONNX being trained: the shipped file has 9 heads, a v2 export has 11
    import onnx as _onnx
    n_heads = len(_onnx.load(a.init).graph.output)
    pyc = ["--init", a.init, "--synth", a.synth, "--steps", str(a.steps), "--batch", str(a.batch),
           "--lr", str(a.lr), "--seed", str(a.seed), "--wd", "0", "--dump-loss", "--device", "cpu"]
    if n_heads <= 9:
        pyc.append("--no-extra-heads")
    print("training %d-head graph %s" % (n_heads, os.path.basename(a.init)))
    if a.alpr:
        pyc += ["--alpr", a.alpr]
    py = run([sys.executable, os.path.join(ROOT, "tools", "train_ocr.py")] + pyc)

    steps = sorted(set(cpp) & set(py))
    if not steps:
        print("no comparable steps (C++ %d, Python %d)" % (len(cpp), len(py)))
        return 1
    ok = True
    print("%-6s %12s %12s %10s" % ("step", "C++", "Python", "|diff|"))
    for s in steps:
        d = abs(cpp[s] - py[s])
        tol = a.tol_first if s == 1 else a.tol_rel * max(abs(cpp[s]), abs(py[s]))
        flag = "" if d <= tol else "  <-- over %.4f" % tol
        if d > tol:
            ok = False
        print("%-6d %12.6f %12.6f %10.6f%s" % (s, cpp[s], py[s], d, flag))
    print("M5 parity (training step): %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
