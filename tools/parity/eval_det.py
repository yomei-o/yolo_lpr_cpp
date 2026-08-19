"""Parity test: the detector metrics must be the same number in both languages — and the mAP must be
the mAP everyone else means by that word.

Two comparisons, because "our two implementations agree" is not the same claim as "this is mAP":

  1. `jlpr val --model det --json`  vs  `python tools/eval_det.py --json` on the same directory and
     the same ONNX. Everything must match: mAP50, mAP50-95, P/R/F1, and every bucket of the recall
     table. Tolerance 1e-3 on the floats (the two run different ONNX interpreters, and the C++ one is
     ~3e-03 off on a 320px detector graph — RESUME), exact on the counts.
  2. our `compute_ap` vs `ultralytics.utils.metrics.compute_ap` on the same precision/recall curve,
     so the integration convention (101-point interpolation, their sentinels) is anchored to theirs
     rather than to our own reading of it.

  python tools/parity/eval_det.py --data data/det_val --det models/plate_det_v8n_320.onnx
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_json(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ROOT)
    out = r.stdout.decode("utf-8", "replace").strip()
    if r.returncode != 0 or not out.startswith("{"):
        print("FAILED: %s" % " ".join(cmd))
        print(out[-2000:])
        print(r.stderr.decode("utf-8", "replace")[-2000:])
        sys.exit(2)
    return json.loads(out)


def check_curve_convention():
    """Anchor our compute_ap to Ultralytics' on random curves."""
    try:
        from ultralytics.utils.metrics import compute_ap as u_compute_ap
    except ImportError:
        print("  (ultralytics not installed here — skipping the compute_ap anchor)")
        return True, 0.0
    import eval_det as E
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        n = int(rng.integers(1, 60))
        rec = np.sort(rng.random(n))
        prec = np.clip(np.sort(rng.random(n))[::-1] + rng.normal(0, 0.05, n), 0, 1)
        ours = E.compute_ap(rec, prec)
        theirs = float(u_compute_ap(rec, prec)[0])
        worst = max(worst, abs(ours - theirs))
    print("  compute_ap vs ultralytics on 20 random curves: worst |diff| %.2e" % worst)
    return worst < 1e-9, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/det_val")
    ap.add_argument("--det", default="models/plate_det_v8n_320.onnx")
    ap.add_argument("--fmt", default="cxcywh", choices=["xyxy", "cxcywh"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jlpr", default=os.path.join(ROOT, "jlpr.exe"))
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args()
    if not os.path.exists(a.jlpr):
        print("no jlpr binary at %s — build it first (build/gcc.sh or build/cc.sh)" % a.jlpr)
        return 2

    common = ["--data", a.data, "--det", a.det, "--fmt", a.fmt, "--conf", str(a.conf), "--json"]
    if a.limit:
        common += ["--limit", str(a.limit)]
    c = run_json([a.jlpr, "val", "--model", "det"] + common)
    p = run_json([sys.executable, os.path.join(ROOT, "tools", "eval_det.py"), "--det-kind", "v8"] + common)

    ok = True
    print("%d frames, %d gt boxes, %s" % (c["frames"], c["gt"], os.path.basename(a.det)))
    for k in ("map50", "map50_95", "precision", "recall", "f1"):
        d = abs(c[k] - p[k])
        ok = ok and d <= a.tol
        print("  %-9s C++ %.4f   python %.4f   diff %.2e" % (k, c[k], p[k], d))
    for k in ("frames", "gt", "tp", "fp", "bucket_fp", "empty_frames", "fp_on_empty"):
        same = c[k] == p[k]
        ok = ok and same
        print("  %-9s C++ %-6d python %-6d %s" % (k, c[k], p[k], "same" if same else "DIFFERENT"))
    for cb, pb in zip(c["buckets"], p["buckets"]):
        same = cb["gt"] == pb["gt"] and cb["found"] == pb["found"]
        ok = ok and same
        if cb["gt"]:
            print("  bucket %2.0f-%2.0f%%  GT %3d found C++ %3d / python %3d %s"
                  % (cb["lo"] * 100, min(cb["hi"], 1.0) * 100, cb["gt"], cb["found"], pb["found"],
                     "" if same else "DIFFERENT"))
    anchored, _ = check_curve_convention()
    ok = ok and anchored
    print("PARITY %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
