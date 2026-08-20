"""Parity test: the graph `jlpr init-det` writes is the graph Ultralytics exports.

`jlpr init-det --from-pt yolov8n.pt` builds a yolov8 ONNX in C++ and fills it from a torch checkpoint,
with no Python in the loop. Two things have to be true for that to be worth anything:

  1. the topology is right — a swapped concat order, a transposed anchor grid or an off-by-half in the
     anchor centres would still *run*, and would still train to something, just not to yolov8;
  2. the transfer is real — the weights that came out of the .pt are in the places that use them.

Both are checked at once by running Ultralytics' own export and ours on the same input and comparing
the *box* rows of the decoded output. Box is the right thing to compare: it flows through the whole
backbone, neck, DFL and anchor arithmetic, and unlike the class rows it transfers unchanged when nc
differs (4*reg_max = 64 channels whatever nc is).

  python tools/parity/init_det.py --pt yolov8n.pt --imgsz 320

The class branch is expected NOT to match: yolov8n.pt is nc=80, whose cv3 middle width is
max(64, min(80,100)) = 80, while an nc=1 head is 64 wide — different shapes, so `init-det` initialises
that branch fresh and says so. That is the "cls head の nc=1 再初期化" the sibling repo calls out.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="a yolov8*.pt checkpoint")
    ap.add_argument("--arch", default="n")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--jlpr", default=os.path.join(ROOT, "jlpr.exe"))
    ap.add_argument("--tol", type=float, default=2e-3, help="max |diff| on the box rows, in pixels")
    a = ap.parse_args()

    import onnxruntime as ort
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed here — cannot anchor against the reference export")
        return 2

    tmp = tempfile.mkdtemp(prefix="initdet")
    ours = os.path.join(tmp, "ours.onnx")
    cmd = [a.jlpr, "init-det", "--out", ours, "--arch", a.arch, "--nc", "1",
           "--imgsz", str(a.imgsz), "--from-pt", a.pt]
    r = subprocess.run(cmd, capture_output=True)
    out = r.stdout.decode("utf-8", "replace")
    if r.returncode != 0 or not os.path.exists(ours):
        print("init-det failed:\n" + out + r.stderr.decode("utf-8", "replace"))
        return 2
    taken = [l for l in out.splitlines() if "taken from the checkpoint" in l]
    print("  init-det:", taken[0].strip() if taken else "(no transfer line)")

    # the reference: Ultralytics' own export of the same checkpoint at the same size
    ref = YOLO(a.pt).export(format="onnx", imgsz=a.imgsz, opset=13, simplify=True, nms=False,
                            dynamic=False, verbose=False)

    x = np.random.default_rng(0).random((1, 3, a.imgsz, a.imgsz), dtype=np.float32)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sa = ort.InferenceSession(open(ours, "rb").read(), so, providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(open(ref, "rb").read(), so, providers=["CPUExecutionProvider"])
    ya = sa.run(None, {sa.get_inputs()[0].name: x})[0]
    yb = sb.run(None, {sb.get_inputs()[0].name: x})[0]
    print("  ours %s   ultralytics %s" % (ya.shape, yb.shape))
    if ya.shape[2] != yb.shape[2]:
        print("  anchor count differs — the graphs are not the same shape at all")
        return 1

    box_a, box_b = ya[0, :4], yb[0, :4]
    d = np.abs(box_a - box_b)
    rel = d.max() / max(1e-9, np.abs(box_b).max())
    print("  box rows: max |diff| %.3e px (of %.1f px max), mean %.3e, rel %.2e"
          % (d.max(), np.abs(box_b).max(), d.mean(), rel))
    # a wrong anchor grid shows up as a large *median* error, not just a tail
    print("  box rows: median |diff| %.3e px" % np.median(d))
    ok = d.max() <= a.tol
    print("PARITY %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
