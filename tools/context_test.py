"""How much car does the detector need around the plate?

The old traffic-camera detector could only see plates that came with a vehicle around them: at ~46%
of the frame width, hand-held, it scored 0.02 and put the box in the wrong place (lpr_cpp RESUME).
That is a property of the training distribution, not a bug, and it decides whether a "point your
phone at a plate" demo can work at all — so it needs measuring, not assuming.

Method: take one photo with a known plate box, then crop a series of windows centred on the plate,
from "the whole car is in shot" to "the plate fills the frame", re-render each window at a fixed
size, and run the detector on it. The only variable is how much context is visible.

  python tools/context_test.py [--img photo.jpg] [--box x1 y1 x2 y2] [--save-dir scratch/context]
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import infer as I  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# plate width as a share of the window width; 0.05 = whole car in shot, 0.9 = plate fills the frame
SHARES = [0.05, 0.08, 0.12, 0.20, 0.30, 0.45, 0.60, 0.80, 0.95]


def crop_window(rgb, box, share, out_w=640):
    """Window centred on the plate, sized so the plate spans `share` of its width."""
    H, W, _ = rgb.shape
    pw = box[2] - box[0]
    win_w = pw / share
    win_h = win_w * 3.0 / 4.0
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0, y0 = cx - win_w / 2, cy - win_h / 2
    out_h = int(round(out_w * 3 / 4))
    xs = x0 + (np.arange(out_w, dtype=np.float32) + 0.5) * (win_w / out_w) - 0.5
    ys = y0 + (np.arange(out_h, dtype=np.float32) + 0.5) * (win_h / out_h) - 0.5
    img = I._sample(rgb, xs[None, :].repeat(out_h, 0), ys[:, None].repeat(out_w, 1))
    # where the plate lands inside the window, for the IoU check
    sx, sy = out_w / win_w, out_h / win_h
    pbox = ((box[0] - x0) * sx, (box[1] - y0) * sy, (box[2] - x0) * sx, (box[3] - y0) * sy)
    return np.clip(img, 0, 255).astype(np.uint8), pbox


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=os.path.join(ROOT, "assets", "tokyu-bus-yokohama200ka3591.jpg"))
    ap.add_argument("--det", default=os.path.join(ROOT, "models", "plate_det_pyj320.onnx"))
    ap.add_argument("--det-kind", dest="det_kind", default="v8")
    ap.add_argument("--ocr", default=os.path.join(ROOT, "models", "plate_ocr.onnx"))
    ap.add_argument("--box", nargs=4, type=float, default=None)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--save-dir", default="")
    a = ap.parse_args()

    rgb = I.load_rgb(a.img)
    pipe = I.Pipeline(a.det, a.ocr, os.path.join(ROOT, "spec", "labels.txt"), a.det_kind)
    box = a.box
    if box is None:
        found = pipe.detect_v8(rgb, conf=0.25) if a.det_kind == "v8" else pipe.detect(rgb, conf=0.25)
        if not found:
            raise SystemExit("no plate on the original; pass --box x1 y1 x2 y2")
        box = found[0][:4]
        print("plate box from the original: %s" % [round(v, 1) for v in box])
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)

    print("\n%-12s %-14s %8s %6s   %s" % ("plate share", "plate px @640", "det", "IoU", "reading"))
    for share in SHARES:
        img, pbox = crop_window(rgb, box, share)
        boxes = pipe.detect_v8(img, conf=a.conf) if a.det_kind == "v8" else pipe.detect(img, conf=a.conf)
        best, best_iou = None, 0.0
        for b in boxes:
            v = iou(b, pbox)
            if v > best_iou:
                best, best_iou = b, v
        det = "%.3f" % best[4] if best is not None else "none"
        read = "-"
        if best is not None:
            arg, conf, _ = pipe.read(img, best)
            read = "%s (region %.2f)" % (I.L.decode(pipe.spec, arg).text, conf[0])
        print("%-12.2f %-14.0f %8s %6.2f   %s" % (share, pbox[2] - pbox[0], det, best_iou, read))
        if a.save_dir:
            from PIL import Image
            Image.fromarray(img).save(os.path.join(a.save_dir, "share_%03d.png" % int(share * 100)))
    if a.save_dir:
        print("\nwrote the windows to %s" % a.save_dir)
    print("\n(IoU is against the known plate position, so a high score with a low IoU means the\n"
          " detector fired somewhere else — the failure mode that looks like success in a demo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
