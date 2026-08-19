"""Detector evaluation, bucketed by how big the plate is in the frame.

A single mAP number hides the thing this project actually cares about: the borrowed detectors are
excellent on small plates that come with a car around them and blind on close-ups (0.83 at 8% of the
frame width, 0.07 at 20%, nothing at 30%+ — tools/context_test.py). So recall is reported per plate
size bucket, and that table is the acceptance test for M7.

Works on any YOLO-format directory (`images/` + `labels/`), which is what `jlpr gen-det` writes.

  python tools/eval_det.py --data data/det --det models/plate_det_pyj320.onnx --det-kind v8
  python tools/eval_det.py --data data/det --det <ultralytics-export.onnx> --det-kind v8 --fmt cxcywh
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

BUCKETS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.60), (0.60, 1.01)]


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def read_labels(path, W, H):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        xc, yc, w, h = [float(v) for v in p[1:5]]
        out.append([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H, w])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with images/ and labels/")
    ap.add_argument("--det", default=os.path.join(ROOT, "models", "plate_det_pyj320.onnx"))
    ap.add_argument("--det-kind", dest="det_kind", default="v8", choices=["v8", "yolox", "plateyolo"])
    ap.add_argument("--fmt", default="xyxy", choices=["xyxy", "cxcywh"],
                    help="box layout of a v8-style head: PlateYOLO(NMS stripped)=xyxy, a plain "
                         "Ultralytics export=cxcywh")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    idir = os.path.join(a.data, "images")
    ldir = os.path.join(a.data, "labels")
    files = sorted(f for f in os.listdir(idir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if a.limit:
        files = files[:a.limit]
    pipe = I.Pipeline(a.det, None, os.path.join(ROOT, "spec", "labels.txt"), a.det_kind, a.fmt)

    tp = [0] * len(BUCKETS)
    gt = [0] * len(BUCKETS)
    fp_total = 0
    det_total = 0
    empty_frames = fp_empty = 0
    for n, f in enumerate(files, 1):
        rgb = I.load_rgb(os.path.join(idir, f))
        H, W, _ = rgb.shape
        want = read_labels(os.path.join(ldir, os.path.splitext(f)[0] + ".txt"), W, H)
        if a.det_kind == "plateyolo":
            got = pipe.detect_plateyolo(rgb, a.conf)
        elif a.det_kind == "v8":
            got = pipe.detect_v8(rgb, a.conf, fmt=a.fmt)
        else:
            got = pipe.detect(rgb, conf=a.conf)
        det_total += len(got)
        if not want:
            empty_frames += 1
            fp_empty += len(got)
        used = [False] * len(got)
        for g in want:
            share = g[4]
            b = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= share < hi)
            gt[b] += 1
            best, bi = 0.0, -1
            for i, d in enumerate(got):
                if used[i]:
                    continue
                v = iou(d, g)
                if v > best:
                    best, bi = v, i
            if best >= a.iou:
                tp[b] += 1
                used[bi] = True
        fp_total += sum(1 for u in used if not u)
        if n % 100 == 0:
            print("  %d/%d" % (n, len(files)), flush=True)

    print("\n%d frames, conf>=%.2f, IoU>=%.2f, model=%s (%s)"
          % (len(files), a.conf, a.iou, os.path.basename(a.det), a.det_kind))
    print("%-16s %8s %8s %8s" % ("plate share", "GT", "found", "recall"))
    for (lo, hi), t, g in zip(BUCKETS, tp, gt):
        if g == 0:
            continue
        print("%-16s %8d %8d %7.1f%%" % ("%.0f-%.0f%%" % (lo * 100, min(hi, 1.0) * 100), g, t,
                                         100.0 * t / g))
    tot_g, tot_t = sum(gt), sum(tp)
    print("%-16s %8d %8d %7.1f%%" % ("all", tot_g, tot_t, 100.0 * tot_t / max(1, tot_g)))
    print("false positives: %d (of %d detections); on the %d plate-free frames: %d"
          % (fp_total, det_total, empty_frames, fp_empty))
    return 0


if __name__ == "__main__":
    sys.exit(main())
