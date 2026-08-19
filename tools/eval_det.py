"""Detector evaluation: mAP, plus recall bucketed by how big the plate is in the frame.

A single mAP number hides the thing this project actually cares about: the borrowed detectors are
excellent on small plates that come with a car around them and blind on close-ups (0.83 at 8% of the
frame width, 0.07 at 20%, nothing at 30%+ — tools/context_test.py). So recall is reported per plate
size bucket, and that table is the acceptance test for M7. mAP50 / mAP50-95 are reported too, with
Ultralytics' matching and their 101-point interpolated `compute_ap`, so a model can be scored without
running Ultralytics — `jlpr val --model det` prints the same numbers from the same algorithm
(pure/eval_det.hpp), and tools/parity/eval_det.py compares the two.

Note the inputs still differ from an `ultralytics val` run even though the metric is the same code:
this project feeds the network a plain resize, Ultralytics letterboxes.

Works on any YOLO-format directory (`images/` + `labels/`), which is what `jlpr gen-det` writes.

  python tools/eval_det.py --data data/det --det models/plate_det_pyj320.onnx --det-kind v8
  python tools/eval_det.py --data data/det --det <ultralytics-export.onnx> --det-kind v8 --fmt cxcywh
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import infer as I  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BUCKETS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.60), (0.60, 1.01)]
IOU_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]


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
        if w <= 0 or h <= 0:
            continue
        out.append([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H, w])
    return out


def match_frame(dets, gts, thresholds):
    """Ultralytics match_predictions: pairs above the threshold, best IoU first, each gt and each
    detection used once. Returns [(conf, [tp at each threshold])] for every detection."""
    pairs = []
    for gi, g in enumerate(gts):
        for di, d in enumerate(dets):
            v = iou(g, d)
            if v > 0:
                pairs.append((v, gi, di))
    pairs.sort(key=lambda p: -p[0])
    out = [[d[4], [0] * len(thresholds)] for d in dets]
    for ti, t in enumerate(thresholds):
        gused, dused = set(), set()
        for v, gi, di in pairs:
            if v < t:
                break
            if gi in gused or di in dused:
                continue
            gused.add(gi)
            dused.add(di)
            out[di][1][ti] = 1
    return out


def compute_ap(recall, precision):
    """Ultralytics compute_ap, 'interp' method: same sentinels, same 101-point trapezoid."""
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(np.interp(x, mrec, mpre), x))


def summarize(scored, n_gt, report_conf):
    """scored = [(conf, [tp per IoU threshold])] over the whole set."""
    scored = sorted(scored, key=lambda s: -s[0])
    aps = []
    for ti in range(len(IOU_THRESHOLDS)):
        rec, prec = [], []
        tpc = fpc = 0
        for conf, tps in scored:
            if tps[ti]:
                tpc += 1
            else:
                fpc += 1
            rec.append(tpc / max(1e-9, n_gt))
            prec.append(tpc / max(1, tpc + fpc))
        aps.append(compute_ap(np.array(rec), np.array(prec)) if rec else 0.0)
    tp = sum(1 for conf, tps in scored if conf >= report_conf and tps[0])
    fp = sum(1 for conf, tps in scored if conf >= report_conf and not tps[0])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_gt if n_gt else 0.0
    return dict(map50=aps[0] if aps else 0.0, map50_95=float(np.mean(aps)) if aps else 0.0,
                precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0,
                tp=tp, fp=fp, fn=n_gt - tp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with images/ and labels/")
    ap.add_argument("--det", default=os.path.join(ROOT, "models", "plate_det_pyj320.onnx"))
    ap.add_argument("--det-kind", dest="det_kind", default="v8", choices=["v8", "yolox", "plateyolo"])
    ap.add_argument("--fmt", default="xyxy", choices=["xyxy", "cxcywh"],
                    help="box layout of a v8-style head: PlateYOLO(NMS stripped)=xyxy, a plain "
                         "Ultralytics export=cxcywh")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--conf-lo", dest="conf_lo", type=float, default=0.001,
                    help="threshold the detector actually runs at; mAP needs the weak detections too. "
                         "NMS keeps the strongest box first, so filtering afterwards at --conf gives "
                         "exactly what a --conf run would have produced")
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    idir = os.path.join(a.data, "images")
    ldir = os.path.join(a.data, "labels")
    files = sorted(f for f in os.listdir(idir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if a.limit:
        files = files[:a.limit]
    pipe = I.Pipeline(a.det, None, os.path.join(ROOT, "spec", "labels.txt"), a.det_kind, a.fmt)
    conf_lo = min(a.conf, a.conf_lo)

    tp = [0] * len(BUCKETS)
    gt = [0] * len(BUCKETS)
    scored = []
    n_gt = 0
    fp_total = 0
    det_total = 0
    empty_frames = fp_empty = 0
    for n, f in enumerate(files, 1):
        rgb = I.load_rgb(os.path.join(idir, f))
        H, W, _ = rgb.shape
        want = read_labels(os.path.join(ldir, os.path.splitext(f)[0] + ".txt"), W, H)
        if a.det_kind == "plateyolo":
            got = pipe.detect_plateyolo(rgb, conf_lo)
        elif a.det_kind == "v8":
            got = pipe.detect_v8(rgb, conf_lo, nms=a.nms, fmt=a.fmt)
        else:
            got = pipe.detect(rgb, conf=conf_lo, nms=a.nms)
        strong = [d for d in got if d[4] >= a.conf]
        n_gt += len(want)
        det_total += len(strong)
        scored += match_frame(got, want, IOU_THRESHOLDS)
        if not want:
            empty_frames += 1
            fp_empty += len(strong)
        used = [False] * len(strong)
        for g in want:
            share = g[4]
            b = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= share < hi)
            gt[b] += 1
            best, bi = 0.0, -1
            for i, d in enumerate(strong):
                if used[i]:
                    continue
                v = iou(d, g)
                if v > best:
                    best, bi = v, i
            if best >= a.iou:
                tp[b] += 1
                used[bi] = True
        fp_total += sum(1 for u in used if not u)
        if n % 100 == 0 and not a.json:
            print("  %d/%d" % (n, len(files)), flush=True)

    m = summarize(scored, n_gt, a.conf)
    if a.json:
        print(json.dumps(dict(frames=len(files), gt=n_gt, map50=m["map50"], map50_95=m["map50_95"],
                              precision=m["precision"], recall=m["recall"], f1=m["f1"],
                              tp=m["tp"], fp=m["fp"],
                              buckets=[dict(lo=lo, hi=hi, gt=g, found=t)
                                       for (lo, hi), t, g in zip(BUCKETS, tp, gt)],
                              bucket_fp=fp_total, empty_frames=empty_frames, fp_on_empty=fp_empty)))
        return 0

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
    print("mAP50 %.4f  mAP50-95 %.4f   (P %.4f  R %.4f  F1 %.4f at conf %.2f, TP %d FP %d FN %d)"
          % (m["map50"], m["map50_95"], m["precision"], m["recall"], m["f1"], a.conf,
             m["tp"], m["fp"], m["fn"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
