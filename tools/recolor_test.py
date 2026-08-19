"""Plate-colour bias test: same photo, same geometry, only the plate's colours changed.

This reproduces the experiment that exposed the old detector's bias (lpr_cpp measured 白 0.85 /
黄 0.40 / 黒 0.21 on one photo), so a detector swap can be judged on the axis that actually mattered
rather than on the one white plate everybody tests with.

How the recolour works: inside the plate box, the luminance is min-max normalised to t in [0,1]
(t=0 is the darkest ink, t=1 the brightest background) and the pixel is replaced by
`fg + (bg - fg) * t`. Shading, blur and the glyph shapes survive; only the palette changes.

  python tools/recolor_test.py --img <photo> [--box x1 y1 x2 y2] [--save-dir scratch/recolor]

With no --box it runs the detector once on the original to find the plate.
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

# kind -> (background, foreground) as in spec/gen.md
PALETTES = {
    "白地緑字 (自家用)":   ((250, 250, 248), (16, 90, 60)),
    "緑地白字 (事業用)":   ((16, 90, 60), (250, 250, 248)),
    "黄地黒字 (軽自家用)": ((240, 205, 20), (25, 25, 25)),
    "黒地黄字 (軽事業用)": ((25, 25, 25), (240, 205, 20)),
}


def recolour(rgb, box, bg, fg):
    out = rgb.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
    patch = rgb[y1:y2, x1:x2].astype(np.float32)
    lum = patch @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lo, hi = np.percentile(lum, 2), np.percentile(lum, 98)
    t = np.clip((lum - lo) / max(1e-6, hi - lo), 0.0, 1.0)[..., None]
    new = np.array(fg, dtype=np.float32) + (np.array(bg, dtype=np.float32) - np.array(fg, dtype=np.float32)) * t
    out[y1:y2, x1:x2] = np.clip(new, 0, 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=os.path.join(ROOT, "assets", "tokyu-bus-yokohama200ka3591.jpg"))
    ap.add_argument("--det", default=os.path.join(ROOT, "models", "plate_det_pyj320.onnx"))
    ap.add_argument("--det-kind", dest="det_kind", default="v8")
    ap.add_argument("--fmt", default="xyxy", choices=["xyxy", "cxcywh"],
                    help="box layout of a v8 head: PlateYOLO(NMS stripped)=xyxy, plain Ultralytics "
                         "export=cxcywh. Reading one as the other puts plausible boxes in the wrong "
                         "place — which is how this very script first reported 'no detections'")
    ap.add_argument("--ocr", default=os.path.join(ROOT, "models", "plate_ocr.onnx"))
    ap.add_argument("--conf", type=float, default=0.05, help="low, so a weak detection still shows")
    ap.add_argument("--box", nargs=4, type=float, default=None)
    ap.add_argument("--save-dir", default="")
    a = ap.parse_args()

    rgb = I.load_rgb(a.img)
    pipe = I.Pipeline(a.det, a.ocr, os.path.join(ROOT, "spec", "labels.txt"), a.det_kind, a.fmt)

    box = a.box
    if box is None:
        found = pipe.detect_v8(rgb, conf=0.25, fmt=a.fmt) if a.det_kind == "v8" else pipe.detect(rgb, conf=0.25)
        if not found:
            raise SystemExit("no plate found on the original image; pass --box x1 y1 x2 y2")
        box = found[0][:4]
        print("plate box from the original: %s" % [round(v, 1) for v in box])

    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)

    print("\n%-22s %8s  %-24s %s" % ("plate colours", "det", "reading (detected box)", "reading (given box)"))
    for name, (bg, fg) in PALETTES.items():
        img = recolour(rgb, box, bg, fg)
        boxes = pipe.detect_v8(img, conf=a.conf, fmt=a.fmt) if a.det_kind == "v8" else pipe.detect(img, conf=a.conf)
        # keep the detection that overlaps the known plate
        best, best_iou = None, 0.0
        for b in boxes:
            iw = min(b[2], box[2]) - max(b[0], box[0])
            ih = min(b[3], box[3]) - max(b[1], box[1])
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            ua = (b[2] - b[0]) * (b[3] - b[1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter
            if inter / ua > best_iou:
                best, best_iou = b, inter / ua
        det = "%.3f" % best[4] if best else "none"
        read_det = "-"
        if best:
            arg, conf, _ = pipe.read(img, best)
            read_det = "%s (%.2f)" % (I.L.decode(pipe.spec, arg).text, conf[0])
        arg, conf, _ = pipe.read(img, list(box) + [1.0])
        read_box = "%s (%.2f)" % (I.L.decode(pipe.spec, arg).text, conf[0])
        print("%-22s %8s  %-24s %s" % (name, det, read_det, read_box))
        if a.save_dir:
            from PIL import Image
            Image.fromarray(img).save(os.path.join(a.save_dir, "%s.png" % name.split(" ")[0]))
    if a.save_dir:
        print("\nwrote recoloured frames to %s" % a.save_dir)
    print("\n(det column = detector score on the plate; 'reading (given box)' isolates the recognizer\n"
          " from the detector by handing it the same box every time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
