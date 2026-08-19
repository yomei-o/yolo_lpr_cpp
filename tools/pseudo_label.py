"""Auto-label real photos with the borrowed detector, to fill the gap our synthetic data leaves.

Why: our composites paste plates onto whatever backgrounds we have, so the detector trained on them
is excellent at close-ups (0.66-0.90 for plates >= 12% of the frame) and weak at the far end (0.09 at
5%). The borrowed PlateYOLO-JP is the mirror image — 0.76-0.83 at 5-12%, nothing past 20%. So it is
a *good teacher exactly where we are bad*, and dyama/alpr_jp ships 1,330 real vehicle photos
(train/sample/) that nobody has labelled.

  python tools/pseudo_label.py --src ../alpr_jp/train/sample --out data/det_real \\
         --det models/plate_det_pyj320.onnx --imgsz 640 --conf 0.5

Output is a standard YOLO directory (images/ + labels/, class 0 = plate) that mixes straight into
`jlpr gen-det` output. Every image is written at --imgsz so the labels stay valid, and images with no
confident detection are skipped (a missing plate would teach the model that plates are background).
`--keep-empty` writes them anyway as hard negatives.
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


def list_images(src):
    out = []
    for root, _dirs, files in os.walk(src):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append(os.path.join(root, f))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of real photos (searched recursively)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--det", default=os.path.join(ROOT, "models", "plate_det_pyj320.onnx"))
    ap.add_argument("--det-kind", dest="det_kind", default="v8")
    ap.add_argument("--fmt", default="xyxy", choices=["xyxy", "cxcywh"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-empty", dest="keep_empty", action="store_true")
    a = ap.parse_args()

    files = list_images(a.src)
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit("no images under %s" % a.src)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    pipe = I.Pipeline(a.det, None, os.path.join(ROOT, "spec", "labels.txt"), a.det_kind, a.fmt)

    from PIL import Image
    kept = boxes_total = empty = 0
    for n, path in enumerate(files, 1):
        rgb = I.load_rgb(path)
        H, W, _ = rgb.shape
        boxes = (pipe.detect_v8(rgb, a.conf, fmt=a.fmt) if a.det_kind == "v8"
                 else pipe.detect(rgb, conf=a.conf))
        if not boxes and not a.keep_empty:
            empty += 1
            continue
        name = "real%05d.png" % kept
        im = Image.fromarray(rgb).resize((a.imgsz, a.imgsz), Image.BILINEAR)
        im.save(os.path.join(a.out, "images", name))
        with open(os.path.join(a.out, "labels", name[:-4] + ".txt"), "wb") as f:
            for b in boxes:
                xc = (b[0] + b[2]) * 0.5 / W
                yc = (b[1] + b[3]) * 0.5 / H
                bw = (b[2] - b[0]) / W
                bh = (b[3] - b[1]) / H
                f.write(("0 %.6f %.6f %.6f %.6f\n" % (xc, yc, bw, bh)).encode("utf-8"))
                boxes_total += 1
        kept += 1
        if n % 100 == 0:
            print("  %d/%d  kept %d  boxes %d  empty %d" % (n, len(files), kept, boxes_total, empty),
                  flush=True)

    print("\nwrote %d images and %d boxes into %s (skipped %d with no confident detection)"
          % (kept, boxes_total, a.out, empty))
    print("teacher: %s at conf>=%.2f — these labels are only as good as that model, which is why they\n"
          "are mixed with synthetic data rather than used alone." % (os.path.basename(a.det), a.conf))
    return 0


if __name__ == "__main__":
    sys.exit(main())
