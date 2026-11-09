"""Python side of `jlpr rot-test`: how reading degrades with plate rotation / viewing angle.

Real plates (the alpr_jp hold-out split — the same 144 the C++ side and the trainers use) are pasted
onto real plate-free backgrounds at a controlled roll/yaw/pitch with the generator's own pinhole
projection, and the whole product path runs on the frame: detector -> (corner net) -> recognizer,
scored on the region head (the only head the real data labels). `--oracle` replaces the detector with
the true bounding box, which separates "the detector lost it" from "it was found but misread".

Same flags, same plate order, same backgrounds, same placement as the C++ command. The pixels differ
slightly (PIL's perspective paste and background resize vs. the C++ sampler — the same accepted
rasteriser difference as the two generators), so the two tables agree to a few plates, not exactly.

  python tools/rot_test.py --alpr ../alpr_jp --tta --rot-search 0,30,-30 --angles 0:0:0,45:0:0
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen as G             # noqa: E402
import gen_det as GD        # noqa: E402
import infer as I           # noqa: E402
import labels as L          # noqa: E402
from rng import Rng         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATEGORIES = ["自家用", "自家用(軽)", "事業用", "事業用(軽)"]   # tools/train_ocr.py (kept torch-free here)
S = 640


def alpr_holdout(root, spec, hold_out=0.2):
    """The alpr_jp hold-out split: same listing, same canonical sort, same Rng(11) shuffle as
    tools/train_ocr.py::AlprSet and trn::split_holdout (C++). Returns [(path, region_index)]."""
    reg = spec.head("region")
    items = []
    for cat in CATEGORIES:
        d = os.path.join(root, cat)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            rd = os.path.join(d, name)
            if not os.path.isdir(rd) or name not in reg.tok:
                continue
            for f in sorted(os.listdir(rd)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    items.append((os.path.join(rd, f), reg.tok.index(name), "%s/%s/%s" % (cat, name, f)))
    items.sort(key=lambda t: t[2])
    rng = Rng(11)
    order = list(range(len(items)))
    for i in range(len(order) - 1, 0, -1):
        j = rng.below(i + 1)
        order[i], order[j] = order[j], order[i]
    cut = int(len(order) * (1 - hold_out))
    return [items[i][:2] for i in sorted(order[cut:])]


def background(path):
    """Centre square, nearest-sampled to SxS exactly as cmd_rot_test does it."""
    from PIL import Image
    canvas = np.full((S, S, 3), 90, dtype=np.uint8)
    try:
        bi = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return canvas
    bh, bw, _ = bi.shape
    side = min(bw, bh)
    ox, oy = (bw - side) // 2, (bh - side) // 2
    g = ((np.arange(S, dtype=np.float32) + np.float32(0.5)) * np.float32(side) / np.float32(S)).astype(np.int64)
    sx = np.clip(ox + g, 0, bw - 1)
    sy = np.clip(oy + g, 0, bh - 1)
    return np.ascontiguousarray(bi[sy[:, None], sx[None, :]])


def paste(canvas, tex_path, quad):
    from PIL import Image
    tex = Image.open(tex_path).convert("RGB")
    coeffs = G._persp_coeffs(quad, tex.size[0], tex.size[1])
    warped = tex.transform((S, S), Image.PERSPECTIVE, data=coeffs, resample=Image.BILINEAR)
    mask = Image.new("L", tex.size, 255).transform((S, S), Image.PERSPECTIVE, data=coeffs,
                                                   resample=Image.BILINEAR)
    im = Image.fromarray(canvas)
    im.paste(warped, (0, 0), mask)
    return np.asarray(im, dtype=np.uint8).copy()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rot_test.py")
    ap.add_argument("--alpr", default=os.path.join(ROOT, "..", "alpr_jp"))
    ap.add_argument("--det", default=os.path.join(ROOT, "models", "plate_det_v8n_640.onnx"))
    ap.add_argument("--ocr", default=os.path.join(ROOT, "models", "plate_ocr_scratch.onnx"))
    ap.add_argument("--corner", default="")
    ap.add_argument("--corner2", default="")
    ap.add_argument("--spec", default=os.path.join(ROOT, "spec", "labels.txt"))
    ap.add_argument("--angles", default="0:0:0,10:0:0,20:0:0,30:0:0,45:0:0,0:30:0,0:50:0,0:0:30,20:40:15")
    ap.add_argument("--dump", default="")
    ap.add_argument("--share", type=float, default=0.30)
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-box", dest="min_box", type=float, default=90.0)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--rot-search", dest="rot_search", default="0")
    ap.add_argument("--rot-fallback", dest="rot_fallback", action="store_true")
    ap.add_argument("--cam-dist", dest="cam_dist", type=float, default=3.0)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--fmt", default="cxcywh", choices=["xyxy", "cxcywh"])
    a = ap.parse_args(argv)

    I.CORNER_MIN_BOX_PX = a.min_box
    sp = L.load(a.spec)
    pipe = I.Pipeline(a.det, a.ocr, a.spec, "v8", a.fmt, corner_path=(a.corner or None),
                      corner2_path=(a.corner2 or None), refine=a.refine)
    items = alpr_holdout(a.alpr, sp)
    if a.limit > 0:
        items = items[:a.limit]
    bgs = GD.image_files(os.path.join(a.alpr, "train", "neg"))
    if not bgs:
        print("no backgrounds under %s/train/neg" % a.alpr)
        return 1
    search = I.parse_angles(a.rot_search)
    angs = [tuple(float(v) for v in (t.split(":") + ["0", "0"])[:3]) for t in a.angles.split(",") if t]

    print("rot-test: %d hold-out plates x %d angles, share %.2f, %s, corner=%s refine=%d"
          % (len(items), len(angs), a.share, "ORACLE box" if a.oracle else "detector box",
             a.corner or "none", a.refine))
    print("rot-search %s%s, cam-dist %.2f" % (a.rot_search, " (fallback only)" if a.rot_fallback else "",
                                               a.cam_dist))
    print("%-16s %7s %7s %7s %8s" % ("roll:yaw:pitch", "found", "region", "|found", "warped"))
    sys.stdout.flush()
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
    for roll, yaw, pitch in angs:
        found = right = warped_n = 0
        for k, (path, want) in enumerate(items):
            canvas = background(bgs[k % len(bgs)])
            p = G.Params()
            p.roll, p.yaw, p.pitch, p.large = roll, yaw, pitch, False
            p.plate_px = a.share * S
            p.cam_dist = a.cam_dist
            cx = S * (0.5 + 0.1 * math.sin(k * 1.7))
            cy = S * (0.5 + 0.1 * math.cos(k * 2.3))
            q = G.project_plate(p, cx, cy)
            canvas = paste(canvas, path, q)
            xs, ys = [c[0] for c in q], [c[1] for c in q]
            gt = [min(xs), min(ys), max(xs), max(ys), 1.0]
            hit = None
            if a.oracle:
                c = pipe.predict_corners(canvas, gt) if pipe.corner is not None else None
                quad = c or [gt[0], gt[1], gt[2], gt[1], gt[2], gt[3], gt[0], gt[3]]
                hit = {"box": gt, "quad": quad, "cornered": c is not None, "angle": 0.0, "rbox": gt}
            else:
                best = 0.3
                for h in pipe.find_plates(canvas, search, 0, a.conf, a.rot_fallback):
                    iou = I.box_iou(h["box"], gt)
                    if iou > best:
                        best, hit = iou, h
            if hit is None:
                continue
            found += 1
            arg, _cf, _n = pipe.read_hit(canvas, hit, a.tta)
            ok = arg[0] == want
            right += int(ok)
            warped_n += int(hit["cornered"])
            if a.dump and k < 6:
                from PIL import Image
                Image.fromarray(canvas).save(os.path.join(
                    a.dump, "r%03d_y%03d_p%03d_%d_%s.png" % (int(roll), int(yaw), int(pitch), k,
                                                           "ok" if ok else "ng")))
        n = max(1, len(items))
        print("%-16s %6.1f%% %6.1f%% %6.1f%% %7.1f%%" % ("%g:%g:%g" % (roll, yaw, pitch), 100.0 * found / n,
              100.0 * right / n, 100.0 * right / max(1, found), 100.0 * warped_n / max(1, found)))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
