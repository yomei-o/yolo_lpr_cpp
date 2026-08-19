"""Detection training data (Python side) — mirror of pure/gen_det.hpp + `jlpr gen-det`.

Full frames with plates pasted at 3%..95% of the frame width, plus empty frames as hard negatives,
because the borrowed detectors only fire on small plates that come with a car around them
(tools/context_test.py: 8% of frame → 0.83, 20% → 0.07, 30%+ → nothing).

  python tools/gen_det.py --out data/det --count 500 [--imgsz 640] [--bg <dir>] [--meta-only]

labels/ is standard YOLO (`0 xc yc w h`, class 0 = plate) so Ultralytics or either language's own
trainer can read it. meta.txt + labels are what tools/parity/gen_det.py compares against C++.
"""
import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import labels as L      # noqa: E402
import gen as G         # noqa: E402
from rng import Rng     # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DET_SALT = 0xD1B54A32D192ED03


class DetSample(object):
    pass


def sample_det(rng, sp, n_bg, n_fonts, n_real=0, real_pct=0):
    """Mirror of gen::sample_det — draw order is the contract (spec/gen.md, detection section)."""
    d = DetSample()
    np_ = rng.below(100)                                              # D1
    d.n_plates = 0 if np_ < 8 else (1 if np_ < 75 else (2 if np_ < 95 else 3))
    d.bg_real = rng.below(100) < 70 and n_bg > 0                      # D2
    d.bg_idx = rng.below(n_bg) if n_bg > 0 else 0                     # D3 (always drawn)
    d.bg_hue = rng.range(0, 1)                                        # D4
    d.bg_dark = rng.range(0.15, 0.85)                                 # D5
    d.plates = []
    for k in range(d.n_plates):
        pl = DetSample()
        pl.p = G.sample(rng, sp)                                      # D6
        pl.share = math.exp(rng.range(math.log(0.03), math.log(0.95)))  # D7
        pl.cx = rng.range(0.15, 0.85)                                 # D8
        pl.cy = rng.range(0.15, 0.85)                                 # D9
        pl.font_idx = rng.below(n_fonts) if n_fonts > 0 else 0         # D10
        # D11/D12 are always drawn so the stream depends on the flags, not on what is on disk
        pl.use_real = rng.below(100) < real_pct and n_real > 0          # D11
        pl.real_idx = rng.below(n_real if n_real > 0 else 1)            # D12
        d.plates.append(pl)
    d.brightness = rng.range(0.5, 1.3)                                # D13
    d.contrast = rng.range(0.75, 1.25)                                # D14
    d.warm = rng.range(-0.1, 0.1)                                     # D15
    d.blur = rng.range(0, 1.2)                                        # D16
    d.motion = rng.range(0, 2.0)                                      # D17
    d.noise = rng.range(0, 8)                                         # D18
    d.jpeg_q = rng.below(50) + 50                                     # D19
    return d


def place_det(d, imgsz):
    for pl in d.plates:
        pl.p.plate_px = pl.share * imgsz
        pl.quad = G.project_plate(pl.p, pl.cx * imgsz, pl.cy * imgsz)
        xs = [q[0] for q in pl.quad]
        ys = [q[1] for q in pl.quad]
        full = max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
        pl.bx0 = min(max(min(xs), 0.0), imgsz)
        pl.by0 = min(max(min(ys), 0.0), imgsz)
        pl.bx1 = min(max(max(xs), 0.0), imgsz)
        pl.by1 = min(max(max(ys), 0.0), imgsz)
        vis = max(0.0, pl.bx1 - pl.bx0) * max(0.0, pl.by1 - pl.by0)
        pl.keep = (vis / full >= 0.35) and (pl.bx1 - pl.bx0) >= 6 and (pl.by1 - pl.by0) >= 4


def det_labels(d, imgsz):
    out = []
    for pl in d.plates:
        if not pl.keep:
            continue
        out.append("0 %.6f %.6f %.6f %.6f\n"
                   % ((pl.bx0 + pl.bx1) * 0.5 / imgsz, (pl.by0 + pl.by1) * 0.5 / imgsz,
                      (pl.bx1 - pl.bx0) / imgsz, (pl.by1 - pl.by0) / imgsz))
    return "".join(out)


def det_corners(file, d, imgsz):
    out = []
    for k, pl in enumerate(d.plates):
        if not pl.keep:
            continue
        v = []
        for (x, y) in pl.quad:
            v += [x / imgsz, y / imgsz]
        out.append("%s %d %s\n" % (file, k, " ".join("%.5f" % z for z in v)))
    return "".join(out)


def det_meta(file, d, sp, bg_name, imgsz):
    kept = sum(1 for pl in d.plates if pl.keep)
    out = ["%s imgsz=%d plates=%d kept=%d bg=%s hue=%.4f dark=%.4f brightness=%.4f contrast=%.4f "
           "warm=%.4f blur=%.4f motion=%.4f noise=%.4f jpeg_q=%d\n"
           % (file, imgsz, d.n_plates, kept, bg_name, d.bg_hue, d.bg_dark, d.brightness,
              d.contrast, d.warm, d.blur, d.motion, d.noise, d.jpeg_q)]
    for k, pl in enumerate(d.plates):
        out.append("  %d share=%.4f cx=%.4f cy=%.4f keep=%d real=%d ridx=%d text=%s kind=%s\n"
                   % (k, pl.share, pl.cx, pl.cy, 1 if pl.keep else 0,
                      1 if pl.use_real else 0, pl.real_idx,
                      L.decode(sp, G.heads(pl.p)).text, sp.head("plate_kind").tok[pl.p.kind]))
    return "".join(out)


def image_files(dirs):
    """Recursive, comma-separated list of directories -> sorted image paths (same order as C++)."""
    out = []
    for d in (dirs.split(",") if dirs else []):
        d = d.strip()
        if not d or not os.path.isdir(d):
            continue
        for root, _sub, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    out.append(os.path.join(root, f).replace(os.sep, "/"))
    return sorted(out)


def bg_files(bg_dir):
    return image_files(bg_dir)


def render_det(d, sp, font_paths, bgs, imgsz, rng, reals=None):
    from PIL import Image, ImageFilter
    canvas = None
    if d.bg_real and bgs:
        try:
            im = Image.open(bgs[d.bg_idx]).convert("RGB")
            side = min(im.size)
            ox, oy = (im.width - side) // 2, (im.height - side) // 2
            canvas = im.crop((ox, oy, ox + side, oy + side)).resize((imgsz, imgsz), Image.BILINEAR)
        except Exception:
            canvas = None
    if canvas is None:
        base = np.zeros((imgsz, imgsz, 3), dtype=np.float32)
        hue = d.bg_hue * 6.0
        i = int(hue)
        fr = hue - i
        v, s = d.bg_dark, 0.35
        pv, qv, tv = v * (1 - s), v * (1 - s * fr), v * (1 - s * (1 - fr))
        rgb = [(v, tv, pv), (qv, v, pv), (pv, v, tv), (pv, qv, v), (tv, pv, v), (v, pv, qv)][i % 6]
        yy, _ = np.mgrid[0:imgsz, 0:imgsz]
        base += np.array(rgb, dtype=np.float32) * 255.0 * (0.75 + 0.5 * yy / imgsz)[..., None]
        canvas = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
        canvas = canvas.filter(ImageFilter.GaussianBlur(max(1.0, imgsz * 0.02)))

    for pl in d.plates:
        tex = None
        if getattr(pl, "use_real", False) and reals:
            try:
                from PIL import Image as _I
                tex = _I.open(reals[pl.real_idx % len(reals)]).convert("RGB")
            except Exception:
                tex = None
        if tex is None:
            tex = G.plate_texture(pl.p, sp, font_paths[pl.font_idx % len(font_paths)], rng)
        coeffs = G._persp_coeffs(pl.quad, tex.size[0], tex.size[1])
        warped = tex.transform((imgsz, imgsz), Image.PERSPECTIVE, data=coeffs, resample=Image.BILINEAR)
        mask = Image.new("L", tex.size, 255).transform((imgsz, imgsz), Image.PERSPECTIVE,
                                                       data=coeffs, resample=Image.BILINEAR)
        canvas.paste(warped, (0, 0), mask)

    a = np.asarray(canvas, dtype=np.float32)
    a = (a / 255.0 - 0.5) * d.contrast + 0.5
    a *= d.brightness
    a[:, :, 0] *= (1.0 + d.warm)
    a[:, :, 2] *= (1.0 - d.warm)
    a = np.clip(a * 255.0, 0, 255)
    if d.noise > 0.1:
        u = rng.units(2 * a.size)
        u1 = np.maximum(1e-9, u[0::2])
        u2 = u[1::2]
        g = (np.sqrt(-2 * np.log(u1)) * np.cos(2 * G.PI * u2)).reshape(a.shape)
        a = np.clip(a + g * d.noise, 0, 255)
    out = Image.fromarray(a.astype(np.uint8))
    if d.blur >= 0.05:
        out = out.filter(ImageFilter.GaussianBlur(d.blur))
    if d.motion >= 1:
        n = int(round(d.motion))
        acc = np.zeros((imgsz, imgsz, 3), dtype=np.float32)
        base = np.asarray(out, dtype=np.float32)
        for k in range(n + 1):
            acc += np.roll(base, k - n // 2, axis=1)
        out = Image.fromarray(np.clip(acc / (n + 1), 0, 255).astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/det")
    ap.add_argument("--spec", default=os.path.join(ROOT, "spec", "labels.txt"))
    ap.add_argument("--fonts", default=os.path.join(ROOT, "fonts"))
    ap.add_argument("--font", default="")
    ap.add_argument("--bg", default="", help="comma-separated dirs of background photos")
    ap.add_argument("--real-plates", dest="real_plates", default="",
                    help="comma-separated dirs of REAL plate photos to paste instead of drawn art")
    ap.add_argument("--real-pct", dest="real_pct", type=int, default=50)
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--meta-only", dest="meta_only", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    sp = L.load(a.spec)
    font_paths = G.font_files(a.fonts, a.font)
    if not font_paths:
        raise SystemExit("no fonts in %s" % a.fonts)
    bgs = bg_files(a.bg)
    if bgs and not a.quiet:
        print("background pool: %d files from %s" % (len(bgs), a.bg))
    reals = image_files(a.real_plates)
    if reals and not a.quiet:
        print("real plate pool: %d files from %s (%d%% of plates)" % (len(reals), a.real_plates, a.real_pct))

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    mode = "ab" if a.start > 0 else "wb"
    fm = open(os.path.join(a.out, "meta.txt"), mode)
    fc = open(os.path.join(a.out, "corners.txt"), mode)

    kept_total = neg_total = 0
    for i in range(a.start, a.start + a.count):
        rng = Rng((a.seed ^ ((i * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)) ^ DET_SALT) & ((1 << 64) - 1))
        d = sample_det(rng, sp, len(bgs), len(font_paths), len(reals), a.real_pct)
        place_det(d, a.imgsz)
        name = "det%06d.png" % i
        bg_name = os.path.basename(bgs[d.bg_idx]) if (d.bg_real and bgs) else "synth"
        fm.write(det_meta(name, d, sp, bg_name, a.imgsz).encode("utf-8"))
        fc.write(det_corners(name, d, a.imgsz).encode("utf-8"))
        kept_total += sum(1 for pl in d.plates if pl.keep)
        neg_total += 1 if d.n_plates == 0 else 0
        if a.meta_only:
            continue
        img = render_det(d, sp, font_paths, bgs, a.imgsz, rng, reals)
        img.save(os.path.join(a.out, "images", name))
        # binary, so the newlines stay LF and match the C++ writer byte for byte on Windows too
        with open(os.path.join(a.out, "labels", name[:-4] + ".txt"), "wb") as lf:
            lf.write(det_labels(d, a.imgsz).encode("utf-8"))
    fm.close()
    fc.close()
    print("%s %d frames into %s (imgsz=%d, %d plates kept, %d empty frames, seed=%d)"
          % ("meta for" if a.meta_only else "wrote", a.count, a.out, a.imgsz, kept_total,
             neg_total, a.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
