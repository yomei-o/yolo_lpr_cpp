"""Synthetic plate generator — Python side. Same draws, same labels, same geometry as
pure/gen.hpp + pure/gen_render.hpp; see spec/gen.md for the draw order that both must follow.

  python tools/gen.py --out data/synth --count 1000 [--seed 12345] [--start 0] [--out-px 192]
                      [--fonts fonts] [--spec spec/labels.txt] [--meta-only] [--quiet]

`--meta-only` writes labels.txt + meta.txt without rendering; tools/parity/gen.py diffs those
against the C++ generator's, which is where "same generator in both languages" is actually proven.
Pixels are NOT expected to match (PIL/FreeType vs stb_truetype rasterise differently).
"""
import argparse
import io as _io
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import labels as L      # noqa: E402
from rng import Rng     # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PI = 3.14159265358979323846
FONT_CANDIDATES = ["GenSenRounded2-B.ttc", "YuGothB.ttc", "meiryo.ttc", "msgothic.ttc"]

PRIVATE_NORMAL, COMMERCIAL_NORMAL, KEI_PRIVATE, KEI_COMMERCIAL, LARGE, DESIGN, OTHER, UNKNOWN = range(8)
PALETTE = {
    COMMERCIAL_NORMAL: ((16, 90, 60), (250, 250, 248)),
    KEI_PRIVATE: ((240, 205, 20), (25, 25, 25)),
    KEI_COMMERCIAL: ((25, 25, 25), (240, 205, 20)),
}
PALETTE_DEFAULT = ((250, 250, 248), (16, 90, 60))


def kind_from_draw(d):
    if d < 50:
        return PRIVATE_NORMAL
    if d < 65:
        return KEI_PRIVATE
    if d < 75:
        return COMMERCIAL_NORMAL
    if d < 80:
        return KEI_COMMERCIAL
    if d < 90:
        return LARGE
    return DESIGN


class Params(object):
    pass


def sample(rng, sp):
    """Mirror of gen::sample — the rng call order is the contract (spec/gen.md)."""
    p = Params()
    reg = sp.head("region")
    cn2 = sp.head("class_num_02")

    p.kind = kind_from_draw(rng.below(100))                                   # 1
    p.region = rng.below(reg.n)                                              # 2
    cl = rng.below(100)                                                     # 3
    p.cls_len = 3 if cl < 80 else (2 if cl < 95 else 1)
    first = rng.below(9) + 1                                                # 4
    p.cls_idx = [first, -1, -1]
    p.cls_text = str(first)
    for i in range(1, p.cls_len):                                           # 5
        t = rng.below(cn2.n)
        p.cls_idx[i] = t
        p.cls_text += cn2.tok[t]
    if p.cls_idx[1] < 0:
        p.cls_idx[1] = 0
    if p.cls_idx[2] < 0:
        p.cls_idx[2] = sp.index_of("class_num_03", "<blank>")

    hset = sp.find("hira_commercial" if p.kind in (COMMERCIAL_NORMAL, KEI_COMMERCIAL) else "hira_private")
    hsel = rng.below(hset.n)                                                # 6
    p.hira = sp.index_of("hiragana", hset.tok[hsel])

    sl = rng.below(100)                                                     # 7
    p.serial_len = 4 if sl < 85 else (3 if sl < 95 else (2 if sl < 99 else 1))
    p.serial_idx = [10, 10, 10, 10]
    p.serial_text = ""
    for i in range(p.serial_len):                                           # 8
        d = rng.below(10)
        slot = 4 - p.serial_len + i
        if i == p.serial_len - 1 and d == 0 and p.serial_text.strip("0") == "":
            d = 1
        p.serial_idx[slot] = d
        p.serial_text += str(d)
    if p.serial_idx[3] == 10:
        p.serial_idx[3] = 0

    p.yaw = rng.range(-35, 35)                                              # 9
    p.pitch = rng.range(-20, 20)                                            # 10
    p.roll = rng.range(-8, 8)                                               # 11
    p.plate_px = rng.range(24, 200)                                         # 12
    p.margin = rng.range(0.02, 0.18)                                        # 13
    p.off_x = rng.range(-0.05, 0.05)                                        # 14
    p.off_y = rng.range(-0.05, 0.05)                                        # 15
    p.brightness = rng.range(0.45, 1.35)                                    # 16
    p.contrast = rng.range(0.7, 1.3)                                        # 17
    p.warm = rng.range(-0.12, 0.12)                                         # 18
    p.blur = rng.range(0, 1.6)                                              # 19
    p.motion = rng.range(0, 3.0)                                            # 20
    p.noise = rng.range(0, 12)                                              # 21
    p.jpeg_q = rng.below(60) + 40                                           # 22
    p.dirt = rng.range(0, 0.35)                                             # 23
    bd = rng.below(100)                                                     # 24
    p.bolts = 2 if bd < 80 else (0 if bd < 95 else 4)
    p.seal = rng.below(100) < 70                                            # 25
    p.frame = rng.below(100) >= 85                                          # 26
    p.bg_hue = rng.range(0, 1)                                              # 27
    p.bg_dark = rng.range(0.15, 0.85)                                       # 28

    p.large = p.kind == LARGE
    p.design = p.kind == DESIGN
    p.bg, p.fg = PALETTE.get(p.kind, PALETTE_DEFAULT)
    p.legible = not (p.plate_px < 40 or p.blur > 1.2 or p.motion > 2.2 or p.noise > 9 or p.dirt > 0.28)
    return p


def heads(p):
    return [p.region, p.cls_idx[0], p.cls_idx[1], p.cls_idx[2], p.hira,
            p.serial_idx[0], p.serial_idx[1], p.serial_idx[2], p.serial_idx[3],
            p.kind, 1 if p.legible else 0]


def labels_line(file, p):
    return file + "".join(" %d" % v for v in heads(p)) + "\n"


def meta_line(file, p, sp, font):
    kg = sp.head("plate_kind")
    return ("%s kind=%s region=%d cls=%s hira=%d serial=%s yaw=%.4f pitch=%.4f roll=%.4f "
            "plate_px=%.4f margin=%.4f off_x=%.4f off_y=%.4f brightness=%.4f contrast=%.4f "
            "warm=%.4f blur=%.4f motion=%.4f noise=%.4f jpeg_q=%d dirt=%.4f bolts=%d seal=%d "
            "frame=%d bg_hue=%.4f bg_dark=%.4f legible=%d font=%s\n"
            % (file, kg.tok[p.kind], p.region, p.cls_text, p.hira, p.serial_text, p.yaw, p.pitch,
               p.roll, p.plate_px, p.margin, p.off_x, p.off_y, p.brightness, p.contrast, p.warm,
               p.blur, p.motion, p.noise, p.jpeg_q, p.dirt, p.bolts, 1 if p.seal else 0,
               1 if p.frame else 0, p.bg_hue, p.bg_dark, 1 if p.legible else 0, font))


def font_files(font_dir):
    return [os.path.join(font_dir, c) for c in FONT_CANDIDATES
            if os.path.exists(os.path.join(font_dir, c))]


# ---- rendering (PIL) ------------------------------------------------------------------------
def _draw_text(d, font_path, text, cx, cy, px_h, colour, track=0.06, max_w=0):
    """Centre `text` on (cx, cy) with an embossed (raised) look, like gen_render.hpp does."""
    from PIL import ImageFont
    size = max(4, int(round(px_h)))
    f = ImageFont.truetype(font_path, size)
    spacing = int(round(size * track))
    widths = [d.textlength(ch, font=f) + spacing for ch in text]
    total = sum(widths)
    if max_w and total > max_w:
        size = max(4, int(round(size * max_w / total)))
        f = ImageFont.truetype(font_path, size)
        spacing = int(round(size * track))
        widths = [d.textlength(ch, font=f) + spacing for ch in text]
        total = sum(widths)
    x = cx - total / 2
    e = max(1, int(round(px_h * 0.035)))
    for ch, w in zip(text, widths):
        bbox = d.textbbox((0, 0), ch, font=f, anchor="lt")
        gx = x + (w - (bbox[2] - bbox[0])) / 2 - bbox[0]
        gy = cy - (bbox[3] + bbox[1]) / 2
        d.text((gx - e, gy - e), ch, font=f, fill=(255, 255, 255), anchor="lt")
        d.text((gx + e, gy + e), ch, font=f, fill=(0, 0, 0), anchor="lt")
        d.text((gx, gy), ch, font=f, fill=colour, anchor="lt")
        x += w
    return total


def plate_texture(p, sp, font_path, rng):
    from PIL import Image, ImageDraw
    mm = 2.0
    W_mm, H_mm = (440.0, 220.0) if p.large else (330.0, 165.0)
    W, H = int(W_mm * mm), int(H_mm * mm)
    r = (13.0 if p.large else 10.0) * mm
    im = Image.new("RGB", (W, H), (90, 90, 90))
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle([2, 2, W - 3, H - 3], radius=r, fill=tuple(p.bg))
    if p.design:
        for k in range(6):
            cx, cy = rng.range(0.1, 0.9) * W, rng.range(0.15, 0.85) * H
            rr = rng.range(0.05, 0.16) * W
            c = (int(rng.range(120, 220)), int(rng.range(160, 230)), int(rng.range(150, 210)), 120)
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=c)
        d.rounded_rectangle([2, 2, W - 3, H - 3], radius=r, fill=(255, 255, 255, 140))

    sx = (440.0 / 330.0) if p.large else 1.0
    sy = (220.0 / 165.0) if p.large else 1.0
    X = lambda v: v * mm * sx          # noqa: E731
    Y = lambda v: v * mm * sy          # noqa: E731

    if p.frame:
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=r, fill=(40, 40, 45))
        d.rounded_rectangle([Y(5), Y(5), W - 1 - Y(5), H - 1 - Y(5)], radius=r, fill=tuple(p.bg))

    reg = sp.head("region")
    hira = sp.head("hiragana")
    _draw_text(d, font_path, reg.tok[p.region], X(95), Y(40), Y(40), tuple(p.fg), 0.02, X(150))
    _draw_text(d, font_path, p.cls_text, X(245), Y(40), Y(42), tuple(p.fg), 0.08, X(140))
    _draw_text(d, font_path, hira.tok[p.hira], X(40), Y(118), Y(58), tuple(p.fg), 0.0)
    s4 = p.serial_text
    while len(s4) < 4:
        s4 = "・" + s4
    _draw_text(d, font_path, s4[:2], X(140), Y(118), Y(86), tuple(p.fg), 0.10)
    d.rounded_rectangle([X(196), Y(115), X(214), Y(122)], radius=1, fill=tuple(p.fg))
    _draw_text(d, font_path, s4[2:], X(270), Y(118), Y(86), tuple(p.fg), 0.10)

    def circle(cx, cy, rr, col):
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=col)

    if p.bolts >= 2:
        circle(X(60), Y(18), Y(7), (160, 160, 165))
        circle(X(270), Y(18), Y(7), (160, 160, 165))
    if p.bolts == 4:
        circle(X(60), Y(150), Y(7), (160, 160, 165))
        circle(X(270), Y(150), Y(7), (160, 160, 165))
    if p.seal:
        circle(X(60), Y(18), Y(9), (70, 110, 90))
        circle(X(60), Y(18), Y(5), (200, 210, 200))

    a = np.asarray(im, dtype=np.float32)
    # gloss band
    gx0, gw = rng.range(-0.3, 0.6) * W, rng.range(0.25, 0.7) * W
    strength = rng.range(0.05, 0.22)
    yy, xx = np.mgrid[0:H, 0:W]
    t = (xx + 0.6 * yy - gx0) / gw
    band = np.where((t >= 0) & (t <= 1), strength * np.sin(np.clip(t, 0, 1) * PI), 0)[..., None]
    a = a + (255.0 - a) * band
    # dirt
    if p.dirt > 0.01:
        for k in range(int(p.dirt * 40)):
            cx, cy = rng.range(0, 1) * W, rng.range(0, 1) * H
            rr = rng.range(0.01, 0.06) * W
            al = p.dirt * rng.range(0.15, 0.5)
            c = np.array([rng.range(60, 140), rng.range(60, 130), rng.range(50, 120)], dtype=np.float32)
            dd = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            m = np.clip(1.0 - dd / rr, 0, 1)[..., None] * al
            a = a * (1 - m) + c * m
    from PIL import Image as _Image
    return _Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def project_plate(p, cx, cy):
    """Same pinhole placement as gen_render.hpp::project_plate -> TL,TR,BR,BL."""
    W_mm, H_mm = (440.0, 220.0) if p.large else (330.0, 165.0)
    ry, rx, rz = p.yaw * PI / 180, p.pitch * PI / 180, p.roll * PI / 180
    hx, hy = W_mm / 2, H_mm / 2
    src = [(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0)]
    c3 = []
    for X, Y, Z in src:
        x1 = X * math.cos(ry) + Z * math.sin(ry)
        z1 = -X * math.sin(ry) + Z * math.cos(ry)
        y2 = Y * math.cos(rx) - z1 * math.sin(rx)
        z2 = Y * math.sin(rx) + z1 * math.cos(rx)
        x3 = x1 * math.cos(rz) - y2 * math.sin(rz)
        y3 = x1 * math.sin(rz) + y2 * math.cos(rz)
        c3.append((x3, y3, z2))
    dist = W_mm * 3.0
    # focal length chosen so the projected plate lands exactly plate_px wide (the C++ side does the
    # same in two passes, where only the first updates f)
    xs = [c[0] / (dist + c[2]) for c in c3]
    w_proj = max(abs(xs[1] - xs[0]), abs(xs[2] - xs[3]))
    f = p.plate_px / max(1e-6, w_proj)
    return [(cx + f * c[0] / (dist + c[2]), cy + f * c[1] / (dist + c[2])) for c in c3]


def render(p, sp, font_path, out_px, rng):
    from PIL import Image, ImageFilter
    tex = plate_texture(p, sp, font_path, rng)
    canvas = max(64, int(math.ceil(p.plate_px * 2.4)))
    q = project_plate(p, canvas * 0.5, canvas * 0.5)

    # background: hsv-ish base + soft blobs, then blur (matches the C++ intent, not its pixels)
    hue = p.bg_hue * 6.0
    i = int(hue)
    fr = hue - i
    v, s = p.bg_dark, 0.35
    pv, qv, tv = v * (1 - s), v * (1 - s * fr), v * (1 - s * (1 - fr))
    rgb = [(v, tv, pv), (qv, v, pv), (pv, v, tv), (pv, qv, v), (tv, pv, v), (v, pv, qv)][i % 6]
    yy, xx = np.mgrid[0:canvas, 0:canvas]
    shade = (0.75 + 0.5 * yy / canvas)[..., None]
    bg = np.array(rgb, dtype=np.float32) * 255.0 * shade
    for k in range(14):
        bx, by = rng.range(-0.1, 1.1) * canvas, rng.range(-0.1, 1.1) * canvas
        br = rng.range(0.08, 0.45) * canvas
        al = rng.range(0.05, 0.25)
        c = np.array([rng.range(0, 255), rng.range(0, 255), rng.range(0, 255)], dtype=np.float32)
        c = 0.6 * c + 0.4 * np.array(rgb, dtype=np.float32) * 255.0
        dd = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2)
        m = (np.clip(1 - dd / br, 0, 1) ** 2)[..., None] * al
        bg = bg * (1 - m) + c * m
    im = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(max(1.0, canvas * 0.02)))

    # paste the plate through the quad (PIL's QUAD transform maps the destination box to the quad)
    tw, th = tex.size
    coeffs = _persp_coeffs(q, tw, th)
    warped = tex.transform((canvas, canvas), Image.PERSPECTIVE, data=coeffs, resample=Image.BILINEAR)
    mask = Image.new("L", (tw, th), 255).transform((canvas, canvas), Image.PERSPECTIVE,
                                                   data=coeffs, resample=Image.BILINEAR)
    im.paste(warped, (0, 0), mask)

    a = np.asarray(im, dtype=np.float32)
    a = (a / 255.0 - 0.5) * p.contrast + 0.5
    a *= p.brightness
    a[:, :, 0] *= (1.0 + p.warm)
    a[:, :, 2] *= (1.0 - p.warm)
    a = np.clip(a * 255.0, 0, 255)
    if p.noise > 0.1:
        n = a.size
        u = rng.units(2 * n)                       # same stream as the C++ per-element loop
        u1 = np.maximum(1e-9, u[0::2])
        u2 = u[1::2]
        g = (np.sqrt(-2 * np.log(u1)) * np.cos(2 * PI * u2)).reshape(a.shape)
        a = np.clip(a + g * p.noise, 0, 255)
    im = Image.fromarray(a.astype(np.uint8))
    if p.blur >= 0.05:
        im = im.filter(ImageFilter.GaussianBlur(p.blur))
    if p.motion >= 1:
        n = int(round(p.motion))
        acc = np.zeros((canvas, canvas, 3), dtype=np.float32)
        base = np.asarray(im, dtype=np.float32)
        for k in range(n + 1):
            acc += np.roll(base, k - n // 2, axis=1)
        im = Image.fromarray(np.clip(acc / (n + 1), 0, 255).astype(np.uint8))

    xs = [pt[0] for pt in q]
    ys = [pt[1] for pt in q]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    x0 = min(xs) - bw * p.margin + bw * p.off_x
    y0 = min(ys) - bh * p.margin + bh * p.off_y
    x1 = max(xs) + bw * p.margin + bw * p.off_x
    y1 = max(ys) + bh * p.margin + bh * p.off_y
    crop = im.transform((out_px, out_px), Image.AFFINE,
                        data=((x1 - x0) / out_px, 0, x0, 0, (y1 - y0) / out_px, y0),
                        resample=Image.BILINEAR)
    corners = []
    for (px_, py_) in q:
        corners += [(px_ - x0) * out_px / (x1 - x0), (py_ - y0) * out_px / (y1 - y0)]

    if p.jpeg_q < 100:
        buf = _io.BytesIO()
        crop.save(buf, format="JPEG", quality=int(p.jpeg_q))
        buf.seek(0)
        from PIL import Image as _I
        crop = _I.open(buf).convert("RGB")
    return crop, corners


def _persp_coeffs(q, tw, th):
    """8 coefficients of the homography canvas(x,y) -> texture(u,v) — the same matrix
    gen_render.hpp::quad_to_texture_h solves. PIL's PERSPECTIVE transform wants exactly this
    (destination -> source), unlike QUAD which maps a source quad onto the whole output."""
    dst = [(0, 0), (tw, 0), (tw, th), (0, th)]
    A, b = [], []
    for (cx, cy), (ux, uy) in zip(q, dst):
        A.append([cx, cy, 1, 0, 0, 0, -cx * ux, -cy * ux])
        b.append(ux)
        A.append([0, 0, 0, cx, cy, 1, -cx * uy, -cy * uy])
        b.append(uy)
    h = np.linalg.solve(np.array(A, dtype=np.float64), np.array(b, dtype=np.float64))
    return tuple(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--spec", default=os.path.join(ROOT, "spec", "labels.txt"))
    ap.add_argument("--fonts", default=os.path.join(ROOT, "fonts"))
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out-px", dest="out_px", type=int, default=192)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--meta-only", dest="meta_only", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    sp = L.load(a.spec)
    font_paths = font_files(a.fonts)
    if not font_paths:
        raise SystemExit("no fonts in %s — run: python tools/fetch_fonts.py --include-system" % a.fonts)
    font_names = [os.path.basename(f) for f in font_paths]

    os.makedirs(a.out, exist_ok=True)
    mode = "ab" if a.start > 0 else "wb"
    fl = open(os.path.join(a.out, "labels.txt"), mode)
    fc = open(os.path.join(a.out, "corners.txt"), mode)
    fm = open(os.path.join(a.out, "meta.txt"), mode)

    for i in range(a.start, a.start + a.count):
        rng = Rng(a.seed ^ ((i * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)))
        p = sample(rng, sp)
        fi = rng.below(len(font_names))                                      # 29
        name = "plate%06d.png" % i
        fm.write(meta_line(name, p, sp, font_names[fi]).encode("utf-8"))
        fl.write(labels_line(name, p).encode("utf-8"))
        if a.meta_only:
            continue
        crop, corners = render(p, sp, font_paths[fi], a.out_px, rng)
        crop.save(os.path.join(a.out, name))
        fc.write((name + "".join(" %.2f" % v for v in corners) + " " + font_names[fi] + "\n").encode("utf-8"))
        if not a.quiet and i - a.start < 12:
            print("  %s  %-28s px=%.0f yaw=%.0f pitch=%.0f blur=%.2f legible=%d font=%s"
                  % (name, L.decode(sp, heads(p)).text, p.plate_px, p.yaw, p.pitch, p.blur,
                     1 if p.legible else 0, font_names[fi]))
    fl.close()
    fc.close()
    fm.close()
    print("%s %d samples into %s (out_px=%d, seed=%d)"
          % ("meta for" if a.meta_only else "wrote", a.count, a.out, a.out_px, a.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
