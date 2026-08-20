"""Python side of the inference pipeline — the same three stages, the same ONNX files, the same
decode as pure/pipeline.hpp. Runs the graphs with onnxruntime; the C++ side runs them with its own
interpreter, so `tools/parity/infer.py` comparing the two JSON outputs is a real cross-check of
both implementations (not a tautology).

Preprocessing is transcribed from pure/crop.hpp deliberately, formula for formula:
  detector  : letterbox to SxS, BGR, 0-255, pad 114, image pinned top-left
  recognizer: box (+margin) -> 128x128 RGB /255 NCHW
Get one of these wrong and everything still "works" while reading the wrong plate.
"""
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labels as L  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- detector config (mirrors jl::DetCfg) ---------------------------------------------------
DET_HEADS = ["/head/Concat_output_0", "/head/Concat_1_output_0", "/head/Concat_2_output_0"]
DET_STRIDES = [8, 16, 32]
DET_NC = 8
PLATE_CLASS = 7
MARGINS = [-0.06, -0.03, 0.0, 0.03, 0.06, 0.10]

# stage 2 (M6): the corner regressor's framing, mirroring pure/warp.hpp + tools/corner_model.py
CORNER_IN_PX = 64
CORNER_EXPAND = 0.25
WARP_MARGIN = 0.06


def load_rgb(path):
    """Decode to an HxWx3 uint8 array. Uses the same pixels the CLI sees for .rgba fixtures."""
    if path.endswith(".rgba"):
        raw = open(path, "rb").read()
        w = int.from_bytes(raw[0:4], "little", signed=True)
        h = int.from_bytes(raw[4:8], "little", signed=True)
        a = np.frombuffer(raw[8:], dtype=np.uint8).reshape(h, w, 4)
        return np.ascontiguousarray(a[:, :, :3])
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _sample(rgb, xs, ys):
    """Bilinear sample with edge clamp; xs/ys are float arrays of source coordinates."""
    H, W, _ = rgb.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = (xs - x0).astype(np.float32)[..., None]
    fy = (ys - y0).astype(np.float32)[..., None]

    def px(yy, xx):
        return rgb[np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)].astype(np.float32)

    return (px(y0, x0) * (1 - fx) * (1 - fy) + px(y0, x0 + 1) * fx * (1 - fy)
            + px(y0 + 1, x0) * (1 - fx) * fy + px(y0 + 1, x0 + 1) * fx * fy)


def letterbox_bgr(rgb, S):
    """-> (1,3,S,S) float32 BGR 0-255, pad 114, top-left; and the scale used."""
    H, W, _ = rgb.shape
    scale = min(S / W, S / H)
    nw, nh = int(W * scale), int(H * scale)
    out = np.full((S, S, 3), 114.0, dtype=np.float32)
    xs = (np.arange(nw, dtype=np.float32) + 0.5) / scale - 0.5
    ys = (np.arange(nh, dtype=np.float32) + 0.5) / scale - 0.5
    out[:nh, :nw] = _sample(rgb, xs[None, :].repeat(nh, 0), ys[:, None].repeat(nw, 1))
    bgr = out[:, :, ::-1]
    return np.ascontiguousarray(bgr.transpose(2, 0, 1)[None]), scale


def crop_to_input(rgb, bx0, by0, bx1, by1, margin):
    """-> (1,3,128,128) float32 RGB /255 over the box expanded by `margin`."""
    bw, bh = bx1 - bx0, by1 - by0
    x0, y0 = bx0 - bw * margin, by0 - bh * margin
    x1, y1 = bx1 + bw * margin, by1 + bh * margin
    g = np.arange(128, dtype=np.float32) + 0.5
    xs = x0 + g * (x1 - x0) / 128.0 - 0.5
    ys = y0 + g * (y1 - y0) / 128.0 - 0.5
    v = _sample(rgb, xs[None, :].repeat(128, 0), ys[:, None].repeat(128, 1)) / np.float32(255.0)
    return np.ascontiguousarray(v.transpose(2, 0, 1)[None])


def solve_h_dst_to_src(dst, src):
    """Homography mapping destination -> source, from 4 point pairs (same as pure/warp.hpp)."""
    A, b = [], []
    for (dx, dy), (sx, sy) in zip(dst, src):
        A.append([dx, dy, 1, 0, 0, 0, -dx * sx, -dy * sx])
        b.append(sx)
        A.append([0, 0, 0, dx, dy, 1, -dx * sy, -dy * sy])
        b.append(sy)
    h = np.linalg.solve(np.array(A, dtype=np.float64), np.array(b, dtype=np.float64))
    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])


def warp_to_input(rgb, corners, margin=WARP_MARGIN, out_px=128):
    """Rectify the plate quad (TL,TR,BR,BL in image pixels) into out_px x out_px RGB /255 NCHW."""
    lo = margin / (1.0 + 2 * margin) * out_px
    hi = (1.0 + margin) / (1.0 + 2 * margin) * out_px
    dst = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    src = [(corners[0], corners[1]), (corners[2], corners[3]),
           (corners[4], corners[5]), (corners[6], corners[7])]
    H = solve_h_dst_to_src(dst, src)
    g = np.arange(out_px, dtype=np.float64) + 0.5
    xx, yy = np.meshgrid(g, g)
    w = H[2, 0] * xx + H[2, 1] * yy + H[2, 2]
    u = (H[0, 0] * xx + H[0, 1] * yy + H[0, 2]) / w - 0.5
    v = (H[1, 0] * xx + H[1, 1] * yy + H[1, 2]) / w - 0.5
    out = _sample(rgb, u.astype(np.float32), v.astype(np.float32)) / np.float32(255.0)
    return np.ascontiguousarray(out.transpose(2, 0, 1)[None])


def corner_input(rgb, box, expand=CORNER_EXPAND, in_px=CORNER_IN_PX):
    bw, bh = box[2] - box[0], box[3] - box[1]
    cx0, cy0 = box[0] - bw * expand, box[1] - bh * expand
    cx1, cy1 = box[2] + bw * expand, box[3] + bh * expand
    g = np.arange(in_px, dtype=np.float32) + 0.5
    sx = cx0 + g * (cx1 - cx0) / in_px - 0.5
    sy = cy0 + g * (cy1 - cy0) / in_px - 0.5
    v = _sample(rgb, sx[None, :].repeat(in_px, 0), sy[:, None].repeat(in_px, 1)) / np.float32(255.0)
    return np.ascontiguousarray(v.transpose(2, 0, 1)[None]), (cx0, cy0, cx1, cy1)


# ---- onnxruntime sessions --------------------------------------------------------------------
def _session(path, extra_outputs=()):
    import onnxruntime as ort
    if extra_outputs:
        import onnx
        m = onnx.load(path)
        have = {o.name for o in m.graph.output}
        for name in extra_outputs:
            if name not in have:
                m.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
        blob = m.SerializeToString()
    else:
        blob = open(path, "rb").read()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"])


class Pipeline:
    """det_kind selects the detector graph:
       'yolox'     — the interim ReLU plate yolox-tiny (letterbox 416 BGR 0-255, heads decoded here).
                     This is what the C++ side runs, so it is the default for parity.
       'plateyolo' — Kazuhito00/PlateYOLO-JP (YOLO12, AGPL-3.0): resize to NxN, RGB /255, and the
                     graph already contains the decode AND NonMaxSuppression, so it outputs
                     [1,300,6] = x1,y1,x2,y2,score,cls directly. Those ops (NMS/GatherND/ScatterND/
                     NonZero) are far outside what the pure C++ interpreter implements, which is
                     exactly why this backend is Python-only: it is the *teacher* for auto-labelling
                     detection data (M7), not the shipped detector.
    """

    def __init__(self, det_path=None, ocr_path=None, spec_path=None, det_kind="yolox", v8_fmt="xyxy",
                 corner_path=None):
        self.spec = L.load(spec_path or os.path.join(ROOT, "spec", "labels.txt"))
        self.corner = _session(corner_path) if corner_path else None
        self.det_kind = det_kind
        self.v8_fmt = v8_fmt
        self.det = _session(det_path, DET_HEADS if det_kind == "yolox" else ()) if det_path else None
        self.ocr = _session(ocr_path) if ocr_path else None
        self.ocr_heads = [o.name for o in self.ocr.get_outputs()] if self.ocr else []

    def detect_plateyolo(self, rgb, conf=0.3):
        """PlateYOLO-JP: plain resize (no letterbox), RGB /255, NMS inside the graph."""
        H, W, _ = rgb.shape
        inp = self.det.get_inputs()[0]
        _, _, ih, iw = inp.shape
        g = np.arange(iw, dtype=np.float32) + 0.5
        xs = g * (W / iw) - 0.5
        ys = (np.arange(ih, dtype=np.float32) + 0.5) * (H / ih) - 0.5
        x = _sample(rgb, xs[None, :].repeat(ih, 0), ys[:, None].repeat(iw, 1)) / np.float32(255.0)
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        out = self.det.run(None, {inp.name: x})[0][0]         # (300, 6)
        boxes = []
        for d in out:
            if float(d[4]) < conf:
                continue
            boxes.append([float(d[0]) * W / iw, float(d[1]) * H / ih,
                          float(d[2]) * W / iw, float(d[3]) * H / ih, float(d[4])])
        boxes.sort(key=lambda b: -b[4])
        return boxes

    def detect_v8(self, rgb, conf=0.25, nms=0.45, plate_class=0, fmt="xyxy"):
        """A [1,4+nc,N] head with the NMS tail stripped (tools/strip_nms.py) — the shape our own
        detector will have after M7. Scores are already sigmoided; the 4 box numbers are in input
        pixels and their layout depends on the export: PlateYOLO cut before its NMS gives **xyxy**,
        a plain Ultralytics export gives **cxcywh**. Mirrors pure/infer_v8.hpp (BoxFmt)."""
        inp = self.det.get_inputs()[0]
        _, _, ih, iw = inp.shape
        xs = (np.arange(iw, dtype=np.float32) + 0.5) * (rgb.shape[1] / iw) - 0.5
        ys = (np.arange(ih, dtype=np.float32) + 0.5) * (rgb.shape[0] / ih) - 0.5
        x = _sample(rgb, xs[None, :].repeat(ih, 0), ys[:, None].repeat(iw, 1)) / np.float32(255.0)
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        t = self.det.run(None, {inp.name: x})[0][0]            # (4+nc, N)
        nc = t.shape[0] - 4
        cls = t[4:]
        best = np.argmax(cls, axis=0)
        bestp = np.max(cls, axis=0)
        cand = []
        for i in np.nonzero(bestp >= conf)[0]:
            p0, p1, p2, p3 = (float(t[0, i]), float(t[1, i]), float(t[2, i]), float(t[3, i]))
            if fmt == "cxcywh":
                p0, p1, p2, p3 = p0 - p2 / 2, p1 - p3 / 2, p0 + p2 / 2, p1 + p3 / 2
            cand.append([p0, p1, p2, p3, float(bestp[i]), int(best[i])])
        cand.sort(key=lambda d: -d[4])
        keep, dead = [], [False] * len(cand)

        def iou(a, b):
            iw_ = min(a[2], b[2]) - max(a[0], b[0])
            ih_ = min(a[3], b[3]) - max(a[1], b[1])
            if iw_ <= 0 or ih_ <= 0:
                return 0.0
            inter = iw_ * ih_
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / ua

        for a in range(len(cand)):
            if dead[a]:
                continue
            keep.append(cand[a])
            for b in range(a + 1, len(cand)):
                if not dead[b] and cand[b][5] == cand[a][5] and iou(cand[a], cand[b]) > nms:
                    dead[b] = True
        H, W, _ = rgb.shape
        sx, sy = W / iw, H / ih
        boxes = []
        for d in keep:
            if d[5] != plate_class:
                continue
            boxes.append([min(max(d[0] * sx, 0), W - 1), min(max(d[1] * sy, 0), H - 1),
                          min(max(d[2] * sx, 0), W - 1), min(max(d[3] * sy, 0), H - 1), d[4]])
        boxes.sort(key=lambda b: -b[4])
        return boxes

    # -- stage 1 ------------------------------------------------------------------------------
    def detect(self, rgb, imgsz=416, conf=0.15, nms=0.45):
        x, scale = letterbox_bgr(rgb, imgsz)
        inp = self.det.get_inputs()[0].name
        outs = self.det.run(DET_HEADS, {inp: x})
        cand = []
        for lvl, t in enumerate(outs):
            t = t[0]                                   # (4+1+nc, h, w)
            s = float(DET_STRIDES[lvl])
            _, hh, ww = t.shape
            obj = t[4]                                 # ONNX head: already sigmoided
            cls = t[5:5 + DET_NC]
            best = np.argmax(cls, axis=0)
            bestp = np.max(cls, axis=0)
            score = obj * bestp
            ii, jj = np.nonzero(score >= conf)
            for i, j in zip(ii, jj):
                cx = (t[0, i, j] + j) * s
                cy = (t[1, i, j] + i) * s
                w = float(np.exp(t[2, i, j])) * s
                h = float(np.exp(t[3, i, j])) * s
                cand.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
                             float(score[i, j]), int(best[i, j])])
        cand.sort(key=lambda d: -d[4])
        keep, dead = [], [False] * len(cand)

        def iou(a, b):
            iw = min(a[2], b[2]) - max(a[0], b[0])
            ih = min(a[3], b[3]) - max(a[1], b[1])
            if iw <= 0 or ih <= 0:
                return 0.0
            inter = iw * ih
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / ua

        for a in range(len(cand)):
            if dead[a]:
                continue
            keep.append(cand[a])
            for b in range(a + 1, len(cand)):
                if not dead[b] and cand[b][5] == cand[a][5] and iou(cand[a], cand[b]) > nms:
                    dead[b] = True
        H, W, _ = rgb.shape
        boxes = []
        for d in keep:
            if d[5] != PLATE_CLASS:
                continue
            boxes.append([min(max(d[0] / scale, 0), W - 1), min(max(d[1] / scale, 0), H - 1),
                          min(max(d[2] / scale, 0), W - 1), min(max(d[3] / scale, 0), H - 1), d[4]])
        boxes.sort(key=lambda b: -b[4])
        return boxes

    # -- stage 2: corners -> rectified crop ---------------------------------------------------
    def predict_corners(self, rgb, box):
        x, win = corner_input(rgb, box)
        out = self.corner.run(None, {self.corner.get_inputs()[0].name: x})[0].reshape(-1)
        if out.size < 8:
            return None
        c = []
        for i in range(4):
            c.append(float(win[0] + out[2 * i] * (win[2] - win[0])))
            c.append(float(win[1] + out[2 * i + 1] * (win[3] - win[1])))
        return c

    def read_warped(self, rgb, corners):
        """One forward pass on the rectified crop — no TTA, because the framing is now fixed."""
        x = warp_to_input(rgb, corners)
        outs = self.ocr.run(self.ocr_heads, {self.ocr.get_inputs()[0].name: x})
        outs = [o.reshape(-1).astype(np.float64) for o in outs]
        arg = [int(np.argmax(t)) for t in outs]
        conf = [float(t[a] / t.sum()) if t.sum() > 0 else 0.0 for t, a in zip(outs, arg)]
        return arg, conf, 1

    # -- stage 3 ------------------------------------------------------------------------------
    def read(self, rgb, box, margins=MARGINS):
        inp = self.ocr.get_inputs()[0].name
        total = None
        for m in margins:
            x = crop_to_input(rgb, box[0], box[1], box[2], box[3], m)
            outs = self.ocr.run(self.ocr_heads, {inp: x})
            outs = [o.reshape(-1).astype(np.float64) for o in outs]
            total = outs if total is None else [a + b for a, b in zip(total, outs)]
        arg = [int(np.argmax(t)) for t in total]
        conf = [float(t[a] / t.sum()) if t.sum() > 0 else 0.0 for t, a in zip(total, arg)]
        return arg, conf, len(margins)

    # -- both ---------------------------------------------------------------------------------
    def run(self, rgb, imgsz=416, conf=0.15, tta=True, box=None):
        if box:
            boxes = [list(box) + [1.0]]
        elif self.det_kind == "plateyolo":
            boxes = self.detect_plateyolo(rgb, conf)
        elif self.det_kind == "v8":
            boxes = self.detect_v8(rgb, conf, fmt=self.v8_fmt)
        else:
            boxes = self.detect(rgb, imgsz, conf)
        margins = MARGINS if tta else [0.0]
        plates = []
        for b in boxes:
            if self.corner is not None:
                corners = self.predict_corners(rgb, b)
                arg, cf, crops = self.read_warped(rgb, corners) if corners else self.read(rgb, b, margins)
            else:
                arg, cf, crops = self.read(rgb, b, margins)
            p = L.decode(self.spec, arg)
            plates.append({"box": [round(v, 1) for v in b[:4]], "det": round(b[4], 3),
                           "crops": crops, "text": p.text, "region": p.region, "cls": p.cls,
                           "hira": p.hira, "num": p.disp, "kind": p.kind,
                           "arg": arg, "conf": [round(c, 4) for c in cf]})
        return {"plates": plates}


def format_json(res):
    """Same textual shape as jl::plates_json (C++), so the two can be diffed as text if wanted."""
    return json.dumps(res, ensure_ascii=False, separators=(",", ":"))


def annotate(rgb, plates, font_dir=None):
    """Draw the boxes and the decoded plate text onto a copy of the frame.

    The text is the point: a box alone cannot show whether the reading was right, and the README needs
    pictures that a reader can check against the photo. Japanese needs a real font, so one is taken
    from ./fonts (tools/fetch_fonts.py puts them there)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(im)
    size = max(16, im.width // 26)
    font = None
    for cand in ("GenSenRounded2-B.ttc", "NotoSansJP-Bold.ttf", "DroidSansFallbackFull.ttf",
                 "meiryo.ttc", "YuGothB.ttc", "msgothic.ttc"):
        path = os.path.join(font_dir or os.path.join(ROOT, "fonts"), cand)
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                pass
    for p in plates:
        x1, y1, x2, y2 = p["box"]
        w = max(3, im.width // 320)
        d.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=w)
        label = "%s  %.2f" % (p["text"], p["det"])
        if font is not None:
            tb = d.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = max(0, y1 - th - 10)
            d.rectangle([x1, ty, x1 + tw + 12, ty + th + 10], fill=(0, 0, 0))
            d.text((x1 + 6, ty + 4), label, font=font, fill=(255, 235, 120))
        else:
            d.text((x1 + 4, max(0, y1 - 14)), label, fill=(255, 235, 120))
    return im


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="infer.py")
    ap.add_argument("--img", required=True)
    ap.add_argument("--det", default=os.path.join(ROOT, "models", "plate_det_pyj320.onnx"))
    ap.add_argument("--ocr", default=os.path.join(ROOT, "models", "plate_ocr_v7_bal.onnx"))
    ap.add_argument("--spec", default=os.path.join(ROOT, "spec", "labels.txt"))
    ap.add_argument("--out", default="")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--corner", default="", help="corner regressor ONNX (M6); enables rectification")
    ap.add_argument("--box", nargs=4, type=float, default=None)
    ap.add_argument("--det-kind", dest="det_kind", default="v8", choices=["yolox", "v8", "plateyolo"])
    ap.add_argument("--fmt", default="xyxy", choices=["xyxy", "cxcywh"],
                    help="v8 box layout: PlateYOLO(NMS stripped)=xyxy, plain Ultralytics export=cxcywh")
    a = ap.parse_args(argv)

    rgb = load_rgb(a.img)
    pipe = Pipeline(a.det, a.ocr, a.spec, a.det_kind, a.fmt, corner_path=(a.corner or None))
    res = pipe.run(rgb, a.imgsz, a.conf, not a.single, a.box)

    if a.json:
        sys.stdout.buffer.write((format_json(res) + "\n").encode("utf-8"))
    else:
        out = io.StringIO()
        out.write("%s %dx%d   det=%s ocr=%s\n" % (a.img, rgb.shape[1], rgb.shape[0], a.det, a.ocr))
        out.write("plates: %d (conf>=%.2f)\n" % (len(res["plates"]), a.conf))
        for i, p in enumerate(res["plates"]):
            b = p["box"]
            out.write("  [%d] box (%.0f,%.0f)-(%.0f,%.0f) det %.2f  crops %d\n"
                      % (i, b[0], b[1], b[2], b[3], p["det"], p["crops"]))
            out.write("       %s\n" % p["text"])
            out.write("       conf" + "".join(" %.2f" % c for c in p["conf"]) + "\n")
        sys.stdout.buffer.write(out.getvalue().encode("utf-8"))

    if a.out:
        annotate(rgb, res["plates"]).save(a.out)
        print("wrote %s" % a.out)
    return 0 if res["plates"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
