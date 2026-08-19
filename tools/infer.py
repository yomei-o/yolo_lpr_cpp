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

    def __init__(self, det_path=None, ocr_path=None, spec_path=None, det_kind="yolox"):
        self.spec = L.load(spec_path or os.path.join(ROOT, "spec", "labels.txt"))
        self.det_kind = det_kind
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

    def detect_v8(self, rgb, conf=0.25, nms=0.45, plate_class=0):
        """A [1,4+nc,N] head with the NMS tail stripped (tools/strip_nms.py) — the shape our own
        detector will have after M7. Boxes are xyxy in input pixels, scores already sigmoided;
        mirrors pure/infer_v8.hpp so the two implementations can be diffed."""
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
            cand.append([float(t[0, i]), float(t[1, i]), float(t[2, i]), float(t[3, i]),
                         float(bestp[i]), int(best[i])])
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
            boxes = self.detect_v8(rgb, conf)
        else:
            boxes = self.detect(rgb, imgsz, conf)
        margins = MARGINS if tta else [0.0]
        plates = []
        for b in boxes:
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


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="infer.py")
    ap.add_argument("--img", required=True)
    ap.add_argument("--det", default=os.path.join(ROOT, "models", "plate_det_pyj320.onnx"))
    ap.add_argument("--ocr", default=os.path.join(ROOT, "models", "plate_ocr.onnx"))
    ap.add_argument("--spec", default=os.path.join(ROOT, "spec", "labels.txt"))
    ap.add_argument("--out", default="")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--box", nargs=4, type=float, default=None)
    ap.add_argument("--det-kind", dest="det_kind", default="v8", choices=["yolox", "v8", "plateyolo"])
    a = ap.parse_args(argv)

    rgb = load_rgb(a.img)
    pipe = Pipeline(a.det, a.ocr, a.spec, a.det_kind)
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
        from PIL import Image, ImageDraw
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        for p in res["plates"]:
            d.rectangle(p["box"], outline=(255, 60, 60), width=3)
        im.save(a.out)
    return 0 if res["plates"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
