"""Evaluate recognizers on real plate crops — the only way to choose between models honestly.

Dataset: dyama/alpr_jp (MIT), whose layout gives free region labels:
    <root>/自家用/<地名>/*.png      自家用(軽) / 事業用 / 事業用(軽) likewise
Only the 地域名 is known from the folder, so that is what this measures (it is also the hard head:
133-139 classes decided by a few small glyphs).

Models are compared by **token**, not by class index, because two recognizers trained by different
people order their classes differently even when the label sets are identical. Each model carries
its own spec file (spec/labels.txt, spec/ekmixer_labels.txt) and we compare the decoded strings.

  python tools/eval_ocr.py --data <alpr_jp root> [--limit 200] [--single] [--models ours,ekmixer]
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import labels as L      # noqa: E402
import infer as I       # noqa: E402
from rng import Rng     # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # Japanese labels on a cp932 console

CATEGORIES = ["自家用", "自家用(軽)", "事業用", "事業用(軽)"]


class Recognizer:
    """A 9/11-head plate classifier behind an ONNX file plus its own label spec."""

    def __init__(self, onnx_path, spec_path, softmax):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(open(onnx_path, "rb").read(), so,
                                         providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.spec = L.load(spec_path)
        self.heads = self.spec.of_kind("head")
        self.softmax = softmax                      # EkMixer emits logits; ours emits probabilities
        if len(self.heads) < len(self.out_names):
            raise SystemExit("spec %s has %d heads but the model has %d outputs"
                             % (spec_path, len(self.heads), len(self.out_names)))

    def _probs(self, x):
        outs = self.sess.run(self.out_names, {self.inp: x})
        outs = [o.reshape(-1).astype(np.float64) for o in outs]
        if self.softmax:
            outs = [np.exp(o - o.max()) / np.exp(o - o.max()).sum() for o in outs]
        return outs

    def read(self, rgb, box, margins):
        total = None
        for m in margins:
            x = I.crop_to_input(rgb, box[0], box[1], box[2], box[3], m)
            p = self._probs(x)
            total = p if total is None else [a + b for a, b in zip(total, p)]
        out = {}
        for h, t in zip(self.heads, total):
            order = np.argsort(-t)
            out[h.name] = {
                "top": [h.tok[i] for i in order[:3]],
                "conf": float(t[order[0]] / t.sum()),
            }
        return out


def scan(root, limit=0):
    items = []
    for cat in CATEGORIES:
        d = os.path.join(root, cat)
        if not os.path.isdir(d):
            continue
        for region in sorted(os.listdir(d)):
            rd = os.path.join(d, region)
            if not os.path.isdir(rd):
                continue                      # files directly under the category have no label
            for f in sorted(os.listdir(rd)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    items.append((os.path.join(rd, f), region, cat))
    items.sort()
    if limit and limit < len(items):
        # deterministic sample across all categories/regions (shared splitmix64, seed 7)
        rng = Rng(7)
        idx = list(range(len(items)))
        for i in range(len(idx) - 1, 0, -1):
            j = rng.below(i + 1)
            idx[i], idx[j] = idx[j], idx[i]
        items = [items[i] for i in sorted(idx[:limit])]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="alpr_jp checkout root")
    ap.add_argument("--models", default="ours,ekmixer")
    ap.add_argument("--ours", default=os.path.join(ROOT, "models", "plate_ocr.onnx"))
    ap.add_argument("--ekmixer", default=os.path.join(ROOT, "..", "_pyj", "weight", "EkMixer-128x128.onnx"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--single", action="store_true", help="one crop instead of the margin spread")
    args = ap.parse_args()

    margins = [0.0] if args.single else I.MARGINS
    items = scan(args.data, args.limit)
    if not items:
        raise SystemExit("no labelled crops under %s (expected 自家用/<地名>/*.png)" % args.data)

    recs = {}
    if "ours" in args.models:
        recs["ours"] = Recognizer(args.ours, os.path.join(ROOT, "spec", "labels.txt"), softmax=False)
    if "ekmixer" in args.models:
        if not os.path.exists(args.ekmixer):
            print("skipping ekmixer: %s not found" % args.ekmixer)
        else:
            recs["ekmixer"] = Recognizer(args.ekmixer,
                                         os.path.join(ROOT, "spec", "ekmixer_labels.txt"), softmax=True)

    stats = {k: {"n": 0, "top1": 0, "top3": 0, "conf": 0.0, "conf_right": 0.0,
                 "by_cat": {}} for k in recs}
    agree = both_right = 0
    t0 = time.time()
    for n, (path, region, cat) in enumerate(items, 1):
        rgb = I.load_rgb(path)
        H, W, _ = rgb.shape
        box = (0.0, 0.0, float(W), float(H))       # the file *is* the plate crop
        preds = {}
        for name, r in recs.items():
            out = r.read(rgb, box, margins)["region"]
            preds[name] = out["top"][0]
            s = stats[name]
            s["n"] += 1
            s["top1"] += int(out["top"][0] == region)
            s["top3"] += int(region in out["top"])
            s["conf"] += out["conf"]
            if out["top"][0] == region:
                s["conf_right"] += out["conf"]
            c = s["by_cat"].setdefault(cat, [0, 0])
            c[0] += 1
            c[1] += int(out["top"][0] == region)
        if len(preds) == 2:
            a, b = list(preds.values())
            agree += int(a == b)
            both_right += int(a == b == region)
        if n % 100 == 0:
            print("  %d/%d  %.0fs" % (n, len(items), time.time() - t0), flush=True)

    print("\n%d crops, %s, %.0fs" % (len(items), "1 crop" if args.single else
                                     "%d-crop TTA" % len(margins), time.time() - t0))
    for name, s in stats.items():
        print("%-8s region top1 %5.1f%%   top3 %5.1f%%   mean conf %.3f (correct only %.3f)"
              % (name, 100.0 * s["top1"] / s["n"], 100.0 * s["top3"] / s["n"],
                 s["conf"] / s["n"], s["conf_right"] / max(1, s["top1"])))
        for cat, (tot, ok) in sorted(s["by_cat"].items()):
            print("           %-12s %4d crops  top1 %5.1f%%" % (cat, tot, 100.0 * ok / tot))
    if len(recs) == 2:
        print("agreement between the two models: %.1f%%   both correct: %.1f%%"
              % (100.0 * agree / len(items), 100.0 * both_right / len(items)))
        print("(disagreement is where a human label actually buys something — see RESUME 疑似ラベル)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
