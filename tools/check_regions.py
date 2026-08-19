"""Can the recogniser actually name every region on the list? (M8 acceptance test)

The region head has 138 classes but the real data (alpr_jp) only contains 63-66 of the names, and
synthetic crops normally have the region loss masked out. That combination left the 2025 additions
(十勝 / 日光 / 江戸川 / 安曇野 / 南信州) at their -10 initialisation: measured probability 4e-11, i.e.
*structurally* unpredictable however clear the photo is. This script is the check for that: it renders
one clean-ish plate per region name and asks the model to read it back.

  python tools/gen.py --out data/region_sweep --count 414 --region sweep --seed 4242
  python tools/check_regions.py --data data/region_sweep --ocr models/plate_ocr_v3.onnx

A synthetic render is an easy test (the glyphs come from the same font list the training data used),
so treat the numbers as "is this class reachable at all", not as real-world accuracy. What matters:
NEW (the five 2025 names) must be readable, and the accuracy on names that real photos do cover must
not have collapsed — teaching regions from synthetic used to cost 11 points there (78% -> 67%).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import infer as I           # noqa: E402
import labels as L          # noqa: E402
import eval_ocr as E        # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NEW_2025 = ["十勝", "日光", "江戸川", "安曇野", "南信州"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="a sweep set from `--region sweep`")
    ap.add_argument("--ocr", default=os.path.join(ROOT, "models", "plate_ocr_v2.onnx"))
    ap.add_argument("--spec", default=os.path.join(ROOT, "spec", "labels.txt"))
    ap.add_argument("--alpr", default="", help="real data root: splits the report into names real "
                                               "photos cover and names only synthetic covers")
    ap.add_argument("--margin", type=float, default=0.03, help="the trainer's clean-eval margin")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--softmax", action="store_true", help="the model emits logits, not probabilities")
    a = ap.parse_args()

    sp = L.load(a.spec)
    reg = sp.head("region")
    r = E.Recognizer(a.ocr, a.spec, softmax=a.softmax)
    items = E.scan_synth(a.data, a.limit)
    if not items:
        raise SystemExit("no crops in %s (generate with --region sweep)" % a.data)

    covered = set()
    if a.alpr:
        import train_ocr as T
        al = T.AlprSet(a.alpr, sp)
        covered = {al.items[i][1] for i in al.train_idx}

    hit = np.zeros(reg.n, dtype=np.int64)
    tot = np.zeros(reg.n, dtype=np.int64)
    conf = np.zeros(reg.n)
    for path, want, box_or_tag in items:
        rgb = I.load_rgb(path)
        H, W, _ = rgb.shape
        box = box_or_tag if isinstance(box_or_tag, tuple) else (0.0, 0.0, float(W), float(H))
        out = r.read(rgb, box, [a.margin])["region"]
        t = want[0]
        tot[t] += 1
        conf[t] += out["conf"]
        if out["top"][0] == reg.tok[t]:
            hit[t] += 1

    seen = tot > 0
    print("%d crops over %d region names (%s)" % (len(items), int(seen.sum()), os.path.basename(a.ocr)))
    print("  overall            %5.1f%%" % (100.0 * hit.sum() / max(1, tot.sum())))
    if covered:
        m = np.array([i in covered for i in range(reg.n)]) & seen
        print("  real photos cover  %5.1f%%   (%d names)" % (100.0 * hit[m].sum() / max(1, tot[m].sum()),
                                                             int(m.sum())))
        m2 = ~np.array([i in covered for i in range(reg.n)]) & seen
        print("  synthetic only     %5.1f%%   (%d names)" % (100.0 * hit[m2].sum() / max(1, tot[m2].sum()),
                                                             int(m2.sum())))
    print("\n  the 2025 additions:")
    bad = 0
    for name in NEW_2025:
        i = sp.index_of("region", name)
        if i < 0:
            print("    %-6s not in the spec" % name)
            continue
        if tot[i] == 0:
            print("    %-6s no crop generated" % name)
            continue
        ok = hit[i] == tot[i]
        bad += 0 if hit[i] else 1
        print("    %-6s %d/%d correct, mean conf %.2f  %s"
              % (name, hit[i], tot[i], conf[i] / tot[i], "OK" if ok else ("読めない" if not hit[i] else "")))

    worst = [(hit[i] / tot[i], reg.tok[i]) for i in range(reg.n) if tot[i]]
    worst.sort()
    zero = [n for v, n in worst if v == 0]
    print("\n  %d names never read correctly%s" % (len(zero), (": " + " ".join(zero[:20])) if zero else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
