"""Parity test M7b: the C++ v8 detection loss vs the one Ultralytics actually trains with.

The recognizer parity test (tools/parity/train.py) compares two trainers on the same batch. Here the
comparison is tighter and cheaper: `jlpr train --model det --dump-fixture` writes the *exact* head
tensors of one step (box logits, class logits), the ground-truth boxes, its loss and its gradients;
this script feeds those same numbers to `ultralytics.utils.loss.v8DetectionLoss` and compares. Same
inputs on both sides means any difference is the loss implementation itself — the assignment, the
CIoU, the DFL, the gains — and not a different image, a different sampler or a different forward.

  jlpr train --model det --data data/det_smoke --steps 1 --batch 2 --dump-fixture scratch/det_fix.bin
  python tools/parity/train_det.py --fixture scratch/det_fix.bin

Passing means: the three loss terms agree to 1e-5 relative, and every gradient of the six head
tensors agrees to 1e-4 relative (float32 on both sides; the assignment is discrete, so if it ever
disagreed the gradients would differ by whole anchors, not by rounding).
"""
import argparse
import os
import struct
import sys
import types

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_fixture(path):
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:8] != b"JLPRDET1":
        raise SystemExit("%s is not a JLPRDET1 fixture" % path)
    off = 8
    L, B, nc, reg, imgsz = struct.unpack_from("<5i", blob, off)
    off += 20
    hs = struct.unpack_from("<%di" % L, blob, off)
    off += 4 * L
    box, cls = [], []
    for h in hs:
        n = B * 4 * reg * h * h
        box.append(np.frombuffer(blob, np.float32, n, off).reshape(B, 4 * reg, h * h).copy())
        off += 4 * n
    for h in hs:
        n = B * nc * h * h
        cls.append(np.frombuffer(blob, np.float32, n, off).reshape(B, nc, h * h).copy())
        off += 4 * n
    (ngt,) = struct.unpack_from("<i", blob, off)
    off += 4
    gts = []
    for _ in range(ngt):
        (b,) = struct.unpack_from("<i", blob, off)
        off += 4
        c, x1, y1, x2, y2 = struct.unpack_from("<5f", blob, off)
        off += 20
        gts.append((b, c, x1, y1, x2, y2))
    parts = struct.unpack_from("<4f", blob, off)
    off += 16
    gbox, gcls = [], []
    for h in hs:
        n = B * 4 * reg * h * h
        gbox.append(np.frombuffer(blob, np.float32, n, off).reshape(B, 4 * reg, h * h).copy())
        off += 4 * n
    for h in hs:
        n = B * nc * h * h
        gcls.append(np.frombuffer(blob, np.float32, n, off).reshape(B, nc, h * h).copy())
        off += 4 * n
    return dict(L=L, B=B, nc=nc, reg=reg, imgsz=imgsz, hs=hs, box=box, cls=cls, gts=gts,
                parts=parts, gbox=gbox, gcls=gcls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="scratch/det_fix.bin")
    ap.add_argument("--loss-tol", type=float, default=1e-5)
    ap.add_argument("--grad-tol", type=float, default=1e-4)
    a = ap.parse_args()

    fx = read_fixture(a.fixture)
    try:
        import torch
        from ultralytics.utils.loss import v8DetectionLoss
        import ultralytics
    except ImportError as e:
        print("needs torch + ultralytics here (%s)" % e)
        return 2

    B, nc, reg, imgsz, hs = fx["B"], fx["nc"], fx["reg"], fx["imgsz"], fx["hs"]
    strides = [imgsz / h for h in hs]

    # The smallest stand-in a v8DetectionLoss will accept: it reads .args (the gains), .model[-1]'s
    # stride/nc/reg_max, and the device of the first parameter. Building a real DetectionModel would
    # add a yaml, a forward pass and a version dependency for nothing.
    class _Detect(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.nc, self.reg_max = nc, reg
            self.stride = torch.tensor(strides, dtype=torch.float)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.ModuleList([_Detect()])
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.args = types.SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    crit = v8DetectionLoss(_Model())

    boxes = torch.from_numpy(np.concatenate(fx["box"], axis=2)).requires_grad_(True)
    scores = torch.from_numpy(np.concatenate(fx["cls"], axis=2)).requires_grad_(True)
    feats = [torch.zeros(B, 4 * reg + nc, h, h) for h in hs]          # only the shapes are read
    preds = {"boxes": boxes, "scores": scores, "feats": feats}

    # batch["bboxes"] is normalised cxcywh — v8DetectionLoss.preprocess scales it by the image size
    # and converts to xyxy, so the fixture's pixel xyxy has to go back through that.
    bidx = torch.tensor([g[0] for g in fx["gts"]], dtype=torch.float32)
    bcls = torch.tensor([g[1] for g in fx["gts"]], dtype=torch.float32)
    bb = torch.tensor([[(g[2] + g[4]) / 2 / imgsz, (g[3] + g[5]) / 2 / imgsz,
                        (g[4] - g[2]) / imgsz, (g[5] - g[3]) / imgsz] for g in fx["gts"]],
                      dtype=torch.float32)
    batch = {"batch_idx": bidx, "cls": bcls, "bboxes": bb}

    # 8.4 returns the three terms as a vector (already multiplied by the batch size) plus a dict of
    # the same numbers before that multiply; older versions returned the summed scalar.
    parts, items = crit(preds, batch)
    total = parts.sum() if parts.ndim else parts
    total.backward()
    py = [float(v) for v in (items.values() if isinstance(items, dict) else items)]

    total_cpp, box_cpp, cls_cpp, dfl_cpp = fx["parts"]
    names = ["box", "cls", "dfl"]
    cpp = [box_cpp, cls_cpp, dfl_cpp]
    print("ultralytics %s, torch %s, %d gt boxes over %d images, %d anchors"
          % (ultralytics.__version__, torch.__version__, len(fx["gts"]), B, boxes.shape[2]))
    ok = True
    for n, c, p in zip(names, cpp, py):
        rel = abs(c - p) / max(1e-9, abs(p))
        ok = ok and rel <= a.loss_tol
        print("  %-4s C++ %.6f   python %.6f   rel %.2e" % (n, c, p, rel))
    rel_total = abs(total_cpp - float(total)) / max(1e-9, abs(float(total)))
    ok = ok and rel_total <= a.loss_tol
    print("  %-4s C++ %.6f   python %.6f   rel %.2e" % ("sum", total_cpp, float(total), rel_total))

    for what, cppg, pyg in (("box heads", fx["gbox"], boxes.grad), ("cls heads", fx["gcls"], scores.grad)):
        g_cpp = np.concatenate(cppg, axis=2)
        g_py = pyg.detach().numpy()
        scale = max(1e-12, float(np.abs(g_py).max()))
        rel = float(np.abs(g_cpp - g_py).max()) / scale
        ok = ok and rel <= a.grad_tol
        print("  grad %-9s max |C++ - python| / max|python| = %.2e  (max |g| %.3e)" % (what, rel, scale))

    print("PARITY %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
