"""Train the 4-corner regressor (M6) on generated data and export it to ONNX.

The data is free: `jlpr gen` writes corners.txt next to every crop, so this trains on exactly the
geometry the renderer used. Each sample is built the way inference will see it —
box = bbox(corners), expanded by 25%, resampled to 64x64 — with a random jitter on the box so the net
learns to cope with a detector that is a few percent off.

  python tools/train_corner.py --synth data/synth --steps 3000 --batch 64 --export models/plate_corner.onnx

Validation reports the mean corner error as a percentage of the plate's width, which is the number
that matters downstream: 1.5% of a 330 mm plate is 5 mm, well under a glyph stroke.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import infer as I           # noqa: E402
import corner_model as C    # noqa: E402
from rng import Rng         # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def sample_crop(rgb, corners, jitter, rng):
    """Crop the way inference will: bbox of the plate, expanded, with a little box jitter.
    Returns the 64x64 input and the 8 target coordinates in that crop's normalised frame."""
    xs, ys = corners[0::2], corners[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    bw, bh = x1 - x0, y1 - y0
    if jitter > 0:
        x0 += bw * rng.range(-jitter, jitter)
        x1 += bw * rng.range(-jitter, jitter)
        y0 += bh * rng.range(-jitter, jitter)
        y1 += bh * rng.range(-jitter, jitter)
        bw, bh = max(1e-3, x1 - x0), max(1e-3, y1 - y0)
    e = C.BOX_EXPAND
    cx0, cy0 = x0 - bw * e, y0 - bh * e
    cx1, cy1 = x1 + bw * e, y1 + bh * e
    g = np.arange(C.IN_PX, dtype=np.float32) + 0.5
    sx = cx0 + g * (cx1 - cx0) / C.IN_PX - 0.5
    sy = cy0 + g * (cy1 - cy0) / C.IN_PX - 0.5
    v = I._sample(rgb, sx[None, :].repeat(C.IN_PX, 0), sy[:, None].repeat(C.IN_PX, 1)) / np.float32(255.0)
    x = np.ascontiguousarray(v.transpose(2, 0, 1))
    t = []
    for k in range(4):
        t.append((corners[2 * k] - cx0) / (cx1 - cx0))
        t.append((corners[2 * k + 1] - cy0) / (cy1 - cy0))
    return x, np.array(t, dtype=np.float32), (cx1 - cx0)


class CornerSet:
    def __init__(self, root):
        self.root = root
        self.items = []
        cp = os.path.join(root, "corners.txt")
        if not os.path.exists(cp):
            raise SystemExit("no corners.txt in %s (generate with `jlpr gen`)" % root)
        for line in open(cp, encoding="utf-8"):
            q = line.split()
            if len(q) >= 9 and os.path.exists(os.path.join(root, q[0])):
                self.items.append((os.path.join(root, q[0]), [float(v) for v in q[1:9]]))

    def __len__(self):
        return len(self.items)

    def batch(self, n, rng, jitter=0.04, pool=None):
        plan = []
        for _ in range(n):
            i = rng.below(len(self.items))
            plan.append((i, [rng.range(-jitter, jitter) for _ in range(4)] if jitter else None))
        # the jitter draws happen above so the rng order does not depend on threading
        def load(job):
            i, j = job
            path, corners = self.items[i]
            rgb = I.load_rgb(path)
            xs, ys = corners[0::2], corners[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            bw, bh = x1 - x0, y1 - y0
            if j:
                x0 += bw * j[0]; x1 += bw * j[1]; y0 += bh * j[2]; y1 += bh * j[3]
                bw, bh = max(1e-3, x1 - x0), max(1e-3, y1 - y0)
            e = C.BOX_EXPAND
            cx0, cy0, cx1, cy1 = x0 - bw * e, y0 - bh * e, x1 + bw * e, y1 + bh * e
            g = np.arange(C.IN_PX, dtype=np.float32) + 0.5
            sx = cx0 + g * (cx1 - cx0) / C.IN_PX - 0.5
            sy = cy0 + g * (cy1 - cy0) / C.IN_PX - 0.5
            v = I._sample(rgb, sx[None, :].repeat(C.IN_PX, 0), sy[:, None].repeat(C.IN_PX, 1))
            xarr = np.ascontiguousarray((v / np.float32(255.0)).transpose(2, 0, 1))
            t = []
            for k in range(4):
                t.append((corners[2 * k] - cx0) / (cx1 - cx0))
                t.append((corners[2 * k + 1] - cy0) / (cy1 - cy0))
            return xarr, np.array(t, dtype=np.float32)
        got = list(pool.map(load, plan)) if pool else [load(j) for j in plan]
        x = torch.from_numpy(np.stack([g[0] for g in got])).float()
        y = torch.from_numpy(np.stack([g[1] for g in got])).float()
        return x, y


@torch.no_grad()
def evaluate(model, ds, device, n=400, seed=9):
    model.eval()
    rng = Rng(seed)
    errs = []
    for _ in range(max(1, n // 32)):
        x, y = ds.batch(32, rng, jitter=0.04)
        p = model(x.to(device)).cpu()
        # error per corner, as a fraction of the plate width (the plate spans 1/(1+2e) of the crop)
        d = (p - y).view(-1, 4, 2)
        px_w = 1.0 / (1.0 + 2 * C.BOX_EXPAND)
        errs.append((d.norm(dim=2) / px_w).mean().item())
    model.train()
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--val", default="")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=250)
    ap.add_argument("--export", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    train = CornerSet(a.synth)
    val = CornerSet(a.val) if a.val else train
    print("corner training: %d crops (val %d), device %s" % (len(train), len(val), a.device))
    model = C.CornerNet(a.width).to(a.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    pool = None
    if a.workers > 0:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=a.workers)
    rng = Rng(a.seed)
    t0 = time.time()
    run = None
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / 100.0) * (0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train.batch(a.batch, rng, 0.04, pool)
        pred = model(x.to(a.device))
        loss = F.smooth_l1_loss(pred, y.to(a.device), beta=0.02)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = loss.detach().item()
        run = lv if run is None else 0.9 * run + 0.1 * lv
        if step % 50 == 0 or step == 1:
            print("  step %5d/%d  loss %.5f  lr %.2e  %5.1fs" % (step, a.steps, run, lr,
                                                                 time.time() - t0), flush=True)
        if a.eval_every and step % a.eval_every == 0:
            err = evaluate(model, val, a.device)
            print("  eval @%d: mean corner error %.2f%% of plate width" % (step, 100 * err), flush=True)
    err = evaluate(model, val, a.device, n=800)
    print("final: mean corner error %.2f%% of plate width (target <= 1.5%%)" % (100 * err))
    if a.save:
        torch.save(model.state_dict(), a.save)
        print("saved %s" % a.save)
    if a.export:
        C.export_onnx(model, a.export)
    return 0


if __name__ == "__main__":
    sys.exit(main())
