"""Train the recognizer (Python side of M5) and export it back to ONNX.

Starts from the shipped weights, not from scratch: `tools/ocr_model.py` loads models/plate_ocr.onnx
into the PyTorch port (verified to 3.3e-05), widens region 133 -> 138 append-only and adds the
plate_kind / legible heads. So step 0 of training is already as accurate as what we ship.

Data sources, mixed per batch:
  * synthetic dirs (`jlpr gen` / tools/gen.py) — every head labelled, unlimited volume
  * dyama/alpr_jp — real crops, but only the 地域名 is known, so the other heads are masked out.
    Masking matters: writing zeros there would actively teach the model wrong digits.

Three things were measured, not assumed, and each one changed the recipe:
  * synthetic glyphs transfer 72-92% on digits but only ~28% on the 133-class region head, so
    **the region head is trained on real crops only** (masked out for synthetic samples). The first
    run without that mask dropped real region accuracy 78% -> 67% in 50 steps.
  * **BatchNorm statistics are frozen** by default. Synthetic-heavy batches move them and drag real
    accuracy down with them; `--train-bn` puts it back if you have enough real data.
  * the pretrained backbone gets **lr * 0.1** while the new heads get the full rate, so the features
    do not drift toward the synthetic font.

  python tools/train_ocr.py --synth data/synth --alpr <alpr_jp root> --steps 2000 --batch 32
  python tools/train_ocr.py ... --export models/plate_ocr_v2.onnx
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
import labels as L        # noqa: E402
import infer as I         # noqa: E402
import ocr_model as M     # noqa: E402
from rng import Rng       # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CATEGORIES = ["自家用", "自家用(軽)", "事業用", "事業用(軽)"]


# ---- data ------------------------------------------------------------------------------------
def load_crop(path, box=None, margin=0.0):
    rgb = I.load_rgb(path)
    H, W, _ = rgb.shape
    if box is None:
        box = (0.0, 0.0, float(W), float(H))
    return I.crop_to_input(rgb, box[0], box[1], box[2], box[3], margin)[0]   # (3,128,128)


class SynthSet:
    """A generated directory: labels.txt (11 head indices) + corners.txt (the plate's true box)."""

    def __init__(self, root, spec, teach_region=False, uncovered=None):
        self.root = root
        self.spec = spec
        self.teach_region = teach_region
        # Regions the real training split has no example of. For those classes synthetic glyphs are
        # the ONLY signal available, so the region loss is enabled for them even though it is masked
        # everywhere else. Without this the 2025 additions (十勝/日光/江戸川/安曇野/南信州) sit at
        # their -10 init forever: measured probability 4e-11, i.e. structurally unpredictable.
        self.uncovered = uncovered or set()
        self.items = []
        boxes = {}
        cp = os.path.join(root, "corners.txt")
        if os.path.exists(cp):
            for line in open(cp, encoding="utf-8"):
                q = line.split()
                if len(q) >= 9:
                    v = [float(x) for x in q[1:9]]
                    boxes[q[0]] = (min(v[0::2]), min(v[1::2]), max(v[0::2]), max(v[1::2]))
        for line in open(os.path.join(root, "labels.txt"), encoding="utf-8"):
            p = line.split()
            if len(p) >= 12 and os.path.exists(os.path.join(root, p[0])):
                self.items.append((os.path.join(root, p[0]), [int(v) for v in p[1:12]],
                                   boxes.get(p[0])))

    def __len__(self):
        return len(self.items)

    def margin(self, rng):
        # the generator already framed the crop; re-crop around the true box with a random margin so
        # the model sees the same framing jitter a detector will produce
        return rng.range(-0.03, 0.12)

    def load(self, i, margin):
        path, heads, box = self.items[i]
        x = load_crop(path, box, margin if box else 0.0)
        mask = [1] * 11
        if not self.teach_region and heads[0] not in self.uncovered:
            # Measured (RESUME): synthetic glyphs transfer 72-92%% on digits but only ~28%% on the
            # 133-class region head, because no free font reproduces the real plate typeface. Letting
            # synthetic data train the region head therefore *overwrites* what the shipped weights
            # learned from real plates — the first smoke run dropped real region 78%% -> 67%% in 50
            # steps. So region is masked out for synthetic samples unless asked for explicitly.
            mask[0] = 0
        return x, heads, mask

    def get(self, i, rng):
        return self.load(i, self.margin(rng))


class AlprSet:
    """Real crops from alpr_jp: only the region head is labelled, the rest is masked."""

    def __init__(self, root, spec, hold_out=0.2):
        self.spec = spec
        reg = spec.head("region")
        self.items = []
        for cat in CATEGORIES:
            d = os.path.join(root, cat)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                rd = os.path.join(d, name)
                if not os.path.isdir(rd):
                    continue
                idx = reg.tok.index(name) if name in reg.tok else -1
                if idx < 0:
                    continue
                for f in sorted(os.listdir(rd)):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        # sort by a canonical "<category>/<region>/<file>" key, never by the OS path:
                        # os.path.join uses backslashes on Windows and the C++ side uses forward
                        # slashes, which would order the list differently and give the two languages
                        # different hold-out splits.
                        self.items.append((os.path.join(rd, f), idx, "%s/%s/%s" % (cat, name, f)))
        self.items.sort(key=lambda t: t[2])
        rng = Rng(11)                                   # deterministic split, shared prng
        order = list(range(len(self.items)))
        for i in range(len(order) - 1, 0, -1):
            j = rng.below(i + 1)
            order[i], order[j] = order[j], order[i]
        cut = int(len(order) * (1 - hold_out))
        self.train_idx = sorted(order[:cut])
        self.val_idx = sorted(order[cut:])

    def margin(self, rng):
        # Three draws, in this order: crop margin, brightness, contrast. The C++ trainer draws the
        # same three for a real sample (pure/train_ocr.hpp), which is what keeps the step-by-step
        # loss parity. 576 real crops seen thousands of times overfit fast without this: the first
        # GPU run peaked at 97.9% (step 500-1000) and slid to 95.8% by step 2000.
        return (rng.range(-0.04, 0.12), rng.range(0.75, 1.25), rng.range(0.85, 1.15))

    def load(self, i, aug):
        path, region = self.items[i][0], self.items[i][1]
        margin, bright, contrast = aug if isinstance(aug, tuple) else (aug, 1.0, 1.0)
        x = load_crop(path, None, margin)
        x = np.clip(((x - 0.5) * contrast + 0.5) * bright, 0.0, 1.0).astype(np.float32)
        heads = [0] * 11
        heads[0] = region
        mask = [1] + [0] * 10                           # region only
        return x, heads, mask

    def get(self, i, rng):
        return self.load(i, self.margin(rng))


def batch_from(sets, weights, batch, rng, spec, pool=None):
    """Draw the whole batch first, then (optionally) decode the images in parallel.

    The draws stay strictly sequential — set choice, item index, crop margin, in that order — because
    that order is the contract with the C++ trainer (tools/parity/train.py compares step-by-step
    losses). Only the file decoding is parallel, and it does not touch the rng."""
    total = sum(weights)
    plan = []
    for _ in range(batch):
        r = rng.unit() * total
        acc = 0.0
        pick = 0
        for k, w in enumerate(weights):
            acc += w
            if r <= acc:
                pick = k
                break
        ds = sets[pick]
        idx = ds.pick(rng)
        margin = ds.margin(rng)              # drawn here so the sequence is identical either way
        plan.append((ds, idx, margin))
    if pool is None:
        got = [ds.load(idx, margin) for (ds, idx, margin) in plan]
    else:
        got = list(pool.map(lambda t: t[0].load(t[1], t[2]), plan))
    xs = [g[0] for g in got]
    ys = [g[1] for g in got]
    ms = [g[2] for g in got]
    x = torch.from_numpy(np.stack(xs)).float()
    return x, torch.tensor(ys, dtype=torch.long), torch.tensor(ms, dtype=torch.float32)


# ---- loss / eval -----------------------------------------------------------------------------
def multihead_loss(out, order, y, m):
    loss = 0.0
    for h, name in enumerate(order):
        logits = out[name]
        ce = F.cross_entropy(logits, y[:, h], reduction="none")
        w = m[:, h]
        denom = w.sum().clamp(min=1.0)
        loss = loss + (ce * w).sum() / denom
    return loss


EVAL_MARGIN = 0.03      # fixed: the metric has to be a property of the model, not of the augmentation


@torch.no_grad()
def eval_region(model, ds, order, idxs, device, limit=0):
    """Clean, fixed-margin evaluation. Passing the training augmentation through here made the
    *baseline* look 8 points worse (84.0% instead of 91.7%) and every comparison noisy."""
    model.eval()
    n = ok = 0
    for i in (idxs[:limit] if limit else idxs):
        x, heads, _ = ds.load(i, (EVAL_MARGIN, 1.0, 1.0) if isinstance(ds, AlprSet) else EVAL_MARGIN)
        out = model(torch.from_numpy(x[None]).float().to(device))
        pred = int(out["region"].argmax(1).item())
        ok += int(pred == heads[0])
        n += 1
    model.train()
    return ok / max(1, n), n


@torch.no_grad()
def eval_synth(model, ds, order, device, limit=200):
    model.eval()
    per = [0] * len(order)
    full = n = 0
    for i in range(min(limit, len(ds))):
        x, heads, _ = ds.load(i, EVAL_MARGIN)
        out = model(torch.from_numpy(x[None]).float().to(device))
        allok = True
        for h, name in enumerate(order):
            p = int(out[name].argmax(1).item())
            good = p == heads[h]
            per[h] += int(good)
            if h < 9:                                   # the 9 shipped heads define "whole plate"
                allok = allok and good
        full += int(allok)
        n += 1
    model.train()
    return full / max(1, n), [p / max(1, n) for p in per], n


# ---- export ----------------------------------------------------------------------------------
class WithSoftmax(torch.nn.Module):
    """Inference wrapper: the pipeline sums probabilities over TTA crops, so the graph softmaxes."""

    def __init__(self, model, order):
        super().__init__()
        self.model = model
        self.order = order

    def forward(self, x):
        out = self.model(x)
        return tuple(F.softmax(out[n], dim=1) for n in self.order)


def export_onnx(model, order, path, opset=13):
    model.eval()
    was = next(model.parameters()).device
    model.to("cpu")                     # tracing a CUDA model with a CPU dummy input fails
    wrapper = WithSoftmax(model, order)
    dummy = torch.zeros(1, 3, 128, 128)
    torch.onnx.export(wrapper, dummy, path, input_names=["input"], output_names=list(order),
                      opset_version=opset, dynamo=False,
                      dynamic_axes=None)
    model.to(was)
    print("wrote %s (%.2f MB)" % (path, os.path.getsize(path) / 1048576))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="append", default=[], help="generated dir (repeatable)")
    ap.add_argument("--alpr", default="", help="alpr_jp checkout root")
    ap.add_argument("--init", default=os.path.join(ROOT, "models", "plate_ocr.onnx"))
    ap.add_argument("--spec", default=os.path.join(ROOT, "spec", "labels.txt"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--real-weight", dest="real_weight", type=float, default=0.35,
                    help="share of each batch drawn from real crops")
    ap.add_argument("--freeze-stem", dest="freeze_stem", action="store_true")
    ap.add_argument("--synth-region", dest="synth_region", action="store_true",
                    help="also train the region head on synthetic glyphs (measured to hurt)")
    ap.add_argument("--train-bn", dest="train_bn", action="store_true",
                    help="update BatchNorm statistics (off by default: synthetic-heavy batches move "
                         "them and real accuracy falls)")
    ap.add_argument("--backbone-lr-mult", dest="bb_mult", type=float, default=0.1)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=100)
    ap.add_argument("--eval-limit", dest="eval_limit", type=int, default=120)
    ap.add_argument("--export", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--keep-last", dest="keep_last", action="store_true",
                    help="export the final weights instead of the best hold-out checkpoint")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=0,
                    help="threads for image decoding (draw order is unaffected; 4 is right on Kaggle)")
    ap.add_argument("--no-extra-heads", dest="no_extra", action="store_true",
                    help="keep the original 9 heads (no plate_kind/legible) — needed when the "
                         "parity test compares against a 9-head ONNX")
    ap.add_argument("--dump-loss", dest="dump_loss", action="store_true",
                    help="print only `step N loss X` with a constant lr — what the C++/Python "
                         "training parity test compares")
    a = ap.parse_args()

    sp = L.load(a.spec)
    model = M.PlateNet(sp, extra_heads=not a.no_extra)
    if a.init and os.path.exists(a.init):
        M.load_onnx_weights(model, a.init)
    order = [g.name for g in sp.of_kind("head") if g.name in model.heads]
    model.to(a.device).train()

    # which regions does the real training split actually contain?
    uncovered = set()
    if a.alpr:
        probe = AlprSet(a.alpr, sp)
        covered = {probe.items[i][1] for i in probe.train_idx}
        uncovered = {i for i in range(sp.head("region").n) if i not in covered}
        print("real data covers %d of %d regions; synthetic will teach the other %d (incl. the 2025 "
              "additions)" % (len(covered), sp.head("region").n, len(uncovered)))

    sets, weights = [], []
    for d in a.synth:
        s = SynthSet(d, sp, teach_region=a.synth_region, uncovered=uncovered)
        if len(s):
            s.pick = lambda rng, n=len(s): rng.below(n)
            sets.append(s)
            weights.append(1.0 - a.real_weight)
            print("synthetic: %d crops from %s" % (len(s), d))
    alpr = None
    if a.alpr:
        alpr = AlprSet(a.alpr, sp)
        alpr.pick = lambda rng: alpr.train_idx[rng.below(len(alpr.train_idx))]
        sets.append(alpr)
        weights.append(a.real_weight)
        print("real: %d crops (%d train / %d hold-out) from %s"
              % (len(alpr.items), len(alpr.train_idx), len(alpr.val_idx), a.alpr))
    if not sets:
        raise SystemExit("no data: pass --synth <dir> and/or --alpr <root>")

    if a.freeze_stem:
        for p in model.stem.parameters():
            p.requires_grad = False
        print("stem frozen")

    def set_bn_eval(m):
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()                        # keep running_mean/var; still learns weight/bias

    # New heads move fast, the pretrained backbone moves slowly. Without this the backbone drifts
    # towards the synthetic font and takes the real-data accuracy with it.
    new_names = ("heads.plate_kind", "heads.legible")
    head_params, bb_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if (n.startswith("heads.") or n.startswith(new_names)) else bb_params).append(p)
    # AdamW with weight_decay=0 is exactly Adam, which is what the C++ optimiser implements — the
    # training parity test runs with --wd 0 so the two updates are comparable step by step.
    opt = torch.optim.AdamW([{"params": head_params, "lr": a.lr},
                             {"params": bb_params, "lr": a.lr * a.bb_mult}], weight_decay=a.wd)
    print("param groups: %d head tensors at lr, %d backbone tensors at lr*%.2f%s"
          % (len(head_params), len(bb_params), a.bb_mult,
             "" if a.train_bn else ", BN statistics frozen"))
    pool = None
    if a.workers > 0:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=a.workers)
        print("image decoding on %d threads" % a.workers)
    rng = Rng(a.seed)
    t0 = time.time()
    if alpr:
        # full hold-out, so the baseline and the final number are the same measurement
        acc0, n0 = eval_region(model, alpr, order, alpr.val_idx, a.device, 0)
        print("step 0 (shipped weights): real hold-out region top1 %.1f%% over %d crops"
              % (100 * acc0, n0))

    # Keep the best hold-out checkpoint, not the last one: with only 576 real crops the region head
    # peaks early and then slides (measured: 97.9% at step 500-1000, 95.8% by 2000).
    import copy
    best = {"acc": -1.0, "step": 0, "state": None}
    run_loss = None
    for step in range(1, a.steps + 1):
        lr = a.lr * (step / max(1, a.warmup)) if step <= a.warmup else \
            a.lr * 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(1, a.steps - a.warmup)))
        opt.param_groups[0]["lr"] = lr
        opt.param_groups[1]["lr"] = lr * a.bb_mult
        if not a.train_bn:
            model.apply(set_bn_eval)
        x, y, m = batch_from(sets, weights, a.batch, rng, sp, pool)
        out = model(x.to(a.device))
        loss = multihead_loss(out, order, y.to(a.device), m.to(a.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.0)
        opt.step()
        lv = loss.detach().item()
        run_loss = lv if run_loss is None else 0.9 * run_loss + 0.1 * lv
        if a.dump_loss:
            print("step %d loss %.6f" % (step, lv), flush=True)
        elif step % 25 == 0 or step == 1:
            print("  step %5d/%d  loss %7.3f  lr %.2e  %5.1fs" % (step, a.steps, run_loss, lr,
                                                                  time.time() - t0), flush=True)
        if a.eval_every and not a.dump_loss and step % a.eval_every == 0:
            msg = "  eval @%d:" % step
            if alpr:
                acc, n = eval_region(model, alpr, order, alpr.val_idx, a.device, a.eval_limit)
                msg += "  real region %.1f%% (%d)" % (100 * acc, n)
            if sets and isinstance(sets[0], SynthSet):
                full, per, n = eval_synth(model, sets[0], order, a.device, 120)
                msg += "  synth whole %.1f%% region %.1f%% kind %.1f%%" % (
                    100 * full, 100 * per[0], 100 * per[9])
            if alpr and acc > best["acc"]:
                best = {"acc": acc, "step": step,
                        "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
                msg += "  <- best"
            print(msg, flush=True)

    if alpr and not a.dump_loss:
        acc, n = eval_region(model, alpr, order, alpr.val_idx, a.device, 0)
        print("final (last step): real hold-out region top1 %.1f%% over %d crops" % (100 * acc, n))
        if acc > best["acc"]:
            best = {"acc": acc, "step": a.steps,
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        if best["state"] is not None and not a.keep_last:
            model.load_state_dict(best["state"])
            model.to(a.device)
            print("exporting the BEST checkpoint: step %d, real hold-out region top1 %.1f%%"
                  % (best["step"], 100 * best["acc"]))
    if a.save:
        torch.save(model.state_dict(), a.save)
        print("saved %s" % a.save)
    if a.export:
        export_onnx(model, order, a.export)
    return 0


if __name__ == "__main__":
    sys.exit(main())
