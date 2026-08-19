"""The recognizer as a PyTorch module, plus loading the shipped ONNX weights into it.

Architecture (lpr_cpp/pure/ref/ARCH.md, and what pure/onnx_export_lpr.hpp emits):

    stem      Conv4x4 s4 (3->128, no pad) -> ReLU -> BN(eps 1e-3)        128x128 -> 32x32
      |
      +-- branch A: block0 + 6 blocks + final 1x1 -> GAP -> feat_A[128]
      +-- branch B: block0 + 5 blocks + final 1x1 -> GAP -> feat_B[128]
    block0    x = x + BN(ReLU(dwconv5x5(x)))
    block     h = BN(ReLU(conv1x1(x)));  x = h + BN(ReLU(dwconv5x5(h)))
    final     x = BN(ReLU(conv1x1(x)))
    heads     Linear(128 -> C) per head; branch A gives region + class_num_01..03,
              branch B gives hiragana + plate_num_01..04

Note the unusual order **Conv -> ReLU -> BN** (the ReLU comes before the BatchNorm). That is what the
original Keras model did; keeping it is what makes the weights transferable.

Heads are widened here, not redesigned: region 133 -> 138 (append-only, so the pretrained rows stay
where they were) and two new heads (plate_kind, legible). New rows/heads start from zero-ish, so the
model begins exactly as accurate as the shipped one.

  python tools/ocr_model.py --check --onnx models/plate_ocr.onnx --ref <lpr_cpp>/pure/ref
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import labels as L  # noqa: E402

BN_EPS = 1e-3
HEADS_A = ["region", "class_num_01", "class_num_02", "class_num_03"]
HEADS_B = ["hiragana", "plate_num_01", "plate_num_02", "plate_num_03", "plate_num_04"]
NEW_A = ["plate_kind"]        # kind is a whole-plate property: branch A (region side)
NEW_B = ["legible"]           # legibility is about the digits: branch B


class CBR(nn.Module):
    """Conv -> ReLU -> BatchNorm, in that order."""

    def __init__(self, cin, cout, k, stride=1, pad=0, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(cout, eps=BN_EPS)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class Branch(nn.Module):
    def __init__(self, ch, nblocks):
        super().__init__()
        self.block0 = CBR(ch, ch, 5, pad=2, groups=ch)
        self.mix = nn.ModuleList([CBR(ch, ch, 1) for _ in range(nblocks)])
        self.dw = nn.ModuleList([CBR(ch, ch, 5, pad=2, groups=ch) for _ in range(nblocks)])
        self.final = CBR(ch, ch, 1)

    def forward(self, x):
        x = x + self.block0(x)
        for mix, dw in zip(self.mix, self.dw):
            h = mix(x)
            x = h + dw(h)
        x = self.final(x)
        # adaptive_avg_pool + flatten, not mean(dim=(2,3)): the former exports as
        # GlobalAveragePool + Flatten, which the C++/WASM ONNX runner already implements, while
        # mean() exports as ReduceMean, which it does not.
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


class PlateNet(nn.Module):
    def __init__(self, spec, ch=128, nblocks_a=6, nblocks_b=5, extra_heads=True):
        super().__init__()
        self.spec = spec
        self.stem = CBR(3, ch, 4, stride=4)
        self.a = Branch(ch, nblocks_a)
        self.b = Branch(ch, nblocks_b)
        self.head_names_a = list(HEADS_A) + (NEW_A if extra_heads else [])
        self.head_names_b = list(HEADS_B) + (NEW_B if extra_heads else [])
        self.head_names = self.head_names_a + self.head_names_b
        dims = {g.name: g.n for g in spec.of_kind("head")}
        self.heads = nn.ModuleDict({n: nn.Linear(ch, dims[n]) for n in self.head_names})

    def forward(self, x):
        s = self.stem(x)
        fa = self.a(s)
        fb = self.b(s)
        out = {}
        for n in self.head_names_a:
            out[n] = self.heads[n](fa)
        for n in self.head_names_b:
            out[n] = self.heads[n](fb)
        return out                       # logits, in spec head order via self.head_names

    def spec_order(self):
        """Head names in the order spec/labels.txt declares them (what ONNX outputs must follow)."""
        return [g.name for g in self.spec.of_kind("head") if g.name in self.heads]


# ---- loading the shipped ONNX weights --------------------------------------------------------
def load_onnx_weights(model, onnx_path, verbose=True):
    """Walk the exported graph in forward order and copy Conv/BN/Gemm parameters in.

    The graph was written by pure/onnx_export_lpr.hpp in exactly the order this module executes, so
    the mapping is positional — no name matching, nothing to get subtly wrong. Heads are matched by
    the graph's output names (region_id_output, hiragana_id_output, ...).
    """
    import onnx
    from onnx import numpy_helper
    g = onnx.load(onnx_path).graph
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}

    convs, bns, gemms = [], [], []
    for n in g.node:
        if n.op_type == "Conv":
            w = init[n.input[1]]
            b = init[n.input[2]] if len(n.input) > 2 else None
            convs.append((w, b))
        elif n.op_type == "BatchNormalization":
            bns.append(tuple(init[n.input[i]] for i in (1, 2, 3, 4)))
        elif n.op_type == "Gemm":
            gemms.append([init[n.input[1]], init[n.input[2]], n.output[0]])
    # the head's public name is on the Softmax that consumes the Gemm, not on the Gemm itself
    soft = {n.input[0]: n.output[0] for n in g.node if n.op_type == "Softmax"}
    for e in gemms:
        e[2] = soft.get(e[2], e[2])

    modules = [model.stem]
    for br in (model.a, model.b):
        modules.append(br.block0)
        for mix, dw in zip(br.mix, br.dw):
            modules += [mix, dw]
        modules.append(br.final)
    if len(modules) != len(convs) or len(modules) != len(bns):
        raise SystemExit("architecture mismatch: %d CBR blocks vs %d convs / %d bns"
                         % (len(modules), len(convs), len(bns)))

    with torch.no_grad():
        for m, (w, b), (gam, beta, mean, var) in zip(modules, convs, bns):
            m.conv.weight.copy_(torch.from_numpy(w.copy()))
            if b is not None and m.conv.bias is not None:
                m.conv.bias.copy_(torch.from_numpy(b.copy()))
            m.bn.weight.copy_(torch.from_numpy(gam.copy()))
            m.bn.bias.copy_(torch.from_numpy(beta.copy()))
            m.bn.running_mean.copy_(torch.from_numpy(mean.copy()))
            m.bn.running_var.copy_(torch.from_numpy(var.copy()))

        # heads: graph output name -> our head name
        alias = {"region_id_output": "region", "hiragana_id_output": "hiragana"}
        loaded = []
        for (W, bias, out_name) in gemms:
            base = out_name.replace("_output", "")
            name = alias.get(out_name, base)
            if name not in model.heads:
                if verbose:
                    print("  skipping graph head %s (not in the spec)" % out_name)
                continue
            lin = model.heads[name]
            src_w = torch.from_numpy(W.copy())          # [in, out] in the graph
            n_src = src_w.shape[1]
            n_dst = lin.weight.shape[0]
            lin.weight.zero_()
            lin.bias.zero_()
            k = min(n_src, n_dst)
            lin.weight[:k].copy_(src_w.t()[:k])
            lin.bias[:k].copy_(torch.from_numpy(bias.copy())[:k])
            # Appended classes (region 133->138) must start out *losing*. With zero weights and zero
            # bias their logit is exactly 0.0, which beats every real class whose logit is negative —
            # measured: that alone cost 6 points of region accuracy (91.7% -> 85.4%) before a single
            # training step. A large negative bias makes them inert until they are trained.
            if n_dst > k:
                lin.bias[k:].fill_(-10.0)
            loaded.append("%s(%d->%d)" % (name, n_src, n_dst))
    if verbose:
        print("loaded %d CBR blocks and heads: %s" % (len(modules), ", ".join(loaded)))
        missing = [n for n in model.head_names if n not in
                   [x.split("(")[0] for x in loaded]]
        if missing:
            print("heads starting from scratch: %s" % ", ".join(missing))
    return model


def onnx_head_order(onnx_path):
    import onnx
    return [o.name for o in onnx.load(onnx_path).graph.output]


# ---- self-check: the port must reproduce the ONNX ---------------------------------------------
def check(onnx_path, ref_dir, spec_path):
    import onnxruntime as ort
    sp = L.load(spec_path)
    model = PlateNet(sp, extra_heads=True)
    load_onnx_weights(model, onnx_path)
    model.eval()

    x = np.fromfile(os.path.join(ref_dir, "input.bin"), dtype=np.float32).reshape(1, 3, 128, 128)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(open(onnx_path, "rb").read(), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    ref = [o.reshape(-1) for o in sess.run(names, {sess.get_inputs()[0].name: x})]

    with torch.no_grad():
        out = model(torch.from_numpy(x))
    alias = {"region_id_output": "region", "hiragana_id_output": "hiragana"}
    worst, argok = 0.0, 0
    for name, r in zip(names, ref):
        ours = out[alias.get(name, name.replace("_output", ""))][0].numpy()
        ours = np.exp(ours - ours.max())
        ours = ours / ours.sum()                      # the graph softmaxes; the module gives logits
        n = min(len(r), len(ours))
        d = float(np.max(np.abs(ours[:n] - r[:n])))
        worst = max(worst, d)
        argok += int(np.argmax(ours) == np.argmax(r))
        print("  %-22s dim %3d->%3d  worst %.2e  argmax %s" %
              (name, len(r), len(ours), d, "ok" if np.argmax(ours) == np.argmax(r) else "MISMATCH"))
    print("PyTorch port vs ONNX: worst %.3e, argmax %d/%d  %s"
          % (worst, argok, len(names), "MATCH" if worst < 1e-4 and argok == len(names) else "MISMATCH"))
    return 0 if (worst < 1e-4 and argok == len(names)) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--onnx", default=os.path.join(ROOT, "models", "plate_ocr.onnx"))
    ap.add_argument("--ref", default="")
    ap.add_argument("--spec", default=os.path.join(ROOT, "spec", "labels.txt"))
    a = ap.parse_args()
    if a.check:
        if not a.ref:
            raise SystemExit("--check needs --ref <dir with input.bin> (lpr_cpp/pure/ref)")
        return check(a.onnx, a.ref, a.spec)
    sp = L.load(a.spec)
    m = PlateNet(sp)
    n = sum(p.numel() for p in m.parameters())
    print("PlateNet: %d parameters (%.2f MB fp32), heads: %s"
          % (n, n * 4 / 1048576, ", ".join("%s:%d" % (k, v.out_features) for k, v in m.heads.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
