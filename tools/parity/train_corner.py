"""Parity test M6b: the C++ corner trainer takes the same step as the PyTorch one.

`jlpr train --model corner --dump-fixture` writes step 1 as it happened — the 64x64 crops it built,
the 8 targets per crop, the loss, and the gradient of every parameter (after clip_grad_norm, before
the optimiser). This script loads the *same* ONNX weights into tools/corner_model.CornerNet, replays
that batch, and compares. Same weights and same pixels on both sides, so a difference is the
implementation (BatchNorm in training mode, the Huber beta, the clip), not the data pipeline.

  jlpr train --model corner --synth data/synth --init models/plate_corner.onnx --steps 1 --batch 8 \
             --dump-fixture scratch/crn_fix.bin
  python tools/parity/train_corner.py --fixture scratch/crn_fix.bin --onnx models/plate_corner.onnx

Passing means the loss agrees to 1e-5 relative and every parameter's gradient to 1e-3 relative
(the C++ conv/BN sum in a different order than PyTorch's; that error is ~3e-05 on a forward pass and
grows through the backward).
"""
import argparse
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_fixture(path):
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:8] != b"JLPRCRN1":
        raise SystemExit("%s is not a JLPRCRN1 fixture" % path)
    off = 8
    B, px, nparam = struct.unpack_from("<3i", blob, off)
    off += 12
    n = B * 3 * px * px
    x = np.frombuffer(blob, np.float32, n, off).reshape(B, 3, px, px).copy()
    off += 4 * n
    y = np.frombuffer(blob, np.float32, B * 8, off).reshape(B, 8).copy()
    off += 4 * B * 8
    (loss,) = struct.unpack_from("<f", blob, off)
    off += 4
    grads = {}
    for _ in range(nparam):
        (ln,) = struct.unpack_from("<i", blob, off)
        off += 4
        name = blob[off:off + ln].decode("utf-8")
        off += ln
        (numel,) = struct.unpack_from("<i", blob, off)
        off += 4
        grads[name] = np.frombuffer(blob, np.float32, numel, off).copy()
        off += 4 * numel
    return dict(B=B, px=px, x=x, y=y, loss=loss, grads=grads)


def check_written_graph(path, C, torch, onnx, np, tol=1e-5):
    """A graph built by pure/train_corner.hpp must mean the same thing to everyone else."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (onnxruntime not installed here — skipping the written-graph check)")
        return True
    init = {t.name: onnx.numpy_helper.to_array(t) for t in onnx.load(path).graph.initializer}
    width = init["body.0.conv.weight"].shape[0]
    model = C.CornerNet(width)
    sd = model.state_dict()
    for k in sd:
        if k in init:
            sd[k] = torch.from_numpy(np.array(init[k]).copy())
    model.load_state_dict(sd)
    model.eval()
    rng = np.random.default_rng(3)
    x = rng.random((1, 3, C.IN_PX, C.IN_PX), dtype=np.float32)   # the graph declares batch 1
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    with torch.no_grad():
        want = model(torch.from_numpy(x)).numpy()
    d = float(np.abs(got - want).max())
    print("  written graph %s: onnxruntime vs torch max |diff| %.2e" % (os.path.basename(path), d))
    return d <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="scratch/crn_fix.bin")
    ap.add_argument("--onnx", default=os.path.join(ROOT, "models", "plate_corner.onnx"))
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--check-graph", dest="check_graph", default="",
                    help="also check an ONNX this repo *wrote* (jlpr train --model corner --init "
                         "random --export ...): onnxruntime must agree with the PyTorch CornerNet "
                         "carrying the same weights, which is what proves the C++ ONNX writer emits "
                         "the graph it thinks it does")
    ap.add_argument("--loss-tol", type=float, default=1e-5)
    ap.add_argument("--grad-tol", type=float, default=1e-3)
    a = ap.parse_args()

    fx = read_fixture(a.fixture)
    try:
        import onnx
        import torch
        import torch.nn.functional as F
        import corner_model as C
    except ImportError as e:
        print("needs torch + onnx here (%s)" % e)
        return 2

    init = {t.name: onnx.numpy_helper.to_array(t) for t in onnx.load(a.onnx).graph.initializer}
    width = init["body.0.conv.weight"].shape[0]
    model = C.CornerNet(width)
    sd = model.state_dict()
    missing = [k for k in sd if k not in init and not k.endswith("num_batches_tracked")]
    if missing:
        print("the ONNX has no weights for: %s" % ", ".join(missing))
        return 2
    for k in sd:
        if k in init:
            sd[k] = torch.from_numpy(np.array(init[k]).copy())
    model.load_state_dict(sd)
    model.train()                      # BatchNorm on batch statistics, as the C++ trainer runs it

    x = torch.from_numpy(fx["x"])
    y = torch.from_numpy(fx["y"])
    pred = model(x)
    loss = F.smooth_l1_loss(pred, y, beta=a.beta)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)

    rel = abs(float(loss) - fx["loss"]) / max(1e-9, abs(float(loss)))
    ok = rel <= a.loss_tol
    print("CornerNet(width=%d), batch %d, %d parameter tensors" % (width, fx["B"], len(fx["grads"])))
    print("  loss      C++ %.8f   python %.8f   rel %.2e" % (fx["loss"], float(loss), rel))

    worst, worst_name = 0.0, ""
    for name, p in model.named_parameters():
        if name not in fx["grads"]:
            print("  no C++ gradient for %s" % name)
            ok = False
            continue
        g_py = p.grad.detach().numpy().ravel()
        g_cpp = fx["grads"][name]
        scale = max(1e-12, float(np.abs(g_py).max()))
        r = float(np.abs(g_cpp - g_py).max()) / scale
        if r > worst:
            worst, worst_name = r, name
    ok = ok and worst <= a.grad_tol
    print("  gradients worst max|C++ - python| / max|python| = %.2e  (%s)" % (worst, worst_name))
    if a.check_graph:
        ok = check_written_graph(a.check_graph, C, torch, onnx, np) and ok

    print("PARITY %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
