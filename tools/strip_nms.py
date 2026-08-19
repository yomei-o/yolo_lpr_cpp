"""Cut an Ultralytics-style detector ONNX just before its NMS tail.

An export made with nms=True ends in NonMaxSuppression / GatherND / ScatterND / NonZero / Where /
Expand / Pad. Those are control-flow-ish ops that a small hand-written interpreter has no business
implementing, and they are not needed: the tensor feeding the NMS already holds pixel-space boxes
and sigmoided class scores, so thresholding + greedy NMS in ~40 lines of C++ gives the same result
(verified: the kept box matches the full model's output to the digit).

  python tools/strip_nms.py <in.onnx> <out.onnx> [--out-tensor NAME]

With no --out-tensor it picks the last Concat that feeds the NMS chain, which for
PlateYOLO-JP-*.onnx is /model/model.21/Concat_5_output_0 -> [1, 4+nc, N].
"""
import argparse
import os
import sys

import onnx


def find_head_tensor(model):
    """The tensor that the NMS subgraph consumes: the head's final Concat."""
    names = [n.output[0] for n in model.graph.node
             if n.op_type == "Concat" and "/model.2" in n.output[0] and "Concat" in n.output[0]]
    if not names:
        names = [n.output[0] for n in model.graph.node if n.op_type == "Concat"]
    return names[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--out-tensor", default="")
    a = ap.parse_args()

    m = onnx.load(a.src)
    tensor = a.out_tensor or find_head_tensor(m)
    inputs = [i.name for i in m.graph.input]
    onnx.utils.extract_model(a.src, a.dst, inputs, [tensor])

    out = onnx.load(a.dst)
    shape = [d.dim_value or d.dim_param for d in out.graph.output[0].type.tensor_type.shape.dim]
    ops = sorted({n.op_type for n in out.graph.node})
    print("cut at %s -> %s" % (tensor, shape))
    print("%d nodes, %.1f MB, ops: %s" % (len(out.graph.node), os.path.getsize(a.dst) / 1048576,
                                          " ".join(ops)))
    print("wrote %s" % a.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
