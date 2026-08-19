"""The 4-corner regressor (M6) — the stage that turns a detector box into a canonical plate crop.

Why it exists, in one measurement: the region head's answer moves with the framing. Swapping the
detector alone (same recogniser, same photo) took the region confidence from 0.57 to 0.92, and
lpr_cpp measured the same plate reading 奄美 / 横浜 / 練馬 depending on a few percent of crop margin.
A box is not a plate: it is an axis-aligned rectangle around a quadrilateral that is usually rotated
and always perspective-distorted. Estimating the four corners and warping to a fixed rectangle
removes that variable instead of averaging over it (which is what the 6-crop TTA does today, at 6x
the inference cost).

Input : 64x64 RGB of the detector box expanded by 25%, /255
Output: 8 numbers = the plate's TL,TR,BR,BL corners in that crop's normalised coordinates
Loss  : smooth L1 (Huber). Corners can fall slightly outside the crop, so the head is linear, not
        sigmoid-squashed.

The training targets come free from the generator (pure/gen_render.hpp writes corners.txt), which is
the only reason this is cheap: nobody can hand-label four corners on 100k plates.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_PX = 64
BOX_EXPAND = 0.25          # how much of the surroundings the corner net gets to see


class CBR(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(cout, eps=1e-3)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class CornerNet(nn.Module):
    """Small on purpose: this runs on every detection, in a browser, on a phone.

    No global pooling anywhere — pooling throws away *where* things are, which is the only thing this
    network is asked about. The 4x4 feature map is flattened straight into the coordinate head.
    """

    def __init__(self, width=24):
        super().__init__()
        w = width
        self.body = nn.Sequential(
            CBR(3, w, 2),          # 64 -> 32
            CBR(w, w * 2, 2),      # 32 -> 16
            CBR(w * 2, w * 3, 2),  # 16 -> 8
            CBR(w * 3, w * 4, 2),  # 8  -> 4
        )
        self.head = nn.Linear(w * 4 * 4 * 4, 8)

    def forward(self, x):
        f = self.body(x)
        return self.head(f.flatten(1))


def export_onnx(model, path, opset=13):
    model.eval()
    dummy = torch.zeros(1, 3, IN_PX, IN_PX)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["corners"],
                      opset_version=opset, dynamo=False, dynamic_axes=None)
    print("wrote %s (%.2f MB)" % (path, os.path.getsize(path) / 1048576))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=24)
    a = ap.parse_args()
    m = CornerNet(a.width)
    n = sum(p.numel() for p in m.parameters())
    print("CornerNet(width=%d): %d parameters (%.2f MB fp32), input %dx%d, output 8"
          % (a.width, n, n * 4 / 1048576, IN_PX, IN_PX))
    return 0


if __name__ == "__main__":
    sys.exit(main())
