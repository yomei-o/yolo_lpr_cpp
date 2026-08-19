// Decode for a YOLOv8/v11/v12-shaped head: one tensor [1, 4+nc, N] plus NMS.
//
// WHAT THE TENSOR ALREADY CONTAINS (get this wrong and nothing looks broken):
//   * class scores are **already sigmoided** in an Ultralytics export — do not sigmoid again
//     (the same trap documented at the top of infer_yolox.hpp);
//   * the 4 box numbers are in **input-image pixels**, not grid units — the graph has already
//     applied the DFL softmax, the anchor offsets and the stride multiply;
//   * whether those 4 numbers are xyxy or cxcywh depends on the export. Measured on
//     PlateYOLO-JP-320x320 (NMS stripped at /model/model.21/Concat_5_output_0): **xyxy**, matching
//     the full model's post-NMS output to the digit. `BoxFmt` makes the choice explicit rather
//     than assumed.
#pragma once
#include "autograd.hpp"
#include "infer_yolox.hpp"   // Det + the shared IoU/NMS shape
#include <algorithm>
#include <vector>

enum class BoxFmt { XYXY, CXCYWH };

// raw: [1, 4+nc, N]. Returns detections in input-image pixel space (before un-letterboxing).
inline std::vector<Det> v8_detect(const Tensor& raw, int64_t nc, float conf_thr = 0.25f,
                                 float nms_thr = 0.45f, BoxFmt fmt = BoxFmt::XYXY) {
  const int64_t C = raw->shape[1], N = raw->shape[2];
  if (C != 4 + nc) {
    printf("v8_detect: head has %lld channels but nc=%lld implies %lld\n", (long long)C,
           (long long)nc, (long long)(4 + nc));
    std::exit(1);
  }
  const float* d = raw->data.data();
  std::vector<Det> cand;
  for (int64_t i = 0; i < N; ++i) {
    int best = 0;
    float bestp = -1.f;
    for (int64_t c = 0; c < nc; ++c) {
      const float p = d[(4 + c) * N + i];
      if (p > bestp) { bestp = p; best = (int)c; }
    }
    if (bestp < conf_thr) continue;
    float a = d[0 * N + i], b = d[1 * N + i], e = d[2 * N + i], f = d[3 * N + i];
    Det det{};
    if (fmt == BoxFmt::XYXY) { det.x1 = a; det.y1 = b; det.x2 = e; det.y2 = f; }
    else { det.x1 = a - e / 2; det.y1 = b - f / 2; det.x2 = a + e / 2; det.y2 = b + f / 2; }
    det.score = bestp;
    det.cls = best;
    cand.push_back(det);
  }
  std::sort(cand.begin(), cand.end(), [](const Det& a, const Det& b) { return a.score > b.score; });
  std::vector<Det> out;
  std::vector<char> dead(cand.size(), 0);
  auto iou = [](const Det& a, const Det& b) {
    float iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
    if (iw <= 0 || ih <= 0) return 0.f;
    float inter = iw * ih;
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return inter / ua;
  };
  for (size_t a = 0; a < cand.size(); ++a) {
    if (dead[a]) continue;
    out.push_back(cand[a]);
    for (size_t b = a + 1; b < cand.size(); ++b)
      if (!dead[b] && cand[b].cls == cand[a].cls && iou(cand[a], cand[b]) > nms_thr) dead[b] = 1;
  }
  return out;
}
