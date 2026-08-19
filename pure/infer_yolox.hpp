// YOLOX inference: anchor-free decode + class-aware NMS. Mirrors YOLOX decode_outputs
// + postprocess (score = obj*cls, batched NMS).
//
// WHO HAS ALREADY APPLIED THE SIGMOID. There are two sources of head tensors in this repo and
// they disagree, which is worth stating loudly because getting it wrong is nearly invisible:
//
//   * the pure C++ net (`yolox_forward`, `yolox_forward_unfused`) emits RAW LOGITS —
//     net_yolox.hpp builds cls/reg/obj as plain convolutions;
//   * the ONNX graph does NOT. Megvii's YOLOXHead.forward in eval mode concatenates
//     `[reg_output, obj_output.sigmoid(), cls_output.sigmoid()]`, so /head/Concat_*_output_0
//     already carries probabilities. (Confirmed on plate_yolox_tiny.onnx: the Concat's
//     inputs are Conv, Sigmoid, Sigmoid.)
//
// Sigmoiding an already-sigmoided head does not crash and does not look obviously wrong. It
// squashes everything into [0.5, 0.73]: objectness comes out with min 0.5000 and mean 0.5007,
// every class sits at ~0.5, and the argmax becomes noise — so the detector reports boxes
// scattered over the image, all at nearly the same score, and the rare correct one is buried.
// That is what "detection doesn't work" looked like here.
//
// `logits` therefore has to be passed explicitly by ONNX-fed callers.
#pragma once
#include "autograd.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

struct Det { float x1, y1, x2, y2, score; int cls; };

static inline float sig_(float x) { return 1.f / (1.f + std::exp(-x)); }

// raw: per-level (1, 4+1+nc, h, w) with strides []. Returns detections (xyxy, input space).
inline std::vector<Det> yolox_detect(const std::vector<Tensor>& raw, const std::vector<int64_t>& strides,
                                     int64_t nc, float conf_thr = 0.25f, float nms_thr = 0.45f,
                                     bool logits = true) {
  std::vector<Det> cand;
  for (size_t L = 0; L < raw.size(); ++L) {
    const Tensor& t = raw[L];
    int64_t C = t->shape[1], H = t->shape[2], W = t->shape[3], hw = H * W;
    float s = (float)strides[L]; const float* d = t->data.data();
    for (int64_t i = 0; i < H; ++i) for (int64_t j = 0; j < W; ++j) {
      int64_t p = i * W + j;
      float obj = logits ? sig_(d[4 * hw + p]) : d[4 * hw + p];
      int best = 0; float bestp = -1.f;
      for (int64_t c = 0; c < nc; ++c) {
        const float pc = logits ? sig_(d[(5 + c) * hw + p]) : d[(5 + c) * hw + p];
        if (pc > bestp) { bestp = pc; best = (int)c; }
      }
      float score = obj * bestp;
      if (score < conf_thr) continue;
      float cx = (d[0 * hw + p] + j) * s, cy = (d[1 * hw + p] + i) * s;
      float w = std::exp(d[2 * hw + p]) * s, h = std::exp(d[3 * hw + p]) * s;
      cand.push_back({cx - w/2, cy - h/2, cx + w/2, cy + h/2, score, best});
    }
  }
  // class-aware greedy NMS
  std::sort(cand.begin(), cand.end(), [](const Det& a, const Det& b){ return a.score > b.score; });
  std::vector<Det> out; std::vector<char> dead(cand.size(), 0);
  auto iou = [](const Det& a, const Det& b){
    float iw = std::min(a.x2,b.x2) - std::max(a.x1,b.x1), ih = std::min(a.y2,b.y2) - std::max(a.y1,b.y1);
    if (iw <= 0 || ih <= 0) return 0.f;
    float inter = iw*ih, ua = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter;
    return inter / ua;
  };
  for (size_t a = 0; a < cand.size(); ++a) {
    if (dead[a]) continue; out.push_back(cand[a]);
    for (size_t b = a + 1; b < cand.size(); ++b)
      if (!dead[b] && cand[b].cls == cand[a].cls && iou(cand[a], cand[b]) > nms_thr) dead[b] = 1;
  }
  return out;
}
