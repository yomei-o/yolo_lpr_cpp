// `jlpr train --model det` — the C++ half of M7b: training the yolov8 nc=1 plate detector.
//
// WHY THIS LOOKS DIFFERENT FROM THE RECOGNIZER TRAINER. RESUME said the detector needed ~1,100
// lines ported from yolov8_cpp because "the v8 loss needs the raw head outputs, which live outside
// the graph". That premise is wrong for *this* export: an Ultralytics ONNX still contains the two
// per-level head convs as ordinary nodes —
//     /model.22/cv2.<l>/cv2.<l>.2/Conv_output_0   [B, 4*reg_max, H, W]   box distribution logits
//     /model.22/cv3.<l>/cv3.<l>.2/Conv_output_0   [B, nc,        H, W]   class logits
// and everything after them (Reshape/Concat/DFL/anchor decode) is only the *inference* tail. So the
// same trick the recognizer uses works here: run the graph with `stop` set to those six tensors,
// attach the loss to them, and backward() reaches every Conv weight. No second architecture, no
// weight-name mapping, no 1,100-line port — the file is the model, exactly as in onnx_train.hpp.
//
// What IS ported here is the loss itself, matching ultralytics 8.4.104 term for term:
//   * make_anchors(offset 0.5) / dist2bbox / bbox2dist            (utils/tal.py)
//   * TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)           (utils/tal.py) — including the
//     "expand a gt smaller than one stride" rule this version added
//   * CIoU exactly as bbox_iou(..., CIoU=True), alpha detached    (utils/metrics.py)
//   * BCEWithLogits on the aligned target scores, DFL, and the 7.5 / 0.5 / 1.5 gains, with the
//     final `* batch_size`                                        (utils/loss.py)
// tools/parity/train_det.py feeds both implementations the same head tensors and compares the loss
// and every gradient.
//
// THE GRADIENT IS HAND-WRITTEN, NOT COMPOSED. The engine's autograd has conv/add/mul but no
// min/max/atan/topk, and the assignment is non-differentiable anyway (Ultralytics runs it under
// no_grad). So v8_loss() is one fused node: it computes the loss and dL/d(head logits) itself and
// hands them to the tape. The only fiddly part — d(CIoU)/d(box) — is done with a 4-way forward-mode
// dual number (D4) rather than by hand, so the derivative cannot drift from the value it comes from.
#pragma once
#include "onnx_train.hpp"
#include "crop.hpp"
#include "rng.hpp"
#include "train_ocr.hpp"     // trn::read_file / list_dir / list_images_recursive (UTF-8 paths)
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace det {

// ---- forward-mode dual number over 4 inputs (the predicted box) --------------------------------
// Only used inside CIoU, where the value depends on exactly 4 variables, so carrying 4 partials is
// cheaper and far less error-prone than writing the reverse pass by hand.
struct D4 {
  float v = 0.f;
  float g[4] = {0.f, 0.f, 0.f, 0.f};
  D4() = default;
  D4(float x) : v(x) {}                                     // NOLINT: implicit on purpose (constants)
};
inline D4 dvar(float x, int i) { D4 d(x); d.g[i] = 1.f; return d; }
inline D4 operator+(const D4& a, const D4& b) { D4 r(a.v + b.v); for (int i = 0; i < 4; ++i) r.g[i] = a.g[i] + b.g[i]; return r; }
inline D4 operator-(const D4& a, const D4& b) { D4 r(a.v - b.v); for (int i = 0; i < 4; ++i) r.g[i] = a.g[i] - b.g[i]; return r; }
inline D4 operator*(const D4& a, const D4& b) { D4 r(a.v * b.v); for (int i = 0; i < 4; ++i) r.g[i] = a.g[i] * b.v + a.v * b.g[i]; return r; }
inline D4 operator/(const D4& a, const D4& b) { D4 r(a.v / b.v); for (int i = 0; i < 4; ++i) r.g[i] = (a.g[i] * b.v - a.v * b.g[i]) / (b.v * b.v); return r; }

inline float ad_min(float a, float b) { return a < b ? a : b; }
inline float ad_max(float a, float b) { return a > b ? a : b; }
inline float ad_clamp0(float a) { return a >= 0.f ? a : 0.f; }
inline float ad_atan(float a) { return std::atan(a); }
inline D4 ad_min(const D4& a, const D4& b) { return a.v < b.v ? a : b; }
inline D4 ad_max(const D4& a, const D4& b) { return a.v > b.v ? a : b; }
inline D4 ad_clamp0(const D4& a) { return a.v >= 0.f ? a : D4(0.f); }   // torch: grad passes at x >= min
inline D4 ad_atan(const D4& a) {
  D4 r(std::atan(a.v));
  const float d = 1.f / (1.f + a.v * a.v);
  for (int i = 0; i < 4; ++i) r.g[i] = a.g[i] * d;
  return r;
}
inline float ad_val(float a) { return a; }
inline float ad_val(const D4& a) { return a.v; }

// CIoU, transcribed from ultralytics.utils.metrics.bbox_iou(xywh=False, CIoU=True). `alpha` is
// computed under no_grad there, so it enters as a constant here too — differentiating it would
// change every box gradient.
template <class T>
inline T ciou(const T b1[4], const T b2[4], float eps = 1e-7f) {
  const T w1 = b1[2] - b1[0], h1 = b1[3] - b1[1] + T(eps);
  const T w2 = b2[2] - b2[0], h2 = b2[3] - b2[1] + T(eps);
  const T inter = ad_clamp0(ad_min(b1[2], b2[2]) - ad_max(b1[0], b2[0])) *
                  ad_clamp0(ad_min(b1[3], b2[3]) - ad_max(b1[1], b2[1]));
  const T uni = w1 * h1 + w2 * h2 - inter + T(eps);
  const T iou = inter / uni;
  const T cw = ad_max(b1[2], b2[2]) - ad_min(b1[0], b2[0]);
  const T ch = ad_max(b1[3], b2[3]) - ad_min(b1[1], b2[1]);
  const T c2 = cw * cw + ch * ch + T(eps);
  const T dx = b2[0] + b2[2] - b1[0] - b1[2], dy = b2[1] + b2[3] - b1[1] - b1[3];
  const T rho2 = (dx * dx + dy * dy) / T(4.f);
  const T at = ad_atan(w2 / h2) - ad_atan(w1 / h1);
  const T v = T(4.f / (3.14159265358979f * 3.14159265358979f)) * at * at;
  const float alpha = ad_val(v) / (ad_val(v) - ad_val(iou) + (1.f + eps));   // detached, as in torch
  return iou - (rho2 / c2 + v * T(alpha));
}

// ---- dataset (standard YOLO layout, what `jlpr gen-det` and Ultralytics both write) -------------
struct Box5 { int cls = 0; float cx = 0, cy = 0, w = 0, h = 0; };     // normalised
struct Item { std::string img; std::vector<Box5> boxes; };

inline std::vector<Item> read_yolo(const std::string& root) {
  std::string R = root;
  if (!R.empty() && R.back() != '/' && R.back() != '\\') R += '/';
  std::vector<std::string> imgs;
  trn::list_images_recursive(R + "images", imgs);
  std::sort(imgs.begin(), imgs.end());
  std::vector<Item> items;
  for (const std::string& p : imgs) {
    Item it;
    it.img = p;
    std::string stem = p.substr(p.find_last_of("/\\") + 1);
    stem = stem.substr(0, stem.find_last_of('.'));
    std::vector<unsigned char> blob = trn::read_file(R + "labels/" + stem + ".txt");
    std::istringstream ss(std::string(blob.begin(), blob.end()));
    std::string line;
    while (std::getline(ss, line)) {
      std::istringstream ls(line);
      Box5 b;
      if (!(ls >> b.cls >> b.cx >> b.cy >> b.w >> b.h)) continue;
      if (b.w <= 0 || b.h <= 0) continue;                 // a zero box means "no object" in v8 too
      it.boxes.push_back(b);
    }
    items.push_back(std::move(it));
  }
  return items;
}

// One batch: [B,3,S,S] RGB 0-1 (plain resize — the same preprocessing pipeline.hpp feeds a V8 graph
// at inference, which is why normalised labels survive it unchanged), plus per-image gt boxes as
// (cls, x1, y1, x2, y2) in network pixels.
struct Batch {
  Tensor x;
  std::vector<std::vector<std::array<float, 5>>> gts;
};

inline Batch make_batch(const std::vector<Item>& items, const std::vector<int>& idx, int S,
                        bool hflip_none = true) {
  (void)hflip_none;
  Batch out;
  const int64_t B = (int64_t)idx.size();
  out.x = make_tensor({B, 3, S, S}, false);
  out.gts.resize((size_t)B);
  parallel_for(B, [&](int64_t b) {                 // decode + resize per image, independently
    const Item& it = items[(size_t)idx[(size_t)b]];
    int W = 0, H = 0, C = 0;
    std::vector<unsigned char> blob = trn::read_file(it.img);
    unsigned char* im = blob.empty() ? nullptr
                                     : stbi_load_from_memory(blob.data(), (int)blob.size(), &W, &H, &C, 3);
    if (im) {
      Tensor one = jl::resize_rgb01(im, W, H, S, S);
      std::copy(one->data.begin(), one->data.end(), out.x->data.begin() + b * 3 * S * S);
      stbi_image_free(im);
    }
    for (const Box5& g : it.boxes) {
      const float cx = g.cx * S, cy = g.cy * S, w = g.w * S, h = g.h * S;
      out.gts[(size_t)b].push_back({(float)g.cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2});
    }
  });
  return out;
}

// ---- finding the head tensors in an exported graph ---------------------------------------------
// Structural, not by name: the inference tail is Reshape(box_l) -> Concat -> Reshape -> Transpose ->
// Softmax (the DFL) for the boxes, and Reshape(cls_l) -> Concat -> Sigmoid for the classes. Walking
// back from those two landmarks gives the per-level tensors in level order for any Ultralytics
// export, whatever the module index in the names happens to be.
struct HeadNames {
  std::vector<std::string> box, cls;
};

inline bool find_v8_heads(const onx::Graph& g, HeadNames& out, std::string* why = nullptr) {
  std::map<std::string, const onx::Node*> prod;
  for (const onx::Node& n : g.nodes)
    for (const std::string& o : n.output) prod[o] = &n;
  auto producer = [&](const std::string& name) -> const onx::Node* {
    auto it = prod.find(name);
    return it == prod.end() ? nullptr : it->second;
  };
  // Concat whose inputs are all Reshape -> collect the tensors those Reshapes consume.
  auto level_inputs = [&](const onx::Node* cat, std::vector<std::string>& names) {
    if (!cat || cat->op_type != "Concat") return false;
    names.clear();
    for (const std::string& in : cat->input) {
      const onx::Node* r = producer(in);
      if (!r || r->op_type != "Reshape" || r->input.empty()) return false;
      names.push_back(r->input[0]);
    }
    return !names.empty();
  };

  bool have_box = false, have_cls = false;
  for (const onx::Node& n : g.nodes) {
    if (!have_box && n.op_type == "Softmax" && !n.input.empty()) {            // the DFL softmax
      const onx::Node* tr = producer(n.input[0]);
      const onx::Node* rs = (tr && tr->op_type == "Transpose" && !tr->input.empty()) ? producer(tr->input[0]) : nullptr;
      if (rs && rs->op_type == "Reshape" && !rs->input.empty())
        have_box = level_inputs(producer(rs->input[0]), out.box);
    }
    if (!have_cls && n.op_type == "Sigmoid" && !n.input.empty())
      have_cls = level_inputs(producer(n.input[0]), out.cls);
    if (have_box && have_cls) break;
  }
  if ((!have_box || !have_cls) && why)
    *why = "could not find the per-level head tensors (box: " + std::string(have_box ? "ok" : "no") +
           ", cls: " + std::string(have_cls ? "ok" : "no") + ")";
  return have_box && have_cls && out.box.size() == out.cls.size();
}

// ---- the loss ----------------------------------------------------------------------------------
struct LossCfg {
  float box = 7.5f, cls = 0.5f, dfl = 1.5f;    // ultralytics default gains
  int topk = 10;
  float alpha = 0.5f, beta = 6.0f, eps = 1e-9f;
};
struct LossOut {
  float box = 0, cls = 0, dfl = 0, total = 0;  // box/cls/dfl are the per-term values (gain applied)
  int fg = 0;                                  // assigned anchors
  float tss = 0;                               // target_scores_sum
};

// bx[l] = [B,4*reg_max,H,W] box logits, cs[l] = [B,nc,H,W] class logits, per level (stride order).
// gts[b] = {cls, x1, y1, x2, y2} in network pixels. Returns a (1,1) scalar on the tape.
inline Tensor v8_loss(const std::vector<Tensor>& bx, const std::vector<Tensor>& cs,
                      const std::vector<float>& strides,
                      const std::vector<std::vector<std::array<float, 5>>>& gts,
                      const LossCfg& cfg, LossOut* rep = nullptr) {
  const size_t L = bx.size();
  const int64_t B = bx[0]->shape[0], nc = cs[0]->shape[1], reg = bx[0]->shape[1] / 4;
  std::vector<int64_t> Hs(L), Ws(L), off(L + 1, 0);
  for (size_t l = 0; l < L; ++l) {
    Hs[l] = bx[l]->shape[2];
    Ws[l] = bx[l]->shape[3];
    off[l + 1] = off[l] + Hs[l] * Ws[l];
  }
  const int64_t A = off[L];

  // anchors: cell centres in grid units, plus the stride that scales them to pixels
  std::vector<float> ax((size_t)A), ay((size_t)A), ast((size_t)A);
  for (size_t l = 0; l < L; ++l)
    for (int64_t y = 0; y < Hs[l]; ++y)
      for (int64_t x = 0; x < Ws[l]; ++x) {
        const size_t a = (size_t)(off[l] + y * Ws[l] + x);
        ax[a] = (float)x + 0.5f;
        ay[a] = (float)y + 0.5f;
        ast[a] = strides[l];
      }

  // ---- gather the predictions into anchor-major arrays ----
  const size_t NS = (size_t)B * A * nc, ND = (size_t)B * A * 4 * reg;
  std::vector<float> pz(NS), psig(NS), pprob(ND), pdist((size_t)B * A * 4), pbox((size_t)B * A * 4);
  for (size_t l = 0; l < L; ++l) {
    const int64_t H = Hs[l], W = Ws[l], P = H * W, Cb = 4 * reg;
    const float* bd = bx[l]->data.data();
    const float* cd = cs[l]->data.data();
    for (int64_t b = 0; b < B; ++b)
      for (int64_t p = 0; p < P; ++p) {
        const size_t a = (size_t)(off[l] + p);
        for (int64_t c = 0; c < nc; ++c) {
          const float z = cd[(b * nc + c) * P + p];
          pz[((size_t)b * A + a) * nc + c] = z;
          psig[((size_t)b * A + a) * nc + c] = 1.f / (1.f + std::exp(-z));
        }
        for (int64_t s = 0; s < 4; ++s) {
          float m = -1e30f;
          for (int64_t i = 0; i < reg; ++i) m = std::max(m, bd[(b * Cb + s * reg + i) * P + p]);
          float sum = 0.f, d = 0.f;
          float* pr = &pprob[(((size_t)b * A + a) * 4 + (size_t)s) * (size_t)reg];
          for (int64_t i = 0; i < reg; ++i) { pr[i] = std::exp(bd[(b * Cb + s * reg + i) * P + p] - m); sum += pr[i]; }
          for (int64_t i = 0; i < reg; ++i) { pr[i] /= sum; d += (float)i * pr[i]; }
          pdist[((size_t)b * A + a) * 4 + (size_t)s] = d;
        }
        float* pb = &pbox[((size_t)b * A + a) * 4];
        const float* pd = &pdist[((size_t)b * A + a) * 4];
        pb[0] = ax[a] - pd[0]; pb[1] = ay[a] - pd[1];        // dist2bbox(ltrb -> xyxy), grid units
        pb[2] = ax[a] + pd[2]; pb[3] = ay[a] + pd[3];
      }
  }

  // ---- task-aligned assignment (no gradient) ----
  std::vector<float> tscore(NS, 0.f), tbox((size_t)B * A * 4, 0.f);
  std::vector<char> fg((size_t)B * A, 0);
  double tss = 0.0;
  const float stride0 = strides[0], stride_val = strides.size() > 1 ? strides[1] : strides[0];
  for (int64_t b = 0; b < B; ++b) {
    const auto& G = gts[(size_t)b];
    const size_t ng = G.size();
    if (!ng) continue;
    std::vector<float> align(ng * (size_t)A, 0.f), ov(ng * (size_t)A, 0.f);
    std::vector<char> pos(ng * (size_t)A, 0);
    for (size_t g = 0; g < ng; ++g) {
      const float gt[4] = {G[g][1], G[g][2], G[g][3], G[g][4]};
      const int gcls = std::min((int)G[g][0], (int)nc - 1);
      // select_candidates_in_gts: a gt thinner than the finest stride is widened to stride[1] first
      float cx = (gt[0] + gt[2]) / 2, cy = (gt[1] + gt[3]) / 2;
      float w = gt[2] - gt[0], h = gt[3] - gt[1];
      if (w < stride0) w = stride_val;
      if (h < stride0) h = stride_val;
      const float ex[4] = {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2};
      std::vector<std::pair<float, int>> cand;
      for (int64_t a = 0; a < A; ++a) {
        const float px = ax[(size_t)a] * ast[(size_t)a], py = ay[(size_t)a] * ast[(size_t)a];
        if (!(px - ex[0] > 1e-9f && ex[2] - px > 1e-9f && py - ex[1] > 1e-9f && ex[3] - py > 1e-9f)) continue;
        const float pb[4] = {pbox[((size_t)b * A + a) * 4 + 0] * ast[(size_t)a],
                             pbox[((size_t)b * A + a) * 4 + 1] * ast[(size_t)a],
                             pbox[((size_t)b * A + a) * 4 + 2] * ast[(size_t)a],
                             pbox[((size_t)b * A + a) * 4 + 3] * ast[(size_t)a]};
        const float o = std::max(0.f, ciou<float>(gt, pb));            // box1 = gt, as in the assigner
        const float sc = psig[((size_t)b * A + a) * nc + gcls];
        ov[g * (size_t)A + a] = o;
        align[g * (size_t)A + a] = std::pow(sc, cfg.alpha) * std::pow(o, cfg.beta);
        cand.push_back({align[g * (size_t)A + a], (int)a});
      }
      std::stable_sort(cand.begin(), cand.end(),
                       [](const std::pair<float, int>& p, const std::pair<float, int>& q) { return p.first > q.first; });
      for (size_t k = 0; k < cand.size() && k < (size_t)cfg.topk; ++k) pos[g * (size_t)A + cand[k].second] = 1;
    }
    // select_highest_overlaps: an anchor claimed by several gts goes to the one it overlaps most
    for (int64_t a = 0; a < A; ++a) {
      int n = 0;
      for (size_t g = 0; g < ng; ++g) n += pos[g * (size_t)A + a];
      if (n <= 1) continue;
      size_t best = 0;
      for (size_t g = 1; g < ng; ++g) if (ov[g * (size_t)A + a] > ov[best * (size_t)A + a]) best = g;
      for (size_t g = 0; g < ng; ++g) pos[g * (size_t)A + a] = (g == best) ? 1 : 0;
    }
    // normalised target score: per gt, max align and max overlap over its positives
    std::vector<float> amax(ng, 0.f), omax(ng, 0.f);
    for (size_t g = 0; g < ng; ++g)
      for (int64_t a = 0; a < A; ++a)
        if (pos[g * (size_t)A + a]) {
          amax[g] = std::max(amax[g], align[g * (size_t)A + a]);
          omax[g] = std::max(omax[g], ov[g * (size_t)A + a]);
        }
    for (int64_t a = 0; a < A; ++a) {
      float norm = 0.f;
      int gsel = -1;
      for (size_t g = 0; g < ng; ++g) {
        if (!pos[g * (size_t)A + a]) continue;
        if (gsel < 0) gsel = (int)g;                                   // mask_pos.argmax(-2): first hit
        norm = std::max(norm, align[g * (size_t)A + a] * omax[g] / (amax[g] + cfg.eps));
      }
      if (gsel < 0) continue;
      fg[(size_t)b * A + a] = 1;
      const int gcls = std::min((int)G[(size_t)gsel][0], (int)nc - 1);
      tscore[((size_t)b * A + a) * nc + gcls] = norm;
      tss += norm;
      for (int k = 0; k < 4; ++k) tbox[((size_t)b * A + a) * 4 + k] = G[(size_t)gsel][k + 1];
    }
  }
  const float tsum = std::max(1.f, (float)tss);

  // ---- losses and their gradients ----
  auto gz = std::make_shared<std::vector<float>>(NS, 0.f);       // dL/d class logit
  auto gd = std::make_shared<std::vector<float>>(ND, 0.f);       // dL/d box distribution logit
  const float sc_cls = cfg.cls * (float)B / tsum;
  const float sc_box = cfg.box * (float)B / tsum;
  const float sc_dfl = cfg.dfl * (float)B / tsum;

  double cls_sum = 0.0;
  for (size_t i = 0; i < NS; ++i) {
    const float z = pz[i], t = tscore[i];
    cls_sum += std::max(z, 0.f) - z * t + std::log1p(std::exp(-std::fabs(z)));   // BCEWithLogits
    (*gz)[i] = (psig[i] - t) * sc_cls;
  }

  double box_sum = 0.0, dfl_sum = 0.0;
  int nfg = 0;
  const float dmax = (float)(reg - 1) - 0.01f;
  for (int64_t b = 0; b < B; ++b)
    for (int64_t a = 0; a < A; ++a) {
      if (!fg[(size_t)b * A + a]) continue;
      ++nfg;
      float w = 0.f;
      for (int64_t c = 0; c < nc; ++c) w += tscore[((size_t)b * A + a) * nc + c];
      const float st = ast[(size_t)a];
      const D4 pd4[4] = {dvar(pbox[((size_t)b * A + a) * 4 + 0], 0), dvar(pbox[((size_t)b * A + a) * 4 + 1], 1),
                         dvar(pbox[((size_t)b * A + a) * 4 + 2], 2), dvar(pbox[((size_t)b * A + a) * 4 + 3], 3)};
      const D4 tb4[4] = {D4(tbox[((size_t)b * A + a) * 4 + 0] / st), D4(tbox[((size_t)b * A + a) * 4 + 1] / st),
                         D4(tbox[((size_t)b * A + a) * 4 + 2] / st), D4(tbox[((size_t)b * A + a) * 4 + 3] / st)};
      const D4 c = ciou<D4>(pd4, tb4);                        // box1 = pred: the gradient goes here
      box_sum += (1.0 - c.v) * w;
      // (1-ciou)*w -> the four box coordinates -> the four ltrb distances
      const float dl = c.g[0] * w * sc_box, dt = c.g[1] * w * sc_box;   // sign: d(1-ciou) = -dciou,
      const float dr = -c.g[2] * w * sc_box, db = -c.g[3] * w * sc_box; // and x1 = ax - l, x2 = ax + r
      const float dside[4] = {dl, dt, dr, db};
      // DFL target: bbox2dist(anchor, target) clamped to [0, reg_max-1-0.01]
      const float tl_[4] = {ax[(size_t)a] - tb4[0].v, ay[(size_t)a] - tb4[1].v,
                            tb4[2].v - ax[(size_t)a], tb4[3].v - ay[(size_t)a]};
      for (int s = 0; s < 4; ++s) {
        float* pr = &pprob[(((size_t)b * A + a) * 4 + (size_t)s) * (size_t)reg];
        float* g = &(*gd)[(((size_t)b * A + a) * 4 + (size_t)s) * (size_t)reg];
        const float dist = pdist[((size_t)b * A + a) * 4 + (size_t)s];
        // CIoU path: d(dist)/d(logit_i) = p_i * (i - dist)
        for (int64_t i = 0; i < reg; ++i) g[i] += dside[s] * pr[i] * ((float)i - dist);
        // DFL path: cross-entropy against the two neighbouring bins
        const float tv = std::min(std::max(tl_[s], 0.f), dmax);
        const int il = (int)tv, ir = il + 1;
        const float wl = (float)ir - tv, wr = 1.f - wl;
        dfl_sum += -(std::log(std::max(pr[il], 1e-30f)) * wl +
                     std::log(std::max(pr[std::min<int>(ir, (int)reg - 1)], 1e-30f)) * wr) * w / 4.0;
        for (int64_t i = 0; i < reg; ++i)
          g[i] += sc_dfl * w * 0.25f * ((pr[i] - (i == il ? 1.f : 0.f)) * wl +
                                        (pr[i] - (i == ir ? 1.f : 0.f)) * wr);
      }
    }

  const float lbox = cfg.box * (float)(box_sum / tsum);
  const float lcls = cfg.cls * (float)(cls_sum / tsum);
  const float ldfl = cfg.dfl * (float)(dfl_sum / tsum);
  if (rep) { rep->box = lbox; rep->cls = lcls; rep->dfl = ldfl; rep->total = (lbox + lcls + ldfl) * B;
             rep->fg = nfg; rep->tss = tsum; }

  Tensor out = make_tensor({1, 1}, true);
  out->data[0] = (lbox + lcls + ldfl) * (float)B;                       // ultralytics: loss * batch
  out->parents = bx;
  for (const Tensor& t : cs) out->parents.push_back(t);
  Node* op = out.get();
  out->backward_fn = [bx, cs, gz, gd, op, B, A, nc, reg, Hs, Ws, off, L] {
    const float s = op->grad[0];
    for (size_t l = 0; l < L; ++l) {
      const int64_t P = Hs[l] * Ws[l], Cb = 4 * reg;
      float* gb = bx[l]->grad.data();
      float* gc = cs[l]->grad.data();
      for (int64_t b = 0; b < B; ++b)
        for (int64_t p = 0; p < P; ++p) {
          const size_t a = (size_t)(off[l] + p);
          for (int64_t c = 0; c < nc; ++c) gc[(b * nc + c) * P + p] += s * (*gz)[((size_t)b * A + a) * nc + c];
          for (int64_t c = 0; c < Cb; ++c) gb[(b * Cb + c) * P + p] += s * (*gd)[((size_t)b * A + a) * (size_t)Cb + c];
        }
    }
  };
  return out;
}

// One training step's forward: run the graph up to the six head tensors and build the loss.
inline Tensor forward_loss(onx::Trainable& t, const HeadNames& hn, const Tensor& x,
                           const std::vector<std::vector<std::array<float, 5>>>& gts,
                           int imgsz, const LossCfg& cfg, LossOut* rep,
                           std::vector<Tensor>* bx_out = nullptr,
                           std::vector<Tensor>* cs_out = nullptr) {
  std::set<std::string> stop(hn.box.begin(), hn.box.end());
  stop.insert(hn.cls.begin(), hn.cls.end());
  std::map<std::string, Tensor> vals = onx::run_onnx(t.g, x, stop, &t.init, false);
  std::vector<Tensor> bx, cs;
  std::vector<float> strides;
  for (size_t l = 0; l < hn.box.size(); ++l) {
    bx.push_back(vals.at(hn.box[l]));
    cs.push_back(vals.at(hn.cls[l]));
    strides.push_back((float)imgsz / (float)bx.back()->shape[2]);
  }
  if (bx_out) *bx_out = bx;
  if (cs_out) *cs_out = cs;
  return v8_loss(bx, cs, strides, gts, cfg, rep);
}

}  // namespace det
