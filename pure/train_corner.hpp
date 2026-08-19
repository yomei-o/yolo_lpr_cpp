// `jlpr train --model corner` — the C++ half of M6, and the last hole in the Python/C++ parity table.
//
// Mirrors tools/train_corner.py step for step, including **the order of the random draws** (item
// index, then four box-jitter values, then the expansion), so both languages asked for the same seed
// see the same batch and their losses and gradients are comparable numerically, not just in shape.
//
// The model is trained the same way everything else in this repo is: **the ONNX file is the model**
// (pure/onnx_train.hpp). Two differences from the recognizer trainer:
//
//   * BatchNorm runs in **training mode** here (run_onnx(..., bn_training=true)): the corner net can
//     be trained from scratch, and a from-scratch net whose BN uses the shipped running statistics
//     learns nothing. The running stats are buffers — updated in place, saved by write_back().
//   * `--init random` builds a fresh CornerNet graph in C++ (PyTorch's default init: uniform
//     ±1/sqrt(fan_in) for conv/linear, gamma 1 / beta 0 / mean 0 / var 1 for BN), so the C++ lane
//     does not need Python to produce a starting file.
//
// Data comes free from the generator: `jlpr gen` writes corners.txt next to every crop
// (`<file> x1 y1 .. x4 y4 <font>`, pixels). Each sample is built the way inference will see it —
// box = bbox(corners), jittered, expanded, resampled to 64x64 — and the target is the 4 corners in
// that crop's normalised frame.
#pragma once
#include "onnx_train.hpp"
#include "rng.hpp"
#include "train_ocr.hpp"      // trn::read_file / list_dir (UTF-8 paths)
#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace crn {

constexpr int IN_PX = 64;
constexpr float BOX_EXPAND = 0.25f;      // tools/corner_model.py BOX_EXPAND — the inference framing

struct Item {
  std::string path;
  float c[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // TL,TR,BR,BL in pixels of that crop image
};

inline std::vector<Item> read_corners(const std::string& root) {
  std::string R = root;
  if (!R.empty() && R.back() != '/' && R.back() != '\\') R += '/';
  std::vector<unsigned char> blob = trn::read_file(R + "corners.txt");
  std::istringstream ss(std::string(blob.begin(), blob.end()));
  std::string line;
  std::vector<Item> out;
  while (std::getline(ss, line)) {
    std::istringstream ls(line);
    std::string name;
    Item it;
    if (!(ls >> name)) continue;
    bool ok = true;
    for (int i = 0; i < 8; ++i) if (!(ls >> it.c[i])) { ok = false; break; }
    if (!ok) continue;
    it.path = R + name;
    out.push_back(std::move(it));
  }
  return out;
}

// Bilinear sample with edge clamp — the same expression as tools/infer.py `_sample`.
inline float sample_px(const unsigned char* rgb, int W, int H, float sx, float sy, int c) {
  const int xi = (int)std::floor(sx), yi = (int)std::floor(sy);
  const float fx = sx - xi, fy = sy - yi;
  auto px = [&](int yy, int xx) {
    yy = std::min(std::max(yy, 0), H - 1);
    xx = std::min(std::max(xx, 0), W - 1);
    return (float)rgb[((size_t)yy * W + xx) * 3 + c];
  };
  return px(yi, xi) * (1 - fx) * (1 - fy) + px(yi, xi + 1) * fx * (1 - fy)
       + px(yi + 1, xi) * (1 - fx) * fy + px(yi + 1, xi + 1) * fx * fy;
}

// One crop + its 8 targets. `jit` = the four box-jitter fractions (nullptr for none).
inline void sample_crop(const unsigned char* rgb, int W, int H, const float c[8], const float* jit,
                        float expand, float* out_x, float* out_t) {
  float x0 = c[0], x1 = c[0], y0 = c[1], y1 = c[1];
  for (int k = 1; k < 4; ++k) {
    x0 = std::min(x0, c[2 * k]); x1 = std::max(x1, c[2 * k]);
    y0 = std::min(y0, c[2 * k + 1]); y1 = std::max(y1, c[2 * k + 1]);
  }
  float bw = x1 - x0, bh = y1 - y0;
  if (jit) {
    x0 += bw * jit[0]; x1 += bw * jit[1];
    y0 += bh * jit[2]; y1 += bh * jit[3];
    bw = std::max(1e-3f, x1 - x0);
    bh = std::max(1e-3f, y1 - y0);
  }
  const float cx0 = x0 - bw * expand, cy0 = y0 - bh * expand;
  const float cx1 = x1 + bw * expand, cy1 = y1 + bh * expand;
  for (int y = 0; y < IN_PX; ++y) {
    const float sy = cy0 + (y + 0.5f) * (cy1 - cy0) / IN_PX - 0.5f;
    for (int x = 0; x < IN_PX; ++x) {
      const float sx = cx0 + (x + 0.5f) * (cx1 - cx0) / IN_PX - 0.5f;
      for (int ch = 0; ch < 3; ++ch)
        out_x[(size_t)(ch * IN_PX + y) * IN_PX + x] = sample_px(rgb, W, H, sx, sy, ch) / 255.f;
    }
  }
  for (int k = 0; k < 4; ++k) {
    out_t[2 * k] = (c[2 * k] - cx0) / (cx1 - cx0);
    out_t[2 * k + 1] = (c[2 * k + 1] - cy0) / (cy1 - cy0);
  }
}

struct Batch {
  Tensor x;
  std::vector<float> y;      // B*8
};

// Draw order (the contract with tools/train_corner.py): per sample, item index, then 4 jitter
// values, then the expansion. All draws happen before any image is touched, exactly as there.
inline Batch make_batch(const std::vector<Item>& items, Rng& rng, int batch, float jitter,
                        float expand_lo, float expand_hi) {
  struct Plan { int i; float j[4]; float e; };
  std::vector<Plan> plan((size_t)batch);
  for (int b = 0; b < batch; ++b) {
    Plan p{};
    p.i = (int)rng.below((uint64_t)items.size());
    if (jitter > 0) for (int k = 0; k < 4; ++k) p.j[k] = (float)rng.range(-jitter, jitter);
    p.e = (float)rng.range(expand_lo, expand_hi);
    plan[(size_t)b] = p;
  }
  Batch out;
  out.x = make_tensor({batch, 3, IN_PX, IN_PX}, false);
  out.y.assign((size_t)batch * 8, 0.f);
  // Decoding a PNG per sample dominates a step of this small net, and the draws are already done, so
  // the loads can run concurrently without changing a single number (the Python side has the same
  // split, with --workers).
  parallel_for(batch, [&](int64_t b) {
    const Item& it = items[(size_t)plan[(size_t)b].i];
    int W = 0, H = 0, C = 0;
    std::vector<unsigned char> blob = trn::read_file(it.path);
    unsigned char* im = blob.empty() ? nullptr
                                     : stbi_load_from_memory(blob.data(), (int)blob.size(), &W, &H, &C, 3);
    if (!im) return;
    sample_crop(im, W, H, it.c, jitter > 0 ? plan[(size_t)b].j : nullptr, plan[(size_t)b].e,
                out.x->data.data() + (size_t)b * 3 * IN_PX * IN_PX, out.y.data() + (size_t)b * 8);
    stbi_image_free(im);
  });
  return out;
}

// Smooth L1 (Huber), mean reduction — torch.nn.functional.smooth_l1_loss(beta=...). One fused node:
// the value and dL/dpred are two lines each, and composing it out of engine ops would cost a dozen
// tensors per step for nothing.
inline Tensor smooth_l1(const Tensor& pred, const std::vector<float>& target, float beta) {
  const int64_t n = pred->numel();
  auto grad = std::make_shared<std::vector<float>>((size_t)n, 0.f);
  double sum = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const float d = pred->data[(size_t)i] - target[(size_t)i], a = std::fabs(d);
    sum += a < beta ? 0.5 * d * d / beta : a - 0.5 * beta;
    (*grad)[(size_t)i] = (a < beta ? d / beta : (d > 0 ? 1.f : -1.f)) / (float)n;
  }
  Tensor out = make_tensor({1, 1}, true);
  out->data[0] = (float)(sum / n);
  out->parents = {pred};
  Node* op = out.get();
  out->backward_fn = [pred, grad, op, n] {
    const float s = op->grad[0];
    for (int64_t i = 0; i < n; ++i) pred->grad[(size_t)i] += s * (*grad)[(size_t)i];
  };
  return out;
}

// torch.nn.utils.clip_grad_norm_ — run 2 of the Python trainer diverged to 240% without it.
inline float clip_grad_norm(std::vector<Tensor>& params, float max_norm) {
  double sq = 0.0;
  for (const Tensor& p : params) for (float g : p->grad) sq += (double)g * g;
  const float total = (float)std::sqrt(sq);
  if (total > max_norm) {
    const float s = max_norm / (total + 1e-6f);
    for (Tensor& p : params) for (float& g : p->grad) g *= s;
  }
  return total;
}

// Mean corner error as a fraction of the plate's width — the number that matters downstream
// (1.5% of a 330 mm plate is 5 mm, under a glyph stroke). Same fixed framing and seed as the
// Python evaluator, so the two report the same quantity on the same data.
inline double eval_error(onx::Trainable& t, const std::vector<Item>& items, int n = 400,
                        uint64_t seed = 9) {
  Rng rng(seed);
  const float px_w = 1.f / (1.f + 2 * BOX_EXPAND);
  double acc = 0.0;
  int batches = std::max(1, n / 32);
  for (int b = 0; b < batches; ++b) {
    Batch ba = make_batch(items, rng, 32, 0.04f, BOX_EXPAND, BOX_EXPAND);
    std::map<std::string, Tensor> vals = onx::run_onnx(t.g, ba.x, {}, &t.init, false);
    const Tensor& p = vals.at(t.g.outputs[0].name);
    double s = 0.0;
    for (int i = 0; i < 32; ++i)
      for (int k = 0; k < 4; ++k) {
        const float dx = p->data[(size_t)i * 8 + 2 * k] - ba.y[(size_t)i * 8 + 2 * k];
        const float dy = p->data[(size_t)i * 8 + 2 * k + 1] - ba.y[(size_t)i * 8 + 2 * k + 1];
        s += std::sqrt((double)dx * dx + (double)dy * dy) / px_w;
      }
    acc += s / (32 * 4);
    free_graph(p);
  }
  return acc / batches;
}

// ---- a fresh CornerNet, built here so the C++ lane needs no Python to start -------------------
// Same graph tools/corner_model.py exports (Conv3x3 s2 -> Relu -> BN, four times, Flatten, Gemm)
// and the same initializer names, so a file from either side loads in both.
inline onx::Graph build_graph(int width, uint64_t seed) {
  Rng rng(seed);
  onx::Graph g;
  g.opset = 13;
  g.inputs.push_back({"input", {1, 3, IN_PX, IN_PX}});
  auto uniform = [&](int64_t n, float bound) {
    std::vector<float> v((size_t)n);
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (float)rng.range(-bound, bound);
    return v;
  };
  auto node = [&](const std::string& op, const std::vector<std::string>& in, const std::string& out,
                  const std::vector<onx::Attr>& attr = {}) {
    onx::Node n;
    n.op_type = op; n.name = out; n.input = in; n.output = {out}; n.attr = attr;
    g.nodes.push_back(std::move(n));
  };
  const int chans[4] = {width, width * 2, width * 3, width * 4};
  std::string x = "input";
  int cin = 3;
  for (int k = 0; k < 4; ++k) {
    const int cout = chans[k];
    const std::string pre = "body." + std::to_string(k) + ".";
    // torch's Conv2d/Linear default: U(-1/sqrt(fan_in), +1/sqrt(fan_in)) for both weight and bias
    const float bound = 1.f / std::sqrt((float)(cin * 3 * 3));
    g.init_f.push_back({pre + "conv.weight", {cout, cin, 3, 3}, uniform((int64_t)cout * cin * 9, bound)});
    g.init_f.push_back({pre + "conv.bias", {cout}, uniform(cout, bound)});
    node("Conv", {x, pre + "conv.weight", pre + "conv.bias"}, "/" + pre + "conv",
         {{"kernel_shape", onx::A_INTS, 0, 0, "", {3, 3}, {}},
          {"strides", onx::A_INTS, 0, 0, "", {2, 2}, {}},
          {"pads", onx::A_INTS, 0, 0, "", {1, 1, 1, 1}, {}},
          {"group", onx::A_INT, 1, 0, "", {}, {}}});
    node("Relu", {"/" + pre + "conv"}, "/" + pre + "relu");
    g.init_f.push_back({pre + "bn.weight", {cout}, std::vector<float>((size_t)cout, 1.f)});
    g.init_f.push_back({pre + "bn.bias", {cout}, std::vector<float>((size_t)cout, 0.f)});
    g.init_f.push_back({pre + "bn.running_mean", {cout}, std::vector<float>((size_t)cout, 0.f)});
    g.init_f.push_back({pre + "bn.running_var", {cout}, std::vector<float>((size_t)cout, 1.f)});
    node("BatchNormalization",
         {"/" + pre + "relu", pre + "bn.weight", pre + "bn.bias", pre + "bn.running_mean",
          pre + "bn.running_var"},
         "/" + pre + "bn", {{"epsilon", onx::A_FLOAT, 0, 1e-3f, "", {}, {}}});
    x = "/" + pre + "bn";
    cin = cout;
  }
  node("Flatten", {x}, "/flat", {{"axis", onx::A_INT, 1, 0, "", {}, {}}});
  const int64_t feat = (int64_t)chans[3] * 4 * 4;
  const float hb = 1.f / std::sqrt((float)feat);
  g.init_f.push_back({"head.weight", {8, feat}, uniform(8 * feat, hb)});   // Gemm transB=1, as torch exports
  g.init_f.push_back({"head.bias", {8}, uniform(8, hb)});
  node("Gemm", {"/flat", "head.weight", "head.bias"}, "corners",
       {{"alpha", onx::A_FLOAT, 0, 1.f, "", {}, {}},
        {"beta", onx::A_FLOAT, 0, 1.f, "", {}, {}},
        {"transB", onx::A_INT, 1, 0, "", {}, {}}});
  g.outputs.push_back({"corners", {1, 8}});
  return g;
}

}  // namespace crn
