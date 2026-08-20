// Build a yolov8 detector graph from nothing but a size spec — the C++ side of "where does the
// starting model come from".
//
// Until now `jlpr train --model det` could only *continue* from an ONNX somebody else exported, so the
// one step of the pipeline that still required Python was the first one. This writes the graph itself:
//
//   jlpr init-det --arch n --nc 1 --imgsz 320 --out models/scratch_320.onnx          # random init
//   jlpr init-det --arch n --nc 1 --imgsz 320 --from-pt yolov8n.pt --out models/x.onnx   # transfer
//
// Two things make this more than a toy:
//
//   * The node topology is the one `det::find_v8_heads` looks for and the one pure/infer_v8.hpp can
//     decode — box heads reshaped and concatenated across levels into the DFL softmax, class heads
//     reshaped and concatenated into a Sigmoid, output [1, 4+nc, anchors] in cxcywh. That is what
//     Ultralytics' `export(simplify=True)` produces, so the same file trains here and runs in the
//     WASM demo without a conversion step.
//   * Parameter names are Ultralytics' state_dict names (`model.0.conv.weight`,
//     `model.22.cv3.2.2.bias`, ...). That is what lets `--from-pt` drop a real yolov8n.pt onto this
//     graph tensor by tensor, and it is also why `--freeze N` (which parses `model.<n>.`) keeps
//     working on a graph we generated ourselves.
//
// Initialisation follows torch: Conv2d's default kaiming_uniform(a=sqrt(5)) reduces to
// U(-1/sqrt(fan_in), +1/sqrt(fan_in)), BN starts at weight 1 / bias 0 / mean 0 / var 1, and the
// detect head gets Ultralytics' bias_init — box bias 1.0, class bias log(5 / nc / (640/stride)^2).
// The class bias is not cosmetic: without it every anchor starts out claiming an object and the first
// hundred steps are spent undoing that (measured on the recogniser's appended classes, same lesson).
#pragma once
#include "onnx.hpp"
#include "ptio.hpp"
#include "rng.hpp"
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace mkdet {

struct Spec {
  char arch = 'n';        // n / s / m / l / x — sets width, depth and the channel cap
  int nc = 1;
  int imgsz = 320;
  int reg = 16;           // DFL bins (yolov8 uses 16 everywhere)
  uint64_t seed = 1234;
};

struct Sizes { float width, depth; int max_ch; };

inline Sizes sizes_of(char arch) {
  switch (arch) {
    case 's': return {0.50f, 0.33f, 1024};
    case 'm': return {0.75f, 0.67f, 768};
    case 'l': return {1.00f, 1.00f, 512};
    case 'x': return {1.25f, 1.00f, 512};
    default:  return {0.25f, 0.33f, 1024};      // 'n'
  }
}

// Ultralytics' make_divisible(min(c, max) * width, 8)
inline int ch_of(int base, const Sizes& s) {
  const int c = (int)std::round(std::min(base, s.max_ch) * s.width);
  return std::max(8, ((c + 4) / 8) * 8);
}
inline int rep_of(int base, const Sizes& s) {
  return std::max(1, (int)std::round(base * s.depth));
}

// ------------------------------------------------------------------ builder

class Builder {
 public:
  Builder(const Spec& sp, const std::map<std::string, pt::Tensor>* src)
      : sp_(sp), sz_(sizes_of(sp.arch)), rng_(sp.seed), src_(src) {}

  // how many tensors came from the .pt vs were initialised here
  int taken = 0, made = 0;
  std::vector<std::string> missed;   // names --from-pt did not provide (the cls head, normally)

  onx::Graph build() {
    g_.opset = 13;
    g_.inputs.push_back({"images", {1, 3, sp_.imgsz, sp_.imgsz}});

    const int c1 = ch_of(64, sz_), c2 = ch_of(128, sz_), c3 = ch_of(256, sz_),
              c4 = ch_of(512, sz_), c5 = ch_of(1024, sz_);
    std::string x = "images";
    x = conv(x, "model.0", 3, c1, 3, 2);
    x = conv(x, "model.1", c1, c2, 3, 2);
    x = c2f(x, "model.2", c2, c2, rep_of(3, sz_), true);
    x = conv(x, "model.3", c2, c3, 3, 2);
    x = c2f(x, "model.4", c3, c3, rep_of(6, sz_), true);
    const std::string p3 = x;
    int p3c = c3;
    x = conv(x, "model.5", c3, c4, 3, 2);
    x = c2f(x, "model.6", c4, c4, rep_of(6, sz_), true);
    const std::string p4 = x;
    int p4c = c4;
    x = conv(x, "model.7", c4, c5, 3, 2);
    x = c2f(x, "model.8", c5, c5, rep_of(3, sz_), true);
    x = sppf(x, "model.9", c5, c5);
    const std::string p5 = x;
    int p5c = c5;

    // neck: two upsample+concat+C2f going down, two stride-2+concat+C2f coming back up
    std::string u = upsample(p5, "up1");
    std::string cat = concat({u, p4}, "cat1", 1);
    const std::string n4 = c2f(cat, "model.12", p5c + p4c, c4, rep_of(3, sz_), false);
    u = upsample(n4, "up2");
    cat = concat({u, p3}, "cat2", 1);
    const std::string h3 = c2f(cat, "model.15", c4 + p3c, c3, rep_of(3, sz_), false);   // P3 head in
    std::string d = conv(h3, "model.16", c3, c3, 3, 2);
    cat = concat({d, n4}, "cat3", 1);
    const std::string h4 = c2f(cat, "model.18", c3 + c4, c4, rep_of(3, sz_), false);    // P4 head in
    d = conv(h4, "model.19", c4, c4, 3, 2);
    cat = concat({d, p5}, "cat4", 1);
    const std::string h5 = c2f(cat, "model.21", c4 + p5c, c5, rep_of(3, sz_), false);   // P5 head in

    detect({h3, h4, h5}, {c3, c4, c5});
    return g_;
  }

 private:
  Spec sp_;
  Sizes sz_;
  Rng rng_;
  const std::map<std::string, pt::Tensor>* src_;
  onx::Graph g_;
  int uid_ = 0;

  std::string uniq(const std::string& base) { return base + "_" + std::to_string(uid_++); }

  void add_node(const std::string& op, const std::vector<std::string>& in,
                const std::string& out, const std::vector<onx::Attr>& attr = {}) {
    onx::Node n;
    n.op_type = op;
    n.name = out;
    n.input = in;
    n.output = {out};
    n.attr = attr;
    g_.nodes.push_back(std::move(n));
  }
  static onx::Attr ai(const std::string& name, int64_t v) {
    onx::Attr a; a.name = name; a.type = onx::A_INT; a.i = v; return a;
  }
  static onx::Attr ais(const std::string& name, std::vector<int64_t> v) {
    onx::Attr a; a.name = name; a.type = onx::A_INTS; a.ints = std::move(v); return a;
  }
  static onx::Attr as(const std::string& name, const std::string& v) {
    onx::Attr a; a.name = name; a.type = onx::A_STRING; a.s = v; return a;
  }
  static onx::Attr af(const std::string& name, float v) {
    onx::Attr a; a.name = name; a.type = onx::A_FLOAT; a.f = v; return a;
  }

  void init_f(const std::string& name, std::vector<int64_t> dims, std::vector<float> data) {
    onx::Tensor64 t;
    t.name = name;
    t.dims = std::move(dims);
    t.data = std::move(data);
    g_.init_f.push_back(std::move(t));
  }
  void init_i(const std::string& name, std::vector<int64_t> dims, std::vector<int64_t> data) {
    onx::IntsTensor t;
    t.name = name;
    t.dims = std::move(dims);
    t.data = std::move(data);
    g_.init_i.push_back(std::move(t));
  }

  // A weight: from the .pt when it has one of that exact shape, otherwise fresh. Shape is checked,
  // not assumed — a 80-class checkpoint's cls head must NOT land on an nc=1 graph.
  void weight(const std::string& name, const std::vector<int64_t>& dims, float fill, bool uniform) {
    int64_t n = 1;
    for (int64_t d : dims) n *= d;
    if (src_) {
      auto it = src_->find(name);
      if (it != src_->end()) {
        bool same = it->second.shape.size() == dims.size();
        for (size_t i = 0; same && i < dims.size(); ++i) same = it->second.shape[i] == dims[i];
        if (same && (int64_t)it->second.data.size() == n) {
          init_f(name, dims, it->second.data);
          ++taken;
          return;
        }
      }
      missed.push_back(name);
    }
    std::vector<float> v((size_t)n, fill);
    if (uniform) {
      // torch's Conv2d default: U(-1/sqrt(fan_in), +1/sqrt(fan_in)), fan_in = cin*k*k
      int64_t fan_in = 1;
      for (size_t i = 1; i < dims.size(); ++i) fan_in *= dims[i];
      const float b = 1.f / std::sqrt((float)std::max<int64_t>(1, fan_in));
      for (float& q : v) q = (float)rng_.range(-b, b);
    }
    init_f(name, dims, v);
    ++made;
  }

  // Conv (no bias) + BatchNormalization + SiLU, named as Ultralytics names them
  std::string conv(const std::string& in, const std::string& mod, int cin, int cout, int k, int s,
                   bool act = true) {
    const int pad = k / 2;
    weight(mod + ".conv.weight", {cout, cin, k, k}, 0.f, true);
    weight(mod + ".bn.weight", {cout}, 1.f, false);
    weight(mod + ".bn.bias", {cout}, 0.f, false);
    weight(mod + ".bn.running_mean", {cout}, 0.f, false);
    weight(mod + ".bn.running_var", {cout}, 1.f, false);
    const std::string c = uniq(mod + "/conv");
    add_node("Conv", {in, mod + ".conv.weight"}, c,
             {ais("kernel_shape", {k, k}), ais("strides", {s, s}), ais("pads", {pad, pad, pad, pad})});
    const std::string b = uniq(mod + "/bn");
    add_node("BatchNormalization",
             {c, mod + ".bn.weight", mod + ".bn.bias", mod + ".bn.running_mean", mod + ".bn.running_var"},
             b, {af("epsilon", 1e-3f), af("momentum", 0.97f)});
    if (!act) return b;
    const std::string sg = uniq(mod + "/sig");
    add_node("Sigmoid", {b}, sg);
    const std::string m = uniq(mod + "/silu");
    add_node("Mul", {b, sg}, m);
    return m;
  }

  std::string concat(const std::vector<std::string>& ins, const std::string& tag, int64_t axis) {
    const std::string o = uniq(tag);
    add_node("Concat", ins, o, {ai("axis", axis)});
    return o;
  }

  std::string upsample(const std::string& in, const std::string& tag) {
    // Resize with scales — the same node an exported yolov8 carries, and the one onnx_run reads with
    // llround() rather than a truncating cast (see the weight-decay bug in RESUME).
    const std::string sc = uniq(tag + "/scales");
    init_f(sc, {4}, {1.f, 1.f, 2.f, 2.f});
    const std::string roi = uniq(tag + "/roi");
    init_f(roi, {0}, {});
    const std::string o = uniq(tag);
    add_node("Resize", {in, roi, sc}, o,
             {as("mode", "nearest"), as("coordinate_transformation_mode", "asymmetric"),
              as("nearest_mode", "floor")});
    return o;
  }

  std::string bottleneck(const std::string& in, const std::string& mod, int c, bool shortcut) {
    std::string y = conv(in, mod + ".cv1", c, c, 3, 1);
    y = conv(y, mod + ".cv2", c, c, 3, 1);
    if (!shortcut) return y;
    const std::string o = uniq(mod + "/add");
    add_node("Add", {in, y}, o);
    return o;
  }

  std::string c2f(const std::string& in, const std::string& mod, int cin, int cout, int n,
                  bool shortcut) {
    const int c = cout / 2;
    const std::string y = conv(in, mod + ".cv1", cin, 2 * c, 1, 1);
    // Split into the two halves. Ultralytics uses chunk(2, 1); Split with equal parts is the same and
    // is what the exported graph carries.
    const std::string s0 = uniq(mod + "/split0");
    const std::string s1 = uniq(mod + "/split1");
    // opset 13 moved Split's sizes from an attribute to an input. Our interpreter reads either form,
    // but onnx.checker rejects the attribute at 13 — and a file only we can read is not an ONNX file.
    const std::string spl = uniq(mod + "/splitsz");
    init_i(spl, {2}, {c, c});
    onx::Node sp;
    sp.op_type = "Split";
    sp.name = s0;
    sp.input = {y, spl};
    sp.output = {s0, s1};
    sp.attr = {ai("axis", 1)};
    g_.nodes.push_back(std::move(sp));

    std::vector<std::string> keep = {s0, s1};
    std::string last = s1;
    for (int i = 0; i < n; ++i) {
      last = bottleneck(last, mod + ".m." + std::to_string(i), c, shortcut);
      keep.push_back(last);
    }
    const std::string cat = concat(keep, mod + "/cat", 1);
    return conv(cat, mod + ".cv2", c * (int)keep.size(), cout, 1, 1);
  }

  std::string sppf(const std::string& in, const std::string& mod, int cin, int cout) {
    const int c = cin / 2;
    const std::string y = conv(in, mod + ".cv1", cin, c, 1, 1);
    std::vector<std::string> parts = {y};
    std::string cur = y;
    for (int i = 0; i < 3; ++i) {
      const std::string p = uniq(mod + "/pool");
      add_node("MaxPool", {cur}, p,
               {ais("kernel_shape", {5, 5}), ais("strides", {1, 1}), ais("pads", {2, 2, 2, 2})});
      parts.push_back(p);
      cur = p;
    }
    const std::string cat = concat(parts, mod + "/cat", 1);
    return conv(cat, mod + ".cv2", 4 * c, cout, 1, 1);
  }

  // A plain Conv with bias (the last layer of each head branch)
  std::string conv_bias(const std::string& in, const std::string& mod, int cin, int cout, int k,
                        float bias_fill) {
    weight(mod + ".weight", {cout, cin, k, k}, 0.f, true);
    weight(mod + ".bias", {cout}, bias_fill, false);
    const std::string o = uniq(mod);
    add_node("Conv", {in, mod + ".weight", mod + ".bias"}, o,
             {ais("kernel_shape", {k, k}), ais("strides", {1, 1}),
              ais("pads", {k / 2, k / 2, k / 2, k / 2})});
    return o;
  }

  std::string reshape(const std::string& in, const std::vector<int64_t>& shape,
                      const std::string& tag) {
    const std::string s = uniq(tag + "/shape");
    init_i(s, {(int64_t)shape.size()}, shape);
    const std::string o = uniq(tag);
    add_node("Reshape", {in, s}, o);
    return o;
  }

  void detect(const std::vector<std::string>& feats, const std::vector<int>& chs) {
    const int reg = sp_.reg, nc = sp_.nc;
    const int c2 = std::max(16, std::max(chs[0] / 4, reg * 4));
    const int c3 = std::max(chs[0], std::min(nc, 100));
    const std::vector<int> strides = {8, 16, 32};

    std::vector<std::string> box_flat, cls_flat;
    int64_t total_a = 0;
    std::vector<int64_t> per_level;
    for (size_t l = 0; l < feats.size(); ++l) {
      const int64_t hw = sp_.imgsz / strides[l];
      const int64_t a = hw * hw;
      per_level.push_back(a);
      total_a += a;
      const std::string m = "model.22";
      // cv2 = box branch, cv3 = class branch; the last conv of each is what the trainer differentiates
      std::string b = conv(feats[l], m + ".cv2." + std::to_string(l) + ".0", chs[l], c2, 3, 1);
      b = conv(b, m + ".cv2." + std::to_string(l) + ".1", c2, c2, 3, 1);
      b = conv_bias(b, m + ".cv2." + std::to_string(l) + ".2", c2, 4 * reg, 1, 1.0f);
      std::string c = conv(feats[l], m + ".cv3." + std::to_string(l) + ".0", chs[l], c3, 3, 1);
      c = conv(c, m + ".cv3." + std::to_string(l) + ".1", c3, c3, 3, 1);
      // Ultralytics' bias_init, with 640 as the reference image size it was tuned at
      const float cb = std::log(5.f / (float)nc / std::pow(640.f / (float)strides[l], 2.f));
      c = conv_bias(c, m + ".cv3." + std::to_string(l) + ".2", c3, nc, 1, cb);
      box_flat.push_back(reshape(b, {1, 4 * reg, a}, "boxflat"));
      cls_flat.push_back(reshape(c, {1, nc, a}, "clsflat"));
    }

    // --- the shape find_v8_heads keys on: one Concat per branch, then DFL / Sigmoid
    const std::string box_cat = concat(box_flat, "box_cat", 2);
    const std::string cls_cat = concat(cls_flat, "cls_cat", 2);

    const std::string r = reshape(box_cat, {1, 4, reg, total_a}, "dfl_r");
    const std::string tr = uniq("dfl_t");
    add_node("Transpose", {r}, tr, {ais("perm", {0, 2, 1, 3})});          // [1, reg, 4, A]
    const std::string sm = uniq("dfl_sm");
    add_node("Softmax", {tr}, sm, {ai("axis", 1)});
    std::vector<float> proj((size_t)reg);
    for (int i = 0; i < reg; ++i) proj[(size_t)i] = (float)i;
    init_f("dfl.conv.weight", {1, reg, 1, 1}, proj);                      // the fixed 0..15 projection
    const std::string dc = uniq("dfl_conv");
    add_node("Conv", {sm, "dfl.conv.weight"}, dc,
             {ais("kernel_shape", {1, 1}), ais("strides", {1, 1}), ais("pads", {0, 0, 0, 0})});
    const std::string dist = reshape(dc, {1, 4, total_a}, "dfl_out");     // ltrb, in cells

    // anchors and strides as constants, exactly as the exported graph carries them
    std::vector<float> anchors((size_t)(2 * total_a));
    std::vector<float> stride_v((size_t)total_a);
    int64_t k = 0;
    for (size_t l = 0; l < feats.size(); ++l) {
      const int64_t hw = sp_.imgsz / strides[l];
      for (int64_t y = 0; y < hw; ++y)
        for (int64_t xx = 0; xx < hw; ++xx, ++k) {
          anchors[(size_t)k] = (float)xx + 0.5f;                       // x row
          anchors[(size_t)(total_a + k)] = (float)y + 0.5f;            // y row
          stride_v[(size_t)k] = (float)strides[l];
        }
    }
    init_f("anchors", {1, 2, total_a}, anchors);
    init_f("strides", {1, 1, total_a}, stride_v);

    auto slice = [&](const std::string& in, int64_t a, int64_t b, const std::string& tag) {
      const std::string s0 = uniq(tag + "/s"), e0 = uniq(tag + "/e"), ax = uniq(tag + "/a");
      init_i(s0, {1}, {a});
      init_i(e0, {1}, {b});
      init_i(ax, {1}, {1});
      const std::string o = uniq(tag);
      add_node("Slice", {in, s0, e0, ax}, o);
      return o;
    };
    const std::string lt = slice(dist, 0, 2, "lt");
    const std::string rb = slice(dist, 2, 4, "rb");
    const std::string x1y1 = uniq("x1y1");
    add_node("Sub", {"anchors", lt}, x1y1);
    const std::string x2y2 = uniq("x2y2");
    add_node("Add", {"anchors", rb}, x2y2);
    const std::string sum = uniq("cxcy_sum");
    add_node("Add", {x1y1, x2y2}, sum);
    init_f("two", {1}, {2.f});
    const std::string cxcy = uniq("cxcy");
    add_node("Div", {sum, "two"}, cxcy);
    const std::string wh = uniq("wh");
    add_node("Sub", {x2y2, x1y1}, wh);
    const std::string cxcywh = concat({cxcy, wh}, "cxcywh", 1);
    const std::string dbox = uniq("dbox");
    add_node("Mul", {cxcywh, "strides"}, dbox);                          // cells -> pixels
    const std::string sig = uniq("cls_sig");
    add_node("Sigmoid", {cls_cat}, sig);
    const std::string out = concat({dbox, sig}, "output0", 1);
    g_.outputs.push_back({out, {1, 4 + nc, total_a}});
  }
};

// One call: build the graph, optionally filling it from a torch checkpoint.
inline onx::Graph build(const Spec& sp, const std::map<std::string, pt::Tensor>* from_pt,
                        int* taken = nullptr, int* made = nullptr,
                        std::vector<std::string>* missed = nullptr) {
  Builder b(sp, from_pt);
  onx::Graph g = b.build();
  if (taken) *taken = b.taken;
  if (made) *made = b.made;
  if (missed) *missed = b.missed;
  return g;
}

}  // namespace mkdet
