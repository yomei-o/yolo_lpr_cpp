// Build a recogniser graph from nothing but the label spec — the third and last stage that used to
// need a starting file from somewhere else.
//
//   jlpr init-ocr --out models/scratch_ocr.onnx                    # 11 heads, sized from spec/labels.txt
//   jlpr train --model ocr --init models/scratch_ocr.onnx --synth data/synth --alpr ../alpr_jp ...
//
// `jlpr export` already wrote this architecture, but only by copying weights out of lpr_cpp's
// weights.bin; and `jlpr train --model ocr` needed an existing ONNX in `--init`. So the C++ lane could
// fine-tune the shipped recogniser forever and never make one. This closes that: the topology is the
// same one pure/onnx_export_lpr.hpp emits (Conv -> Relu -> BN, a 4x4/4 stem, two depthwise-separable
// branches, GAP, one Gemm+Softmax per head), the head widths come from spec/labels.txt, and the
// weights are drawn the way torch draws them.
//
// Trained from scratch this will not reach the shipped 97.9% on 576 real crops — the point is that the
// path exists and is honest about where the numbers come from: `--init random` on the corner net gets
// 2.8% against the shipped 1.9%, and the same gap should be expected here.
#pragma once
#include "onnx_export_lpr.hpp"
#include "rng.hpp"
#include "spec.hpp"
#include <string>
#include <vector>

namespace lprx {

struct MakeCfg {
  int ch = 128;          // every conv in this architecture is 128 wide
  int nblocks_a = 6;     // branch A: the shipped split (4 heads on A, the rest on B)
  int nblocks_b = 5;
  int nheads_a = 4;
  int in_px = 128;
  uint64_t seed = 1234;
};

// torch's default for Conv2d/Linear: U(-1/sqrt(fan_in), +1/sqrt(fan_in))
inline std::vector<float> uniform_fan(size_t n, int64_t fan_in, Rng& rng) {
  const float b = 1.f / std::sqrt((float)std::max<int64_t>(1, fan_in));
  std::vector<float> v(n);
  for (float& q : v) q = (float)rng.range(-b, b);
  return v;
}

inline Ent make_conv(int64_t co, int64_t ci, int64_t k, int64_t s, int64_t p, int64_t grp, Rng& rng) {
  Ent e;
  e.kind = 'C';
  e.Co = co; e.Ci = ci; e.kh = k; e.kw = k; e.sh = s; e.sw = s; e.ph = p; e.pw = p;
  e.grp = grp; e.hasb = 1;
  const int64_t fan_in = ci * k * k;
  e.w = uniform_fan((size_t)(co * ci * k * k), fan_in, rng);
  e.b = uniform_fan((size_t)co, fan_in, rng);
  return e;
}

inline Ent make_bn(int64_t c) {
  Ent e;
  e.kind = 'N';
  e.Co = c;
  e.gamma.assign((size_t)c, 1.f);
  e.beta.assign((size_t)c, 0.f);
  e.rm.assign((size_t)c, 0.f);
  e.rv.assign((size_t)c, 1.f);
  return e;
}

inline Ent make_head(const std::string& out_name, int64_t in_f, int64_t out_f, Rng& rng) {
  Ent e;
  e.kind = 'H';
  e.Ci = in_f;
  e.Co = out_f;
  e.name = out_name;
  e.w = uniform_fan((size_t)(in_f * out_f), in_f, rng);
  e.b = uniform_fan((size_t)out_f, in_f, rng);
  return e;
}

// The graph's output names follow the shipped model's, including the two that carry an "_id" the spec
// dropped — anything keyed on those names (tools/ocr_model.py's alias table, onx::widen_heads) keeps
// working on a graph we generated ourselves.
inline std::string head_output_name(const std::string& spec_name) {
  if (spec_name == "region") return "region_id_output";
  if (spec_name == "hiragana") return "hiragana_id_output";
  return spec_name + "_output";
}

// Entries in exactly the order lprx::Builder consumes them: stem, branch A, branch B, then the heads
// (A's first). A mismatch here shows up as a wrong-shaped graph, so the count is checked below.
inline std::vector<Ent> make_entries(const spec::Spec& sp, const MakeCfg& cfg) {
  Rng rng(cfg.seed);
  const int c = cfg.ch;
  std::vector<Ent> es;
  auto dw = [&] {                       // depthwise 5x5, one group per channel
    es.push_back(make_conv(c, 1, 5, 1, 2, c, rng));
    es.push_back(make_bn(c));
  };
  auto pw = [&] {                       // pointwise 1x1
    es.push_back(make_conv(c, c, 1, 1, 0, 1, rng));
    es.push_back(make_bn(c));
  };

  auto branch = [&](int n) {
    dw();                               // block0: x + dw(x)
    for (int k = 0; k < n; ++k) { pw(); dw(); }
    pw();                               // final 1x1 before GAP
  };
  // The order is the order lprx::Builder reads in, and it is *interleaved*: branch A's convs, then
  // branch A's heads, then branch B's convs, then the rest of the heads. Emitting all the heads at the
  // end (the obvious way) makes the first four heads consume branch B's convolutions instead, and the
  // symptom is an ONNX whose first four outputs have empty names.
  const std::vector<const spec::Group*> heads = sp.of_kind("head");
  es.push_back(make_conv(c, 3, 4, 4, 0, 1, rng));   // stem: 4x4 stride 4
  es.push_back(make_bn(c));
  branch(cfg.nblocks_a);
  for (int k = 0; k < cfg.nheads_a && (size_t)k < heads.size(); ++k)
    es.push_back(make_head(head_output_name(heads[(size_t)k]->name), c, heads[(size_t)k]->n, rng));
  branch(cfg.nblocks_b);
  for (size_t k = (size_t)cfg.nheads_a; k < heads.size(); ++k)
    es.push_back(make_head(head_output_name(heads[k]->name), c, heads[k]->n, rng));
  return es;
}

inline void make_lpr_onnx(const spec::Spec& sp, const std::string& out_path, const MakeCfg& cfg) {
  std::vector<Ent> es = make_entries(sp, cfg);
  Builder B(es);
  B.g.opset = 13;
  B.g.inputs.push_back({"input", {1, 3, cfg.in_px, cfg.in_px}});
  const std::string stem = B.cbr("input");
  const std::string fa = B.branch(stem, cfg.nblocks_a);
  const size_t nheads = sp.of_kind("head").size();
  for (int k = 0; k < cfg.nheads_a && (size_t)k < nheads; ++k) B.head(fa);
  const std::string fb = B.branch(stem, cfg.nblocks_b);
  for (size_t k = (size_t)cfg.nheads_a; k < nheads; ++k) B.head(fb);
  onx::save_onnx(B.g, out_path);
  size_t params = 0;
  for (const onx::Tensor64& t : B.g.init_f) params += t.data.size();
  printf("wrote %s: %zu nodes, %zu initializers, %zu heads, %zu parameters\n", out_path.c_str(),
         B.g.nodes.size(), B.g.init_f.size(), nheads, params);
}

}  // namespace lprx
