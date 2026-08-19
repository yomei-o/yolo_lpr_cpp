// Recognizer weights -> ONNX. The classifier's trained weights live as lpr_cpp's forward-order
// blob (`manifest.txt` + `weights.bin`, extracted from the original Keras/TF ONNX); this builds a
// standalone .onnx from them so every consumer in this project — C++, Python, WASM — loads the
// same file. Model data is ONNX everywhere; the .bin stays an internal training format.
//
// Graph (mirrors lpr_cpp/pure/ref/ARCH.md, conv order Conv -> Relu -> BN, BN eps 1e-3):
//   input NCHW [1,3,128,128]  (note: the original ONNX took NHWC; this one is NCHW)
//   stem cbr(4x4 s4) -> branch A (block0 + 6 blocks + final 1x1) -> GAP -> 4 heads
//                    -> branch B (block0 + 5 blocks + final 1x1) -> GAP -> 5 heads
//   head = Gemm(feat[1,128], W[128,C]) + b -> Softmax        (probabilities, not logits)
#pragma once
#include "onnx.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>

namespace lprx {

struct Ent {
  char kind = 'C';                                   // 'C' conv, 'N' batchnorm, 'H' dense head
  int64_t Co = 0, Ci = 0, kh = 0, kw = 0, sh = 1, sw = 1, ph = 0, pw = 0, grp = 1, hasb = 0;
  std::string name;                                  // head output name (kind 'H')
  std::vector<float> w, b, gamma, beta, rm, rv;
};

// Read manifest.txt + weights.bin in forward order (same order load_lpr consumes them).
inline std::vector<Ent> read_blob(const std::string& dir) {
  std::string D = dir;
  if (!D.empty() && D.back() != '/' && D.back() != '\\') D += '/';
  std::ifstream mf(D + "manifest.txt");
  std::ifstream wf(D + "weights.bin", std::ios::binary);
  if (!mf) { printf("missing %smanifest.txt\n", D.c_str()); std::exit(1); }
  if (!wf) { printf("missing %sweights.bin\n", D.c_str()); std::exit(1); }
  auto rd = [&](int64_t n) { std::vector<float> v(n); wf.read((char*)v.data(), n * 4); return v; };
  int N = 0; mf >> N;
  std::vector<Ent> es; es.reserve(N);
  for (int i = 0; i < N; ++i) {
    std::string k; mf >> k;
    Ent e;
    if (k == "C") {
      e.kind = 'C';
      mf >> e.Co >> e.Ci >> e.kh >> e.kw >> e.sh >> e.sw >> e.ph >> e.pw >> e.grp >> e.hasb;
      e.w = rd(e.Co * e.Ci * e.kh * e.kw);
      if (e.hasb) e.b = rd(e.Co);
    } else if (k == "N") {
      e.kind = 'N';
      mf >> e.Co;
      e.gamma = rd(e.Co); e.beta = rd(e.Co); e.rm = rd(e.Co); e.rv = rd(e.Co);
    } else {
      e.kind = 'H';
      mf >> e.Ci >> e.Co >> e.name;
      e.w = rd(e.Ci * e.Co); e.b = rd(e.Co);
    }
    es.push_back(std::move(e));
  }
  return es;
}

struct Builder {
  onx::Graph g;
  const std::vector<Ent>& es;
  size_t i = 0;
  int uid = 0;
  explicit Builder(const std::vector<Ent>& e) : es(e) {}

  std::string fresh(const std::string& p) { return p + "_" + std::to_string(uid++); }
  std::string init_f(const std::string& name, const std::vector<int64_t>& dims, const std::vector<float>& d) {
    g.init_f.push_back({name, dims, d});
    return name;
  }
  void node(const std::string& op, const std::vector<std::string>& in,
            const std::string& out, const std::vector<onx::Attr>& attr = {}) {
    onx::Node n; n.op_type = op; n.name = out; n.input = in; n.output = {out}; n.attr = attr;
    g.nodes.push_back(std::move(n));
  }

  // Conv -> Relu -> BatchNormalization, consuming one 'C' and one 'N' entry.
  std::string cbr(const std::string& x) {
    const Ent& c = es[i++];
    const Ent& n = es[i++];
    std::string base = fresh("cbr");
    std::vector<std::string> in{x, init_f(base + "_w", {c.Co, c.Ci, c.kh, c.kw}, c.w)};
    if (c.hasb) in.push_back(init_f(base + "_b", {c.Co}, c.b));
    std::string yc = base + "_conv";
    node("Conv", in, yc, {
      {"kernel_shape", onx::A_INTS, 0, 0, "", {c.kh, c.kw}, {}},
      {"strides",      onx::A_INTS, 0, 0, "", {c.sh, c.sw}, {}},
      {"pads",         onx::A_INTS, 0, 0, "", {c.ph, c.pw, c.ph, c.pw}, {}},
      {"group",        onx::A_INT,  c.grp, 0, "", {}, {}},
    });
    std::string yr = base + "_relu";
    node("Relu", {yc}, yr);
    std::string yb = base + "_bn";
    node("BatchNormalization",
         {yr, init_f(base + "_g", {n.Co}, n.gamma), init_f(base + "_be", {n.Co}, n.beta),
          init_f(base + "_m", {n.Co}, n.rm), init_f(base + "_v", {n.Co}, n.rv)},
         yb, {{"epsilon", onx::A_FLOAT, 0, 1e-3f, "", {}, {}}});
    return yb;
  }

  // block0 (x + dw) -> nblocks x (h = 1x1; x = h + dw) -> final 1x1 -> GAP -> Flatten
  std::string branch(const std::string& stem, int nblocks) {
    std::string x = fresh("b0_add");
    node("Add", {stem, cbr(stem)}, x);
    for (int k = 0; k < nblocks; ++k) {
      std::string h = cbr(x);
      std::string a = fresh("blk_add");
      node("Add", {h, cbr(h)}, a);
      x = a;
    }
    x = cbr(x);
    std::string gp = fresh("gap");
    node("GlobalAveragePool", {x}, gp);
    std::string fl = fresh("flat");
    node("Flatten", {gp}, fl, {{"axis", onx::A_INT, 1, 0, "", {}, {}}});
    return fl;
  }

  void head(const std::string& feat) {
    const Ent& h = es[i++];
    std::string base = fresh("head");
    std::string gm = base + "_gemm";
    node("Gemm", {feat, init_f(base + "_w", {h.Ci, h.Co}, h.w), init_f(base + "_b", {h.Co}, h.b)}, gm);
    node("Softmax", {gm}, h.name, {{"axis", onx::A_INT, 1, 0, "", {}, {}}});
    g.outputs.push_back({h.name, {1, h.Co}});
  }
};

// nblocks_a/nblocks_b default to the shipped architecture (branch A 6, branch B 5).
inline void export_lpr_onnx(const std::string& ref_dir, const std::string& out_path,
                            int nblocks_a = 6, int nblocks_b = 5, int nheads_a = 4) {
  std::vector<Ent> es = read_blob(ref_dir);
  size_t nheads = 0, nconv = 0;
  for (auto& e : es) { if (e.kind == 'H') ++nheads; if (e.kind == 'C') ++nconv; }
  size_t expect_conv = 1 + (1 + 2 * nblocks_a + 1) + (1 + 2 * nblocks_b + 1);
  if (nconv != expect_conv) {
    printf("export_lpr_onnx: %zu convs but the %d/%d-block architecture needs %zu — "
           "manifest and architecture disagree\n", nconv, nblocks_a, nblocks_b, expect_conv);
    std::exit(1);
  }
  Builder B(es);
  B.g.opset = 13;
  B.g.inputs.push_back({"input", {1, 3, 128, 128}});
  std::string stem = B.cbr("input");
  std::string fa = B.branch(stem, nblocks_a);
  for (int k = 0; k < nheads_a; ++k) B.head(fa);
  std::string fb = B.branch(stem, nblocks_b);
  for (size_t k = nheads_a; k < nheads; ++k) B.head(fb);
  onx::save_onnx(B.g, out_path);
  printf("wrote %s  (%zu nodes, %zu initializers, %zu heads)\n",
         out_path.c_str(), B.g.nodes.size(), B.g.init_f.size(), nheads);
}

}  // namespace lprx
