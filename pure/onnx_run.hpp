// Graph-driven ONNX interpreter — one runner for every model in this project (detector, corner,
// recognizer). No architecture is hardcoded: it walks the parsed Graph (onnx.hpp) and executes
// nodes on the pure engine's ops. Merged from yolov8_cpp/yolox_cpp (Conv/Sigmoid/Mul/MaxPool/
// Resize/Concat/Slice) and lpr_cpp's facenet runner (Relu/GAP/Flatten/Gemm), plus
// BatchNormalization / Softmax / Split which the recognizer graph needs.
//
// WHO HAS ALREADY APPLIED THE ACTIVATION. Read the header of infer_yolox.hpp before touching the
// decode: a graph whose head ends in Sigmoid hands you probabilities, and sigmoiding them again
// is invisible in a parity test but destroys detection. Pass `logits=false` for such graphs.
#pragma once
#include "onnx.hpp"
#include "autograd.hpp"
#include "ops_yolox.hpp"   // dwconv2d, slice_hw
#include "ops2d.hpp"       // mul, reshape, mul_scalar, softmax_rows
#include "face_ops.hpp"    // relu, conv2d_hw, gap, add_rowvec
#include "linalg.hpp"      // matmul, transpose2d
#include "bn.hpp"          // batchnorm2d
#include "nd.hpp"          // rank-agnostic forward-only ops (transpose/softmax/matmul/broadcast)
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace onx {

inline const Attr* find_attr(const Node& n, const std::string& name) {
  for (auto& a : n.attr) if (a.name == name) return &a;
  return nullptr;
}
inline int64_t attr_i0(const Node& n, const std::string& name, int64_t def) {
  const Attr* a = find_attr(n, name);
  if (!a) return def;
  return a->ints.empty() ? a->i : a->ints[0];
}
inline float attr_f(const Node& n, const std::string& name, float def) {
  const Attr* a = find_attr(n, name);
  return a ? a->f : def;
}
inline std::vector<int64_t> attr_ints(const Node& n, const std::string& name) {
  const Attr* a = find_attr(n, name);
  return a ? a->ints : std::vector<int64_t>{};
}

// Run the graph on input x; returns name -> Tensor for every value produced. If `stop` is
// non-empty, execution halts once all named tensors exist (used to grab per-level detector head
// outputs and skip a decode tail the 4D engine can't express).
// `preset` lets a caller supply the initializer tensors instead of having them rebuilt from the
// graph on every call — that is what makes training possible: the same parameter Tensors persist
// across steps, so gradients accumulate into them and an optimiser can update them in place.
inline std::map<std::string, Tensor> run_onnx(const Graph& g, const Tensor& x,
                                             const std::set<std::string>& stop = {},
                                             const std::map<std::string, Tensor>* preset = nullptr,
                                             bool bn_training = false) {
  std::map<std::string, Tensor> vals;
  std::map<std::string, const IntsTensor*> imap;
  std::deque<IntsTensor> const_ints;      // storage for Constant nodes (deque: stable addresses)
  if (preset) vals = *preset;
  else for (const auto& t : g.init_f) vals[t.name] = from_data(t.dims, t.data);
  for (const auto& t : g.init_i) imap[t.name] = &t;

  std::string in_name;                      // the declared input that isn't an initializer
  for (const auto& vi : g.inputs) if (!vals.count(vi.name)) { in_name = vi.name; break; }
  if (in_name.empty() && !g.inputs.empty()) in_name = g.inputs[0].name;
  vals[in_name] = x;

  // Named lookup with a real error message. This used to be vals.at(n), which threw
  // std::out_of_range with no context — and because the interpreter also *silently* skipped
  // unsupported ops when `stop` was set, an unimplemented op turned into an uncaught exception that
  // killed the process before stdout was even flushed. Both halves of that are fixed here.
  const Node* cur = nullptr;
  auto get = [&](const std::string& n) -> Tensor {
    auto it = vals.find(n);
    if (it == vals.end()) {
      fprintf(stderr, "onnx_run: tensor '%s' is missing (needed by %s '%s'). An op earlier in the "
                      "graph is probably unimplemented.\n", n.c_str(),
              cur ? cur->op_type.c_str() : "?", cur ? cur->name.c_str() : "?");
      fflush(stderr);
      std::exit(2);
    }
    return it->second;
  };

  for (const auto& nd : g.nodes) {
    const std::string& op = nd.op_type;
    cur = &nd;
    Tensor y;
    if (op == "Constant") {
      // A shape/scale carried in an attribute rather than as an initializer.
      const Attr* at = find_attr(nd, "value");
      if (at && at->has_tensor) {
        if (at->t_dtype == 7) {
          const_ints.push_back(IntsTensor{nd.output[0], at->t_dims, at->t_ints});
          imap[nd.output[0]] = &const_ints.back();
        } else {
          std::vector<int64_t> dims = at->t_dims;
          if (dims.empty()) dims.push_back((int64_t)at->t_floats.size());
          vals[nd.output[0]] = from_data(dims, at->t_floats);
        }
      }
      continue;
    } else if (op == "Conv") {
      Tensor w = get(nd.input[1]);
      Tensor b = (nd.input.size() >= 3 && !nd.input[2].empty()) ? get(nd.input[2]) : nullptr;
      auto st = attr_ints(nd, "strides"), pd = attr_ints(nd, "pads");
      int64_t s = st.empty() ? attr_i0(nd, "strides", 1) : st[0];
      int64_t ph = pd.size() > 0 ? pd[0] : 0, pw = pd.size() > 1 ? pd[1] : ph;
      int64_t grp = attr_i0(nd, "group", 1);
      if (grp > 1 && grp == w->shape[0] && w->shape[1] == 1) y = dwconv2d(get(nd.input[0]), w, b, s, ph);
      else if (ph == pw)                                     y = conv2d(get(nd.input[0]), w, b, s, ph, grp);
      else                                                   y = conv2d_hw(get(nd.input[0]), w, b, s, ph, pw);
    } else if (op == "Relu") {
      y = relu(get(nd.input[0]));
    } else if (op == "Sigmoid") {
      y = sigmoid(get(nd.input[0]));
    } else if (op == "Mul") {
      Tensor a = get(nd.input[0]), b = get(nd.input[1]);
      y = b->numel() == 1 ? mul_scalar(a, b->data[0])
        : (a->numel() == 1 ? mul_scalar(b, a->data[0])
        : (a->shape == b->shape ? mul(a, b) : ::nd::mul(a, b)));
    } else if (op == "Add") {
      Tensor a = get(nd.input[0]), b = get(nd.input[1]);
      y = (a->shape == b->shape) ? add(a, b) : ::nd::add(a, b);
    } else if (op == "Sub") {
      y = ::nd::sub(get(nd.input[0]), get(nd.input[1]));
    } else if (op == "Div") {
      y = ::nd::div(get(nd.input[0]), get(nd.input[1]));
    } else if (op == "Transpose") {
      y = ::nd::transpose(get(nd.input[0]), attr_ints(nd, "perm"));
    } else if (op == "BatchNormalization") {
      Tensor gamma = get(nd.input[1]), beta = get(nd.input[2]);
      Tensor rm = get(nd.input[3]), rv = get(nd.input[4]);
      std::vector<float> m = rm->data, v = rv->data;              // read-only copies of the stats
      y = batchnorm2d(get(nd.input[0]), gamma, beta, m, v, attr_f(nd, "epsilon", 1e-5f),
                      bn_training, 0.f);
    } else if (op == "MaxPool") {
      int64_t k = attr_i0(nd, "kernel_shape", 1), s = attr_i0(nd, "strides", 1), p = attr_i0(nd, "pads", 0);
      y = maxpool2d(get(nd.input[0]), k, s, p);
    } else if (op == "Resize") {
      Tensor sc = vals.count(nd.input.back()) ? get(nd.input.back()) : nullptr;
      int64_t f = (sc && sc->numel() >= 4) ? (int64_t)sc->data[2] : 2;
      y = upsample_nearest(get(nd.input[0]), f);
    } else if (op == "Concat") {
      std::vector<Tensor> xs;
      for (auto& s : nd.input) xs.push_back(get(s));
      int64_t axis = attr_i0(nd, "axis", 1);
      y = (axis == 1 && xs[0]->shape.size() == 4) ? concat_ch(xs) : ::nd::concat(xs, axis);
    } else if (op == "Slice") {
      auto* st = imap.at(nd.input[1]); auto* en = imap.at(nd.input[2]);
      const IntsTensor* ax = nd.input.size() > 3 ? imap.at(nd.input[3]) : nullptr;
      const IntsTensor* stp = nd.input.size() > 4 ? imap.at(nd.input[4]) : nullptr;
      Tensor xin = get(nd.input[0]);
      if (ax && ax->data.size() == 2 && ax->data[0] == 2 && ax->data[1] == 3) {
        int64_t hstep = stp ? stp->data[0] : 1, wstep = stp ? stp->data[1] : 1;   // Focus
        y = slice_hw(xin, st->data[0], st->data[1], hstep, wstep);
      } else if (xin->shape.size() == 4 && ax && ax->data.size() == 1 && ax->data[0] == 1) {
        int64_t c0 = st->data[0], c1 = en->data[0], C = xin->shape[1];
        if (c1 > C) c1 = C;
        y = slice_ch(xin, c0, c1);                                // the verified 4D channel path
      } else {
        y = ::nd::slice(xin, st->data, en->data, ax ? ax->data : std::vector<int64_t>{},
                        stp ? stp->data : std::vector<int64_t>{});
      }
    } else if (op == "Split") {
      Tensor xin = get(nd.input[0]);
      int64_t axis = attr_i0(nd, "axis", 0);
      std::vector<int64_t> parts = attr_ints(nd, "split");
      if (parts.empty() && nd.input.size() > 1 && imap.count(nd.input[1])) parts = imap.at(nd.input[1])->data;
      if (parts.empty()) {                                        // equal split
        int64_t k = (int64_t)nd.output.size(), C = xin->shape[axis];
        for (int64_t i = 0; i < k; ++i) parts.push_back(C / k);
      }
      if (axis == 1 && xin->shape.size() == 4) {
        int64_t off = 0;
        for (size_t o = 0; o < nd.output.size(); ++o) {
          vals[nd.output[o]] = slice_ch(xin, off, off + parts[o]);
          off += parts[o];
        }
      } else {
        std::vector<Tensor> ps = ::nd::split(xin, axis, parts);
        for (size_t o = 0; o < nd.output.size() && o < ps.size(); ++o) vals[nd.output[o]] = ps[o];
      }
      continue;                                                   // multi-output: already stored
    } else if (op == "GlobalAveragePool") {
      Tensor t = gap(get(nd.input[0]));
      y = reshape(t, {t->shape[0], t->shape[1], 1, 1});
    } else if (op == "Flatten") {
      Tensor t = get(nd.input[0]);
      y = reshape(t, {t->shape[0], t->numel() / t->shape[0]});
    } else if (op == "Reshape") {
      Tensor t = get(nd.input[0]);
      std::vector<int64_t> shp = imap.count(nd.input[1]) ? imap.at(nd.input[1])->data : std::vector<int64_t>{};
      if (shp.empty()) {
        fprintf(stderr, "onnx_run: Reshape '%s' has no shape input ('%s' is not an initializer or "
                        "Constant)\n", nd.name.c_str(), nd.input[1].c_str());
        std::exit(1);
      }
      int64_t known = 1; int neg = -1;
      for (size_t i = 0; i < shp.size(); ++i) {
        if (shp[i] == -1) neg = (int)i;
        else if (shp[i] == 0) { shp[i] = t->shape[i]; known *= shp[i]; }
        else known *= shp[i];
      }
      if (neg >= 0) shp[neg] = t->numel() / known;
      y = reshape(t, shp);
    } else if (op == "Gemm") {
      Tensor a = get(nd.input[0]), W = get(nd.input[1]);
      Tensor prod = attr_i0(nd, "transB", 0) ? matmul(a, transpose2d(W)) : matmul(a, W);
      y = (nd.input.size() >= 3 && !nd.input[2].empty()) ? add_rowvec(prod, get(nd.input[2])) : prod;
    } else if (op == "MatMul") {
      Tensor a = get(nd.input[0]), b = get(nd.input[1]);
      y = (a->shape.size() == 2 && b->shape.size() == 2) ? matmul(a, b) : ::nd::matmul(a, b);
    } else if (op == "Softmax") {
      Tensor a = get(nd.input[0]);
      int64_t axis = attr_i0(nd, "axis", a->shape.size() == 2 ? 1 : -1);
      y = (a->shape.size() == 2 && (axis == 1 || axis == -1)) ? softmax_rows(a)   // verified path
                                                             : ::nd::softmax(a, axis);
    } else if (op == "Identity") {
      y = get(nd.input[0]);
    } else {
      if (!stop.empty()) continue;                                // tolerate a decode tail we stop before
      printf("unsupported ONNX op: %s\n", op.c_str());
      std::exit(1);
    }
    vals[nd.output[0]] = y;
    if (!stop.empty()) {
      bool all = true;
      for (auto& s : stop) if (!vals.count(s)) { all = false; break; }
      if (all) break;
    }
  }
  return vals;
}

}  // namespace onx
