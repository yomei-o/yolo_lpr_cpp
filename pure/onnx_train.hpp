// Training an ONNX graph in place, with the pure engine.
//
// The idea that makes this small: the interpreter in onnx_run.hpp already executes any graph on
// autograd Tensors. If the initializers are handed in as *persistent* trainable Tensors instead of
// being rebuilt per call, then a forward pass builds a tape, backward() fills their .grad, and Adam
// can update them. No second architecture definition, no weight-name mapping, no risk of the C++
// model and the Python model quietly diverging — the file *is* the model.
//
//   Trainable t = make_trainable(load_onnx(path));      // params = float initializers
//   Tensor loss = ocr_loss(t, x, labels, mask);         // logits taken before the graph's Softmax
//   loss->backward();  adam.step();  free_graph(loss);
//   write_back(t); save_onnx(t.g, out);                 // the updated file
//
// BatchNorm stays in inference mode (running stats untouched). That is not a limitation here but the
// measured recipe: fine-tuning this model with BN statistics moving drags real-data accuracy down
// (see RESUME / tools/train_ocr.py).
#pragma once
#include "onnx.hpp"
#include "onnx_run.hpp"
#include "optim.hpp"
#include "ops2d.hpp"        // log_softmax_rows, gather_row
#include <map>
#include <string>
#include <vector>

namespace onx {

struct Trainable {
  Graph g;
  std::map<std::string, Tensor> init;        // every float initializer, as a live Tensor
  std::vector<Tensor> params;                // the subset we optimise
  std::vector<std::string> param_names;
  std::vector<std::string> heads;            // graph output names, in declaration order
  std::vector<std::string> logits;           // the tensor feeding each head's Softmax
};

// Widen a head so it can name classes the shipped graph never had: the region head arrives with 133
// names and spec/labels.txt now lists 138 (十勝/日光/江戸川/安曇野/南信州 were added in 2025). The
// Python side does this while building its PlateNet (tools/ocr_model.py); doing it here is what lets
// `jlpr train` start from the same ONNX and reach the same model.
//
// `new_bias` is the initial bias of the appended classes and is not a detail: with zero weights their
// logit is exactly 0, which beats every real class whose logit is negative (measured: -6 points of
// region accuracy before a single step), so a class nobody trains must start at -10. A class that IS
// trained should start at 0 instead, or it spends the whole run climbing out of that hole (measured:
// 1000 steps left 江戸川 at 1e-7, rank 133 of 138).
//
// Returns the heads it changed, as "region(133->138)" strings, in graph output order.
inline std::vector<std::string> widen_heads(Graph& g, const std::map<std::string, int>& want,
                                           float new_bias) {
  std::vector<std::string> changed;
  std::map<std::string, std::string> pre;   // head output -> the tensor its Softmax consumes
  for (const Node& n : g.nodes)
    if (n.op_type == "Softmax" && !n.input.empty() && !n.output.empty()) pre[n.output[0]] = n.input[0];

  for (ValueInfo& out : g.outputs) {
    // head names in the spec have no "_output" suffix; the graph outputs do. Two of them also carry an
    // "_id" the spec dropped (same alias table as tools/ocr_model.py).
    std::string base = out.name;
    const std::string suffix = "_output";
    if (base.size() > suffix.size() && base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0)
      base = base.substr(0, base.size() - suffix.size());
    if (base == "region_id") base = "region";
    else if (base == "hiragana_id") base = "hiragana";
    auto w = want.find(base);
    if (w == want.end()) continue;
    const std::string logits = pre.count(out.name) ? pre[out.name] : out.name;

    for (const Node& n : g.nodes) {
      bool produces = false;
      for (const std::string& o : n.output) produces = produces || (o == logits);
      if (!produces || n.input.size() < 3) continue;
      Tensor64* wt = nullptr;
      Tensor64* bt = nullptr;
      for (Tensor64& t : g.init_f) {
        if (t.name == n.input[1]) wt = &t;
        if (t.name == n.input[2]) bt = &t;
      }
      if (!wt || !bt || bt->dims.size() != 1) break;
      // Gemm without transB: weight is [in, out], so the classes are *columns* and widening restrides.
      // Conv/transposed layouts put them in dim 0, which is a plain append.
      bool cols = (n.op_type == "Gemm" && wt->dims.size() == 2 && wt->dims[1] == bt->dims[0]);
      bool rows = (wt->dims.size() >= 1 && wt->dims[0] == bt->dims[0]);
      int have = (int)bt->dims[0];
      if (w->second <= have || (!cols && !rows)) break;
      int add = w->second - have;
      if (cols) {
        int in = (int)wt->dims[0];
        std::vector<float> nd((size_t)in * (size_t)w->second, 0.f);
        for (int i = 0; i < in; ++i)
          for (int j = 0; j < have; ++j) nd[(size_t)i * w->second + j] = wt->data[(size_t)i * have + j];
        wt->data.swap(nd);
        wt->dims[1] = w->second;
      } else {
        size_t per = wt->data.size() / (size_t)have;      // elements per output class
        wt->data.resize((size_t)w->second * per, 0.f);
        wt->dims[0] = w->second;
      }
      bt->data.resize((size_t)w->second, new_bias);
      bt->dims[0] = w->second;
      (void)add;
      if (!out.dims.empty()) out.dims.back() = w->second;
      char buf[96];
      snprintf(buf, sizeof buf, "%s(%d->%d)", base.c_str(), have, w->second);
      changed.push_back(buf);
      break;
    }
  }
  return changed;
}

// Which initializers are *weights* — the only tensors an optimiser may touch.
//
// THIS FILTER IS NOT COSMETIC. `init_f` also holds tensors the graph needs to keep **exactly**: the
// Resize `scales` of a yolov8 neck is the float tensor [1,1,2,2], and an optimiser with weight decay
// shrinks it to 1.9999996 — after which the interpreter's `(int64_t)scale` upsamples by 1 instead of
// 2, the neck stops aligning, and the model is destroyed by a change of 2e-7. Measured: one AdamW
// step with the default 5e-4 decay took the training loss from 2.90 to 23.13 and mAP50 from 0.995 to
// 0.005, with every actual weight still correct to seven digits. Gradient descent alone never caught
// it because those tensors get no gradient — decoupled weight decay does not ask.
//
// `needed` (optional) restricts the walk to the subgraph that produces those tensors: for the
// detector the loss is attached to the six head convs, so the DFL projection and the decode tail
// behind them are not parameters either, however conv-shaped they look.
inline std::set<std::string> weight_initializers(const Graph& g, const std::set<std::string>& needed = {}) {
  std::map<std::string, const Node*> prod;
  for (const Node& n : g.nodes)
    for (const std::string& o : n.output) prod[o] = &n;
  std::set<const Node*> keep;
  if (needed.empty()) {
    for (const Node& n : g.nodes) keep.insert(&n);
  } else {
    std::vector<std::string> stack(needed.begin(), needed.end());
    while (!stack.empty()) {
      const std::string name = stack.back();
      stack.pop_back();
      auto it = prod.find(name);
      if (it == prod.end() || keep.count(it->second)) continue;
      keep.insert(it->second);
      for (const std::string& in : it->second->input) stack.push_back(in);
    }
  }
  std::set<std::string> inits;
  for (const Tensor64& t : g.init_f) inits.insert(t.name);
  std::set<std::string> out;
  auto take = [&](const Node* n, size_t i) {
    if (i < n->input.size() && inits.count(n->input[i])) out.insert(n->input[i]);
  };
  for (const Node* n : keep) {
    if (n->op_type == "Conv" || n->op_type == "ConvTranspose" || n->op_type == "Gemm") { take(n, 1); take(n, 2); }
    else if (n->op_type == "BatchNormalization") { take(n, 1); take(n, 2); }   // scale/bias; stats below
    else if (n->op_type == "MatMul" || n->op_type == "PRelu") take(n, 1);
  }
  return out;
}

// Build the trainable view. `train_stats=false` keeps BN running mean/var out of the parameter list
// (they are statistics, not weights, and the graph runs BN in inference mode anyway). `needed` is
// passed through to weight_initializers: name the tensors the loss actually reads.
inline Trainable make_trainable(const Graph& gin, bool freeze_backbone_bn_affine = false,
                                const std::set<std::string>& needed = {}) {
  Trainable t;
  t.g = gin;
  const std::set<std::string> weights = weight_initializers(gin, needed);
  // which initializers are BN statistics (inputs 3 and 4 of a BatchNormalization node)
  std::set<std::string> stats, bn_affine;
  for (const Node& n : t.g.nodes) {
    if (n.op_type != "BatchNormalization" || n.input.size() < 5) continue;
    bn_affine.insert(n.input[1]);
    bn_affine.insert(n.input[2]);
    stats.insert(n.input[3]);
    stats.insert(n.input[4]);
  }
  for (const Tensor64& tt : t.g.init_f) {
    bool trainable = weights.count(tt.name) && !stats.count(tt.name) &&
                     !(freeze_backbone_bn_affine && bn_affine.count(tt.name));
    Tensor v = from_data(tt.dims, tt.data, trainable);
    t.init[tt.name] = v;
    if (trainable) {
      t.params.push_back(v);
      t.param_names.push_back(tt.name);
    }
  }
  for (const ValueInfo& vi : t.g.outputs) t.heads.push_back(vi.name);
  // the logits are what each Softmax consumes; training on them keeps log-softmax stable
  std::map<std::string, std::string> pre;
  for (const Node& n : t.g.nodes)
    if (n.op_type == "Softmax" && !n.input.empty() && !n.output.empty()) pre[n.output[0]] = n.input[0];
  for (const std::string& h : t.heads) t.logits.push_back(pre.count(h) ? pre[h] : h);
  return t;
}

inline std::map<std::string, Tensor> forward(Trainable& t, const Tensor& x) {
  return run_onnx(t.g, x, {}, &t.init, false);
}

// Summed multi-head cross-entropy with a per-sample, per-head mask. Masking is what lets real crops
// with only a 地域名 label train alongside fully-labelled synthetic ones without teaching the other
// heads garbage.
inline Tensor multihead_ce(const std::map<std::string, Tensor>& vals, const Trainable& t,
                          const std::vector<std::vector<int>>& labels,   // [head][batch]
                          const std::vector<std::vector<float>>& mask,   // [head][batch]
                          std::vector<float>* per_head_loss = nullptr) {
  Tensor total;
  for (size_t h = 0; h < t.logits.size(); ++h) {
    const Tensor& z = vals.at(t.logits[h]);
    Tensor lp = log_softmax_rows(z);
    std::vector<int64_t> idx(labels[h].begin(), labels[h].end());
    Tensor picked = gather_row(lp, idx);                    // (B,1) log-prob of the target class
    float denom = 0.f;
    for (float m : mask[h]) denom += m;
    if (denom <= 0) continue;
    // loss_h = -(sum over the batch of mask_b * logp_b) / (number of unmasked samples)
    std::vector<float> coeff(mask[h].size());
    for (size_t b = 0; b < mask[h].size(); ++b) coeff[b] = -mask[h][b] / denom;
    Tensor l = wsum_rows(transpose2d(picked), coeff);       // (1,B) -> (1,1)
    total = total ? add(total, l) : l;
    if (per_head_loss) {
      float acc = 0.f;
      for (size_t b = 0; b < mask[h].size(); ++b) acc += coeff[b] * picked->data[b];
      per_head_loss->push_back(acc);
    }
  }
  return total;
}

// Copy the trained parameter values back into the graph's initializers, ready for save_onnx.
inline void write_back(Trainable& t) {
  for (Tensor64& tt : t.g.init_f) {
    auto it = t.init.find(tt.name);
    if (it == t.init.end()) continue;
    tt.data = it->second->data;
  }
}

inline size_t param_count(const Trainable& t) {
  size_t n = 0;
  for (const Tensor& p : t.params) n += (size_t)p->numel();
  return n;
}

}  // namespace onx
