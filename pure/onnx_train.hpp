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

// Build the trainable view. `train_stats=false` keeps BN running mean/var out of the parameter list
// (they are statistics, not weights, and the graph runs BN in inference mode anyway).
inline Trainable make_trainable(const Graph& gin, bool freeze_backbone_bn_affine = false) {
  Trainable t;
  t.g = gin;
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
    bool trainable = !stats.count(tt.name) &&
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
