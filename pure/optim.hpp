// Optimizers on the pure engine, matching torch.optim semantics: SGD (momentum,
// weight decay, Nesterov, dampening=0) and Adam / AdamW (bias correction, L2 vs
// decoupled weight decay). Each operates on Tensor params via p->data / p->grad.
#pragma once
#include "autograd.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

// Cosine LR schedule with linear warmup (like ultralytics/torch schedulers).
inline float cosine_lr(int step, int total, float base, int warmup = 0, float min_lr = 0.f) {
  if (step < warmup) return base * (float)(step + 1) / (float)std::max(1, warmup);
  float t = (float)(step - warmup) / (float)std::max(1, total - warmup);
  return min_lr + 0.5f * (base - min_lr) * (1.f + std::cos(3.14159265358979f * std::min(1.f, t)));
}

struct SGD {
  std::vector<Tensor> params; float lr, momentum, wd; bool nesterov;
  std::vector<std::vector<float>> buf; bool started = false;
  SGD(std::vector<Tensor> p, float lr, float momentum = 0.f, float weight_decay = 0.f, bool nesterov = false)
    : params(std::move(p)), lr(lr), momentum(momentum), wd(weight_decay), nesterov(nesterov) {
    for (auto& q : params) buf.emplace_back(q->numel(), 0.f);
  }
  void zero_grad() { for (auto& p : params) std::fill(p->grad.begin(), p->grad.end(), 0.f); }
  void step() {
    for (size_t k = 0; k < params.size(); ++k) {
      auto& p = params[k];
      for (int64_t i = 0; i < p->numel(); ++i) {
        float g = p->grad[i] + wd * p->data[i];
        if (momentum != 0.f) {
          float& b = buf[k][i];
          b = started ? momentum * b + g : g;          // torch: first-step buffer = grad
          g = nesterov ? g + momentum * b : b;
        }
        p->data[i] -= lr * g;
      }
    }
    started = true;
  }
};

struct Adam {
  std::vector<Tensor> params; float lr, b1, b2, eps, wd; bool decoupled;   // decoupled = AdamW
  std::vector<std::vector<float>> m, v; int t = 0;
  Adam(std::vector<Tensor> p, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
       float eps = 1e-8f, float weight_decay = 0.f, bool decoupled = false)
    : params(std::move(p)), lr(lr), b1(beta1), b2(beta2), eps(eps), wd(weight_decay), decoupled(decoupled) {
    for (auto& q : params) { m.emplace_back(q->numel(), 0.f); v.emplace_back(q->numel(), 0.f); }
  }
  void zero_grad() { for (auto& p : params) std::fill(p->grad.begin(), p->grad.end(), 0.f); }
  void step() {
    ++t;
    float bc1 = 1.f - std::pow(b1, t), bc2 = 1.f - std::pow(b2, t);
    for (size_t k = 0; k < params.size(); ++k) {
      auto& p = params[k];
      for (int64_t i = 0; i < p->numel(); ++i) {
        if (decoupled) p->data[i] -= lr * wd * p->data[i];     // AdamW: decoupled decay
        float g = p->grad[i] + (decoupled ? 0.f : wd * p->data[i]);   // Adam: L2 into grad
        m[k][i] = b1 * m[k][i] + (1 - b1) * g;
        v[k][i] = b2 * v[k][i] + (1 - b2) * g * g;
        float mhat = m[k][i] / bc1, vhat = v[k][i] / bc2;
        p->data[i] -= lr * mhat / (std::sqrt(vhat) + eps);
      }
    }
  }
};
