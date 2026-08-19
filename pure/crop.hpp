// Image sampling helpers shared by the CLI, the WASM demo and (later) the trainers.
//   crop_to_input  : plate box (+margin) -> 128x128 RGB /255 NCHW, the recognizer's input
//   letterbox_bgr  : whole frame -> SxS BGR 0-255 pad-114 top-left, the YOLOX detector's input
// crop_to_input is lifted from lpr_cpp/pure/lpr_tta.hpp so this repo does not depend on the
// classifier's non-ONNX code path.
#pragma once
#include "autograd.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace jl {

inline std::vector<float> crop_to_input(const unsigned char* rgb, int W, int H,
                                        float bx0, float by0, float bx1, float by1, float margin) {
  const float bw = bx1 - bx0, bh = by1 - by0;
  const float x0 = bx0 - bw * margin, y0 = by0 - bh * margin;
  const float x1 = bx1 + bw * margin, y1 = by1 + bh * margin;
  std::vector<float> out((size_t)(3 * 128 * 128), 0.f);
  auto px = [&](int yy, int xx, int c) {
    yy = std::clamp(yy, 0, H - 1); xx = std::clamp(xx, 0, W - 1);
    return (float)rgb[((size_t)yy * W + xx) * 3 + c];
  };
  for (int y = 0; y < 128; ++y)
    for (int x = 0; x < 128; ++x) {
      const float sx = x0 + (x + 0.5f) * (x1 - x0) / 128.f - 0.5f;
      const float sy = y0 + (y + 0.5f) * (y1 - y0) / 128.f - 0.5f;
      const int xi = (int)std::floor(sx), yi = (int)std::floor(sy);
      const float fx = sx - xi, fy = sy - yi;
      for (int c = 0; c < 3; ++c) {
        const float v = px(yi, xi, c) * (1 - fx) * (1 - fy) + px(yi, xi + 1, c) * fx * (1 - fy)
                      + px(yi + 1, xi, c) * (1 - fx) * fy + px(yi + 1, xi + 1, c) * fx * fy;
        out[(size_t)((c * 128 + y) * 128 + x)] = v / 255.f;
      }
    }
  return out;
}

// Letterbox into an SxS BGR 0-255 tensor, image pinned to the top-left, background 114 —
// the preprocessing the plate YOLOX was trained with. `scale` comes back so boxes can be
// mapped to original pixels (x_orig = x_net / scale).
inline Tensor letterbox_bgr(const unsigned char* rgb, int W, int H, int S, float& scale) {
  scale = std::min((float)S / W, (float)S / H);
  const int nw = (int)(W * scale), nh = (int)(H * scale);
  Tensor t = from_data({1, 3, S, S}, std::vector<float>((size_t)3 * S * S, 114.f));
  auto px = [&](int yy, int xx, int c) {
    yy = std::clamp(yy, 0, H - 1); xx = std::clamp(xx, 0, W - 1);
    return (float)rgb[((size_t)yy * W + xx) * 3 + c];
  };
  for (int y = 0; y < nh; ++y)
    for (int x = 0; x < nw; ++x) {
      const float sx = (x + 0.5f) / scale - 0.5f, sy = (y + 0.5f) / scale - 0.5f;
      const int xi = (int)std::floor(sx), yi = (int)std::floor(sy);
      const float fx = sx - xi, fy = sy - yi;
      for (int c = 0; c < 3; ++c) {
        const float v = px(yi, xi, c) * (1 - fx) * (1 - fy) + px(yi, xi + 1, c) * fx * (1 - fy)
                      + px(yi + 1, xi, c) * (1 - fx) * fy + px(yi + 1, xi + 1, c) * fx * fy;
        t->data[(size_t)((2 - c) * S + y) * S + x] = v;   // BGR
      }
    }
  return t;
}

// Plain resize (no letterbox) into RGB 0-1 — what an Ultralytics-style export of PlateYOLO-JP was
// fed at training time. Aspect ratio is NOT preserved, so boxes map back with separate x/y scales.
inline Tensor resize_rgb01(const unsigned char* rgb, int W, int H, int iw, int ih) {
  Tensor t = make_tensor({1, 3, ih, iw}, false);
  auto px = [&](int yy, int xx, int c) {
    yy = std::clamp(yy, 0, H - 1); xx = std::clamp(xx, 0, W - 1);
    return (float)rgb[((size_t)yy * W + xx) * 3 + c];
  };
  for (int y = 0; y < ih; ++y)
    for (int x = 0; x < iw; ++x) {
      const float sx = (x + 0.5f) * ((float)W / iw) - 0.5f;
      const float sy = (y + 0.5f) * ((float)H / ih) - 0.5f;
      const int xi = (int)std::floor(sx), yi = (int)std::floor(sy);
      const float fx = sx - xi, fy = sy - yi;
      for (int c = 0; c < 3; ++c) {
        const float v = px(yi, xi, c) * (1 - fx) * (1 - fy) + px(yi, xi + 1, c) * fx * (1 - fy)
                      + px(yi + 1, xi, c) * (1 - fx) * fy + px(yi + 1, xi + 1, c) * fx * fy;
        t->data[(size_t)((c * ih + y) * iw + x)] = v / 255.f;
      }
    }
  return t;
}

}  // namespace jl
