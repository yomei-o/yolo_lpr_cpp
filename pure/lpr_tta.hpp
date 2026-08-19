// Test-time augmentation for the classifier: read the same plate from several crops and add
// up the per-head probabilities.
//
// WHY. The 9 heads are not equally robust. On a real hand-held 黒ナンバー the class number,
// hiragana and 4-digit number are stable, but the 地域名 head — 130-odd classes decided by a
// few small glyphs at the top of the plate — flips with a few percent of crop margin:
//
//     margin  -10%   奄美 480 ま 4567
//             -5%    横浜 480 り 4567     <- right
//              0%    奄美 480 り 4567
//             +5%    横浜 480 り 4567     <- right
//            +10%    横浜 480 り 4567     <- right
//            +20%    練馬 480 り 4567
//
// A single crop is therefore a coin flip on the region, whether that crop came from a
// detector box or a guide box. Summing the probabilities over a spread of margins picks the
// reading that survives reframing, which is the one actually supported by the glyphs.
//
// Accumulating probabilities rather than counting hard votes matters: a crop that is unsure
// between two regions should not get the same say as one that is certain.
#pragma once
#include "net_lpr.hpp"
#include "lpr_labels.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace lpr {

struct TtaOut {
  std::array<int, 9> arg{};       // winning index per head
  std::array<float, 9> conf{};    // winning share of the summed probability, 0..1
  int crops = 0;
};

// Default spread. Kept modest and symmetric: wide margins pull in background and start
// voting for the wrong thing (see +20% above).
inline const std::vector<float>& tta_margins() {
  static const std::vector<float> m = {-0.06f, -0.03f, 0.0f, 0.03f, 0.06f, 0.10f};
  return m;
}

// Sample a 128x128 network input from an interleaved RGB frame, over the box expanded by
// `margin` (a fraction of the box size). Reads outside the frame clamp to the edge, which is
// what a plate at the image border needs.
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

// Accumulate one crop's head into the running total.
//
// The heads are ALREADY probabilities — net_lpr.hpp applies softmax_rows at inference, and
// each head sums to exactly 1.0. Softmaxing them again (the obvious thing to write) squashes
// a 133-class head to within a whisker of uniform: the region head's winning share came out
// at 0.01 against a uniform 1/133 = 0.0076, so the vote was being decided by noise. Same trap
// as the detector's double sigmoid, one file over.
//
// The assert is the guard: if a future change makes the heads emit logits instead, this must
// fail loudly rather than silently average the wrong thing.
inline void accumulate_probs(const std::vector<float>& probs, std::vector<double>& acc) {
  if (acc.size() != probs.size()) acc.assign(probs.size(), 0.0);
  double s = 0;
  for (float v : probs) s += v;
  if (s < 0.99 || s > 1.01) {
    printf("lpr_tta: head does not sum to 1 (%.4f) -- heads are expected to be softmaxed "
           "(net_lpr.hpp). Refusing to guess.\n", s);
    std::exit(1);
  }
  for (size_t i = 0; i < probs.size(); ++i) acc[i] += probs[i];
}

inline TtaOut recognize_tta(const unsigned char* rgb, int W, int H,
                            float bx0, float by0, float bx1, float by1,
                            LProv& p, const std::vector<float>& margins = tta_margins()) {
  std::array<std::vector<double>, 9> acc;
  TtaOut r;
  for (float m : margins) {
    std::vector<float> v = crop_to_input(rgb, W, H, bx0, by0, bx1, by1, m);
    LprOut o = lpr_forward(from_data({1, 3, 128, 128}, std::move(v)), p, false);
    for (int h = 0; h < 9; ++h) accumulate_probs(o.heads[h]->data, acc[(size_t)h]);
    ++r.crops;
  }
  for (int h = 0; h < 9; ++h) {
    const std::vector<double>& a = acc[(size_t)h];
    int best = 0; double total = 0;
    for (size_t i = 0; i < a.size(); ++i) { total += a[i]; if (a[i] > a[(size_t)best]) best = (int)i; }
    r.arg[(size_t)h] = best;
    r.conf[(size_t)h] = total > 0 ? (float)(a[(size_t)best] / total) : 0.f;
  }
  return r;
}

}  // namespace lpr
