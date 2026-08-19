// Perspective rectification: four plate corners -> a canonical square input.
//
// This is the stage that replaces guessing. A detector box is an axis-aligned rectangle around a
// quadrilateral, so the crop it gives depends on the plate's rotation and on the detector's framing —
// and the region head is measurably sensitive to exactly that (0.57 vs 0.92 confidence on the same
// photo with a different detector; 奄美/横浜/練馬 on the same plate with a few percent of margin).
// Warping from the corners removes the variable instead of averaging over it, which is what the
// 6-crop TTA does at 6x the cost.
//
// Conventions: corners are TL, TR, BR, BL in image pixels. `margin` is the border kept around the
// plate in the output, as a fraction of the plate's size, so the recognizer still sees a little
// context (its training crops have the same border).
#pragma once
#include "autograd.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace jl {

// Homography mapping destination (x,y) -> source (u,v), solved from 4 point pairs by Gauss-Jordan.
// Returns false if the quad is degenerate.
inline bool solve_h_dst_to_src(const double dx[4], const double dy[4],
                               const double sx[4], const double sy[4], double H[9]) {
  double A[8][9] = {};
  for (int i = 0; i < 4; ++i) {
    A[2 * i][0] = dx[i]; A[2 * i][1] = dy[i]; A[2 * i][2] = 1;
    A[2 * i][6] = -dx[i] * sx[i]; A[2 * i][7] = -dy[i] * sx[i]; A[2 * i][8] = sx[i];
    A[2 * i + 1][3] = dx[i]; A[2 * i + 1][4] = dy[i]; A[2 * i + 1][5] = 1;
    A[2 * i + 1][6] = -dx[i] * sy[i]; A[2 * i + 1][7] = -dy[i] * sy[i]; A[2 * i + 1][8] = sy[i];
  }
  for (int c = 0; c < 8; ++c) {
    int piv = c;
    for (int r = c + 1; r < 8; ++r) if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
    if (std::fabs(A[piv][c]) < 1e-12) return false;
    if (piv != c) for (int k = 0; k < 9; ++k) std::swap(A[c][k], A[piv][k]);
    for (int r = 0; r < 8; ++r) {
      if (r == c) continue;
      double m = A[r][c] / A[c][c];
      for (int k = c; k < 9; ++k) A[r][k] -= m * A[c][k];
    }
  }
  for (int i = 0; i < 8; ++i) H[i] = A[i][8] / A[i][i];
  H[8] = 1.0;
  return true;
}

// Rectify: sample an out_px x out_px RGB /255 NCHW input from the quad, keeping `margin` of border.
// Falls back to the quad's bounding box if the corners are degenerate.
inline std::vector<float> warp_to_input(const unsigned char* rgb, int W, int H,
                                        const float corners[8], float margin, int out_px = 128) {
  const double m = margin;
  const double lo = m / (1.0 + 2 * m) * out_px, hi = (1.0 + m) / (1.0 + 2 * m) * out_px;
  const double dx[4] = {lo, hi, hi, lo};        // where the plate's corners land in the output
  const double dy[4] = {lo, lo, hi, hi};
  double sx[4], sy[4];
  for (int i = 0; i < 4; ++i) { sx[i] = corners[2 * i]; sy[i] = corners[2 * i + 1]; }

  double Hm[9];
  std::vector<float> out((size_t)3 * out_px * out_px, 0.f);
  auto px = [&](int yy, int xx, int c) {
    yy = std::clamp(yy, 0, H - 1); xx = std::clamp(xx, 0, W - 1);
    return (float)rgb[((size_t)yy * W + xx) * 3 + c];
  };
  if (!solve_h_dst_to_src(dx, dy, sx, sy, Hm)) return out;

  for (int y = 0; y < out_px; ++y)
    for (int x = 0; x < out_px; ++x) {
      const double w = Hm[6] * (x + 0.5) + Hm[7] * (y + 0.5) + Hm[8];
      if (std::fabs(w) < 1e-12) continue;
      const double u = (Hm[0] * (x + 0.5) + Hm[1] * (y + 0.5) + Hm[2]) / w - 0.5;
      const double v = (Hm[3] * (x + 0.5) + Hm[4] * (y + 0.5) + Hm[5]) / w - 0.5;
      const int xi = (int)std::floor(u), yi = (int)std::floor(v);
      const float fx = (float)(u - xi), fy = (float)(v - yi);
      for (int c = 0; c < 3; ++c) {
        const float val = px(yi, xi, c) * (1 - fx) * (1 - fy) + px(yi, xi + 1, c) * fx * (1 - fy)
                        + px(yi + 1, xi, c) * (1 - fx) * fy + px(yi + 1, xi + 1, c) * fx * fy;
        out[(size_t)((c * out_px + y) * out_px + x)] = val / 255.f;
      }
    }
  return out;
}

// The corner regressor's own input: the detector box expanded by `expand`, resampled to in_px.
// Returns the crop and, through box_out, the window it covered (needed to map corners back).
inline std::vector<float> corner_input(const unsigned char* rgb, int W, int H,
                                      float x0, float y0, float x1, float y1,
                                      float expand, int in_px, float box_out[4]) {
  const float bw = x1 - x0, bh = y1 - y0;
  const float cx0 = x0 - bw * expand, cy0 = y0 - bh * expand;
  const float cx1 = x1 + bw * expand, cy1 = y1 + bh * expand;
  box_out[0] = cx0; box_out[1] = cy0; box_out[2] = cx1; box_out[3] = cy1;
  std::vector<float> out((size_t)3 * in_px * in_px, 0.f);
  auto px = [&](int yy, int xx, int c) {
    yy = std::clamp(yy, 0, H - 1); xx = std::clamp(xx, 0, W - 1);
    return (float)rgb[((size_t)yy * W + xx) * 3 + c];
  };
  for (int y = 0; y < in_px; ++y)
    for (int x = 0; x < in_px; ++x) {
      const float sx = cx0 + (x + 0.5f) * (cx1 - cx0) / in_px - 0.5f;
      const float sy = cy0 + (y + 0.5f) * (cy1 - cy0) / in_px - 0.5f;
      const int xi = (int)std::floor(sx), yi = (int)std::floor(sy);
      const float fx = sx - xi, fy = sy - yi;
      for (int c = 0; c < 3; ++c) {
        const float v = px(yi, xi, c) * (1 - fx) * (1 - fy) + px(yi, xi + 1, c) * fx * (1 - fy)
                      + px(yi + 1, xi, c) * (1 - fx) * fy + px(yi + 1, xi + 1, c) * fx * fy;
        out[(size_t)((c * in_px + y) * in_px + x)] = v / 255.f;
      }
    }
  return out;
}

}  // namespace jl
