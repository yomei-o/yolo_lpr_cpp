// Detection training data: full frames with plates pasted at every scale, from "the whole car is in
// shot" down to "the plate fills the frame".
//
// WHY THIS EXISTS. Both borrowed detectors only fire on small plates that come with a vehicle around
// them — measured with tools/context_test.py: plate at 8% of the frame width scores 0.83, at 20%
// 0.07, at 30%+ nothing at all. A detector that cannot see a plate held up to the camera is useless
// for the browser demo, and no amount of threshold tuning fixes a training distribution. So the
// synthetic set deliberately spans plate shares of 3%..95% (log-uniform), plus frames with no plate
// at all as hard negatives.
//
// Output is a standard YOLO layout so either language's trainer (or Ultralytics) can read it:
//   <out>/images/det%06d.png
//   <out>/labels/det%06d.txt   lines: `0 xc yc w h` (normalised, class 0 = plate)
//   <out>/corners.txt          `<file> <plate#> x1 y1 .. x4 y4` (normalised) — for the corner net
//   <out>/meta.txt             every draw, for the C++/Python parity test
#pragma once
#include "gen_render.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace gen {

struct DetPlate {
  Params p;
  double share = 0.2, cx = 0.5, cy = 0.5;
  int font_idx = 0;
  bool use_real = false;         // paste a real plate photo instead of the drawn art
  int real_idx = 0;
  Quad quad{};
  double bx0 = 0, by0 = 0, bx1 = 0, by1 = 0;    // clipped bbox in pixels
  bool keep = true;
};

struct DetSample {
  int n_plates = 1;
  bool bg_real = false;
  int bg_idx = 0;
  double bg_hue = 0, bg_dark = 0.5;
  std::vector<DetPlate> plates;
  double brightness = 1, contrast = 1, warm = 0, blur = 0, motion = 0, noise = 0;
  int jpeg_q = 90;
};

// Draw order (spec/gen.md, detection section). Sampling is separated from rendering so --meta-only
// can reproduce the labels without fonts or images.
// `real_pct` = percentage of plates that are pasted from real photos instead of drawn. Real crops
// carry the actual plate typeface, lighting and dirt, which is what the detector needs: trained on
// drawn plates only, it reached mAP50 0.99 on synthetic frames and still scored 0.1 on a real photo.
inline DetSample sample_det(Rng& rng, const spec::Spec& sp, int n_bg, int n_fonts,
                           int n_real = 0, int real_pct = 0) {
  DetSample d;
  uint64_t np = rng.below(100);                                     // D1
  d.n_plates = np < 8 ? 0 : (np < 75 ? 1 : (np < 95 ? 2 : 3));
  d.bg_real = rng.below(100) < 70 && n_bg > 0;                      // D2
  d.bg_idx = n_bg > 0 ? (int)rng.below((uint64_t)n_bg) : 0;         // D3 (always drawn)
  d.bg_hue = rng.range(0, 1);                                       // D4
  d.bg_dark = rng.range(0.15, 0.85);                                // D5
  for (int k = 0; k < d.n_plates; ++k) {
    DetPlate pl;
    pl.p = sample(rng, sp);                                         // D6: the 28 plate draws
    double lo = std::log(0.03), hi = std::log(0.95);
    pl.share = std::exp(rng.range(lo, hi));                         // D7: log-uniform scale
    pl.cx = rng.range(0.15, 0.85);                                  // D8
    pl.cy = rng.range(0.15, 0.85);                                  // D9
    pl.font_idx = n_fonts > 0 ? (int)rng.below((uint64_t)n_fonts) : 0;  // D10
    // D11/D12 are always drawn (even with no real plates available) so the stream depends only on
    // the flags, never on what happens to be on disk.
    pl.use_real = (int)rng.below(100) < real_pct && n_real > 0;         // D11
    pl.real_idx = (int)rng.below((uint64_t)(n_real > 0 ? n_real : 1));  // D12
    d.plates.push_back(pl);
  }
  d.brightness = rng.range(0.5, 1.3);                               // D13
  d.contrast = rng.range(0.75, 1.25);                               // D14
  d.warm = rng.range(-0.1, 0.1);                                    // D15
  d.blur = rng.range(0, 1.2);                                       // D16
  d.motion = rng.range(0, 2.0);                                     // D17
  d.noise = rng.range(0, 8);                                        // D18
  d.jpeg_q = (int)rng.below(50) + 50;                               // D19
  return d;
}

// Fill in quads/bboxes for the sampled plates at a given image size (no rendering needed).
inline void place_det(DetSample& d, int imgsz) {
  for (DetPlate& pl : d.plates) {
    double px = pl.share * imgsz;
    pl.quad = project_plate(pl.p, pl.cx * imgsz, pl.cy * imgsz);
    double minx = pl.quad.x[0], maxx = pl.quad.x[0], miny = pl.quad.y[0], maxy = pl.quad.y[0];
    (void)px;
    for (int i = 1; i < 4; ++i) {
      minx = std::min(minx, pl.quad.x[i]); maxx = std::max(maxx, pl.quad.x[i]);
      miny = std::min(miny, pl.quad.y[i]); maxy = std::max(maxy, pl.quad.y[i]);
    }
    double full = std::max(1.0, (maxx - minx) * (maxy - miny));
    pl.bx0 = std::clamp(minx, 0.0, (double)imgsz);
    pl.by0 = std::clamp(miny, 0.0, (double)imgsz);
    pl.bx1 = std::clamp(maxx, 0.0, (double)imgsz);
    pl.by1 = std::clamp(maxy, 0.0, (double)imgsz);
    double vis = std::max(0.0, pl.bx1 - pl.bx0) * std::max(0.0, pl.by1 - pl.by0);
    pl.keep = vis / full >= 0.35 && (pl.bx1 - pl.bx0) >= 6 && (pl.by1 - pl.by0) >= 4;
  }
}

// project_plate() sizes the plate from p.plate_px, so the share has to be written into the params
// before projecting. Called by place_det via a copy — kept explicit to avoid a silent mismatch.
inline void apply_share(DetSample& d, int imgsz) {
  for (DetPlate& pl : d.plates) pl.p.plate_px = pl.share * imgsz;
}

inline std::string det_labels(const DetSample& d, int imgsz) {
  std::string out;
  char b[160];
  for (const DetPlate& pl : d.plates) {
    if (!pl.keep) continue;
    double xc = (pl.bx0 + pl.bx1) * 0.5 / imgsz, yc = (pl.by0 + pl.by1) * 0.5 / imgsz;
    double w = (pl.bx1 - pl.bx0) / imgsz, h = (pl.by1 - pl.by0) / imgsz;
    snprintf(b, sizeof b, "0 %.6f %.6f %.6f %.6f\n", xc, yc, w, h);
    out += b;
  }
  return out;
}

inline std::string det_corners(const std::string& file, const DetSample& d, int imgsz) {
  std::string out;
  char b[256];
  for (size_t k = 0; k < d.plates.size(); ++k) {
    const DetPlate& pl = d.plates[k];
    if (!pl.keep) continue;
    snprintf(b, sizeof b, "%s %zu %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n", file.c_str(), k,
             pl.quad.x[0] / imgsz, pl.quad.y[0] / imgsz, pl.quad.x[1] / imgsz, pl.quad.y[1] / imgsz,
             pl.quad.x[2] / imgsz, pl.quad.y[2] / imgsz, pl.quad.x[3] / imgsz, pl.quad.y[3] / imgsz);
    out += b;
  }
  return out;
}

inline std::string det_meta(const std::string& file, const DetSample& d, const spec::Spec& sp,
                           const std::string& bg_name, int imgsz) {
  std::string out;
  char b[512];
  int kept = 0;
  for (const DetPlate& pl : d.plates) kept += pl.keep ? 1 : 0;
  snprintf(b, sizeof b, "%s imgsz=%d plates=%d kept=%d bg=%s hue=%.4f dark=%.4f "
           "brightness=%.4f contrast=%.4f warm=%.4f blur=%.4f motion=%.4f noise=%.4f jpeg_q=%d\n",
           file.c_str(), imgsz, d.n_plates, kept, bg_name.c_str(), d.bg_hue, d.bg_dark,
           d.brightness, d.contrast, d.warm, d.blur, d.motion, d.noise, d.jpeg_q);
  out += b;
  for (size_t k = 0; k < d.plates.size(); ++k) {
    const DetPlate& pl = d.plates[k];
    snprintf(b, sizeof b, "  %zu share=%.4f cx=%.4f cy=%.4f keep=%d text=%s kind=%s\n", k, pl.share,
             pl.cx, pl.cy, pl.keep ? 1 : 0, plate_text(pl.p, sp).c_str(),
             sp.head("plate_kind").tok[pl.p.kind].c_str());
    out += b;
  }
  return out;
}

}  // namespace gen
