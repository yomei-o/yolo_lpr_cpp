// Synthetic plate generator — the sampler half.
//
// Every random decision for one sample is drawn here, in the order fixed by spec/gen.md, from the
// shared splitmix64 (pure/rng.hpp). tools/gen.py draws the same values in the same order from the
// same seed, and tools/parity/gen.py diffs the two meta dumps byte for byte. Rendering (the other
// half, gen_render.hpp) cannot match across rasterisers, so parity stops at labels + geometry.
//
// Nothing about the plate's contents is hardcoded: the class lists and the per-kind hiragana
// subsets come from spec/labels.txt via spec.hpp.
#pragma once
#include "rng.hpp"
#include "spec.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace gen {

// plate_kind tokens, in the order spec/labels.txt declares them
enum Kind { PRIVATE_NORMAL = 0, COMMERCIAL_NORMAL = 1, KEI_PRIVATE = 2, KEI_COMMERCIAL = 3,
            LARGE = 4, DESIGN = 5, OTHER = 6, UNKNOWN = 7 };

struct Rgb { int r, g, b; };

struct Params {
  // --- contents -------------------------------------------------------------------------------
  int kind = PRIVATE_NORMAL;
  int region = 0;                       // index into the region head
  int cls_len = 3;
  int cls_idx[3] = {0, 0, 0};           // indices into class_num_01/02/03
  std::string cls_text;
  int hira = 0;                         // index into the hiragana head
  int serial_len = 4;
  int serial_idx[4] = {10, 10, 10, 0};  // indices into plate_num_01..04 (10 = blank, right-aligned)
  std::string serial_text;              // digits only
  // --- geometry ------------------------------------------------------------------------------
  double yaw = 0, pitch = 0, roll = 0;
  double plate_px = 100, margin = 0.1, off_x = 0, off_y = 0;
  // --- photometry / degradation ---------------------------------------------------------------
  double brightness = 1, contrast = 1, warm = 0, blur = 0, motion = 0, noise = 0, dirt = 0;
  int jpeg_q = 90;
  int bolts = 2;                        // 0, 2 or 4
  bool seal = true, frame = false;
  double bg_hue = 0, bg_dark = 0.5;
  // --- derived --------------------------------------------------------------------------------
  bool large = false, design = false;
  Rgb bg{250, 250, 248}, fg{16, 90, 60};
  bool legible = true;

  std::vector<int> heads() const {       // the 11 head indices, in spec head order
    return {region, cls_idx[0], cls_idx[1], cls_idx[2], hira,
            serial_idx[0], serial_idx[1], serial_idx[2], serial_idx[3], kind, legible ? 1 : 0};
  }
};

inline int kind_from_draw(uint64_t d) {
  if (d < 50) return PRIVATE_NORMAL;
  if (d < 65) return KEI_PRIVATE;
  if (d < 75) return COMMERCIAL_NORMAL;
  if (d < 80) return KEI_COMMERCIAL;
  if (d < 90) return LARGE;
  return DESIGN;
}

inline bool commercial(int kind) { return kind == COMMERCIAL_NORMAL || kind == KEI_COMMERCIAL; }

inline void palette_of(int kind, Rgb& bg, Rgb& fg) {
  switch (kind) {
    case COMMERCIAL_NORMAL: bg = {16, 90, 60};    fg = {250, 250, 248}; break;
    case KEI_PRIVATE:       bg = {240, 205, 20};  fg = {25, 25, 25};    break;
    case KEI_COMMERCIAL:    bg = {25, 25, 25};    fg = {240, 205, 20};  break;
    default:                bg = {250, 250, 248}; fg = {16, 90, 60};    break;   // private / large / design
  }
}

// Draw one sample. The order of rng calls is the contract — see spec/gen.md.
inline Params sample(Rng& rng, const spec::Spec& sp) {
  Params p;
  const spec::Group& reg = sp.head("region");
  const spec::Group& cn2 = sp.head("class_num_02");
  const spec::Group& hira_all = sp.head("hiragana");

  p.kind = kind_from_draw(rng.below(100));                                  // 1
  p.region = (int)rng.below((uint64_t)reg.n);                               // 2
  uint64_t cl = rng.below(100);                                            // 3
  p.cls_len = cl < 80 ? 3 : (cl < 95 ? 2 : 1);
  int first = (int)rng.below(9) + 1;                                       // 4
  p.cls_idx[0] = first;                                                    // class_num_01 tokens are "0".."9"
  p.cls_text = std::string(1, (char)('0' + first));
  p.cls_idx[1] = p.cls_idx[2] = -1;
  for (int i = 1; i < p.cls_len; ++i) {                                    // 5
    int t = (int)rng.below((uint64_t)cn2.n);
    p.cls_idx[i] = t;
    p.cls_text += cn2.tok[t];
  }
  // unfilled class-number digits: class_num_02 has no blank token, class_num_03 does (index 20)
  if (p.cls_idx[1] < 0) p.cls_idx[1] = 0;                                  // 2桁未満は 0 を置く（描画しない）
  if (p.cls_idx[2] < 0) p.cls_idx[2] = sp.index_of("class_num_03", "<blank>");

  const spec::Group& hset = *sp.find(commercial(p.kind) ? "hira_commercial" : "hira_private");
  int hsel = (int)rng.below((uint64_t)hset.n);                             // 6
  p.hira = sp.index_of("hiragana", hset.tok[hsel]);

  uint64_t sl = rng.below(100);                                            // 7
  p.serial_len = sl < 85 ? 4 : (sl < 95 ? 3 : (sl < 99 ? 2 : 1));
  for (int i = 0; i < 4; ++i) p.serial_idx[i] = 10;                        // blank
  p.serial_text.clear();
  for (int i = 0; i < p.serial_len; ++i) {                                 // 8
    int d = (int)rng.below(10);
    int slot = 4 - p.serial_len + i;                                       // right aligned
    if (i == p.serial_len - 1 && d == 0 && p.serial_text.find_first_not_of('0') == std::string::npos)
      d = 1;                                                               // never an all-zero serial
    p.serial_idx[slot] = d;
    p.serial_text += (char)('0' + d);
  }
  p.serial_idx[3] = p.serial_idx[3] == 10 ? 0 : p.serial_idx[3];           // head 04 has no blank class

  p.yaw = rng.range(-35, 35);                                              // 9
  p.pitch = rng.range(-20, 20);                                            // 10
  p.roll = rng.range(-8, 8);                                               // 11
  p.plate_px = rng.range(24, 200);                                         // 12
  p.margin = rng.range(0.02, 0.18);                                        // 13
  p.off_x = rng.range(-0.05, 0.05);                                        // 14
  p.off_y = rng.range(-0.05, 0.05);                                        // 15
  p.brightness = rng.range(0.45, 1.35);                                    // 16
  p.contrast = rng.range(0.7, 1.3);                                        // 17
  p.warm = rng.range(-0.12, 0.12);                                         // 18
  p.blur = rng.range(0, 1.6);                                              // 19
  p.motion = rng.range(0, 3.0);                                            // 20
  p.noise = rng.range(0, 12);                                              // 21
  p.jpeg_q = (int)rng.below(60) + 40;                                      // 22
  p.dirt = rng.range(0, 0.35);                                             // 23
  uint64_t bd = rng.below(100);                                            // 24
  p.bolts = bd < 80 ? 2 : (bd < 95 ? 0 : 4);
  p.seal = rng.below(100) < 70;                                            // 25
  p.frame = rng.below(100) >= 85;                                          // 26
  p.bg_hue = rng.range(0, 1);                                              // 27
  p.bg_dark = rng.range(0.15, 0.85);                                       // 28

  p.large = (p.kind == LARGE);
  p.design = (p.kind == DESIGN);
  palette_of(p.kind, p.bg, p.fg);
  p.legible = !(p.plate_px < 40 || p.blur > 1.2 || p.motion > 2.2 || p.noise > 9 || p.dirt > 0.28);
  return p;
}

// Debug / curriculum helper: keep the contents and the framing, drop the degradation. The rng
// draws already happened, so --clean does not disturb parity or reproducibility; it just makes the
// image easy. Used to separate "our plate art is wrong" from "our degradation is too harsh".
inline void make_clean(Params& p) {
  p.brightness = 1.0; p.contrast = 1.0; p.warm = 0.0;
  p.blur = 0.0; p.motion = 0.0; p.noise = 0.0; p.dirt = 0.0;
  p.jpeg_q = 97;
  if (p.plate_px < 140) p.plate_px = 140;
  p.yaw *= 0.25; p.pitch *= 0.25; p.roll *= 0.25;
  p.legible = true;
}

// ---- dumps that must match Python byte for byte ---------------------------------------------
inline std::string labels_line(const std::string& file, const Params& p) {
  std::string s = file;
  for (int v : p.heads()) s += " " + std::to_string(v);
  return s + "\n";
}

inline std::string meta_line(const std::string& file, const Params& p, const spec::Spec& sp,
                             const std::string& font) {
  char b[1024];
  const spec::Group& kg = sp.head("plate_kind");
  snprintf(b, sizeof b,
           "%s kind=%s region=%d cls=%s hira=%d serial=%s yaw=%.4f pitch=%.4f roll=%.4f "
           "plate_px=%.4f margin=%.4f off_x=%.4f off_y=%.4f brightness=%.4f contrast=%.4f "
           "warm=%.4f blur=%.4f motion=%.4f noise=%.4f jpeg_q=%d dirt=%.4f bolts=%d seal=%d "
           "frame=%d bg_hue=%.4f bg_dark=%.4f legible=%d font=%s\n",
           file.c_str(), kg.tok[p.kind].c_str(), p.region, p.cls_text.c_str(), p.hira,
           p.serial_text.c_str(), p.yaw, p.pitch, p.roll, p.plate_px, p.margin, p.off_x, p.off_y,
           p.brightness, p.contrast, p.warm, p.blur, p.motion, p.noise, p.jpeg_q, p.dirt, p.bolts,
           p.seal ? 1 : 0, p.frame ? 1 : 0, p.bg_hue, p.bg_dark, p.legible ? 1 : 0, font.c_str());
  return b;
}

// Human-readable plate string, for eyeballing a batch (uses the same decode as inference).
inline std::string plate_text(const Params& p, const spec::Spec& sp) {
  return spec::decode(sp, p.heads()).text;
}

}  // namespace gen
