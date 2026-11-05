// The one-pass pipeline: frame -> plate boxes -> normalised 128x128 crop -> per-position heads
// -> plate string. Everything runs from ONNX files through the shared runner (onnx_run.hpp), so
// the CLI, the WASM demo and the Python side all execute the same graphs.
//
// Stage 2 (4-corner rectification) is not in yet — until it is, the crop is the detector box
// widened by a margin, read at several margins and summed (see `read_plate_tta`), because the
// region head flips with a few percent of framing. That TTA is a stopgap for the missing
// rectification, not the fix; see README.
#pragma once
#include "onnx.hpp"
#include "onnx_run.hpp"
#include "infer_yolox.hpp"
#include "infer_v8.hpp"
#include "crop.hpp"
#include "warp.hpp"
#include "spec.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace jl {

// ---- detector -------------------------------------------------------------------------------
// The interim detector is lpr_cpp's ReLU plate yolox-tiny (8 classes, class 7 = plate) at 416.
// It is ONNX-fed, so its head already carries probabilities: logits=false. A yolov8 nc=1
// detector replaces it at M6 (see RESUME) — the box interface here does not change.
// Two detector kinds are supported, because the two available graphs are shaped differently:
//   YOLOX  — the interim ReLU plate yolox-tiny: letterbox 416, BGR 0-255, three per-level heads
//            that this code decodes (anchor-free, obj*cls, already sigmoided in the ONNX).
//   V8     — a YOLOv8/v11/v12 head: plain resize, RGB 0-1, one [1,4+nc,N] tensor already carrying
//            pixel-space boxes and sigmoided scores. PlateYOLO-JP with the NMS tail stripped is
//            this shape, and so is our own detector once M7 trains it.
enum class DetKind { YOLOX, V8 };

struct DetCfg {
  DetKind kind = DetKind::YOLOX;
  int imgsz = 416;                    // 0 = take it from the graph's declared input dims
  int plate_class = 7;                // YOLOX: 8 classes, plate is 7. V8 plate-only: 0
  float conf = 0.15f;                 // low on purpose: black/yellow plates score ~1/4 of white
  float nms = 0.45f;
  bool logits = false;
  std::vector<std::string> heads = {"/head/Concat_output_0", "/head/Concat_1_output_0",
                                    "/head/Concat_2_output_0"};
  std::vector<int64_t> strides = {8, 16, 32};
  int64_t nc = 8;
  std::string v8_out;                 // empty = the graph's first declared output
  BoxFmt v8_fmt = BoxFmt::XYXY;
};

struct Box { float x1, y1, x2, y2, score; };

// Input size the graph itself declares (NCHW); 0 if dynamic/unknown.
inline void graph_input_hw(const onx::Graph& g, int& iw, int& ih) {
  iw = ih = 0;
  for (const auto& vi : g.inputs) {
    if (vi.dims.size() == 4 && vi.dims[2] > 0 && vi.dims[3] > 0) {
      ih = (int)vi.dims[2];
      iw = (int)vi.dims[3];
      return;
    }
  }
}

inline std::vector<Box> detect_plates(const onx::Graph& det, const unsigned char* rgb, int W, int H,
                                      const DetCfg& cfg, std::vector<Det>* all_out = nullptr) {
  std::vector<Det> dets;
  float sx = 1.f, sy = 1.f;                                  // net pixels -> image pixels
  if (cfg.kind == DetKind::YOLOX) {
    float scale = 1.f;
    Tensor x = letterbox_bgr(rgb, W, H, cfg.imgsz, scale);
    std::set<std::string> stop(cfg.heads.begin(), cfg.heads.end());
    auto vals = onx::run_onnx(det, x, stop);
    std::vector<Tensor> raw;
    for (auto& h : cfg.heads) raw.push_back(vals.at(h));
    dets = yolox_detect(raw, cfg.strides, cfg.nc, cfg.conf, cfg.nms, cfg.logits);
    sx = sy = 1.f / scale;
  } else {
    int iw = 0, ih = 0;
    graph_input_hw(det, iw, ih);
    if (cfg.imgsz > 0) { iw = ih = cfg.imgsz; }
    if (iw <= 0 || ih <= 0) { printf("detect_plates: cannot tell the V8 graph's input size\n"); std::exit(1); }
    Tensor x = resize_rgb01(rgb, W, H, iw, ih);
    std::string out = cfg.v8_out.empty() ? (det.outputs.empty() ? std::string() : det.outputs[0].name)
                                         : cfg.v8_out;
    auto vals = onx::run_onnx(det, x, {out});
    dets = v8_detect(vals.at(out), cfg.nc, cfg.conf, cfg.nms, cfg.v8_fmt);
    sx = (float)W / iw;
    sy = (float)H / ih;
  }
  if (all_out) *all_out = dets;
  std::vector<Box> out;
  for (const Det& d : dets) {
    if (d.cls != cfg.plate_class) continue;
    Box b{d.x1 * sx, d.y1 * sy, d.x2 * sx, d.y2 * sy, d.score};
    b.x1 = std::clamp(b.x1, 0.f, (float)W - 1); b.x2 = std::clamp(b.x2, 0.f, (float)W - 1);
    b.y1 = std::clamp(b.y1, 0.f, (float)H - 1); b.y2 = std::clamp(b.y2, 0.f, (float)H - 1);
    out.push_back(b);
  }
  std::sort(out.begin(), out.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
  return out;
}

// ---- recognizer -----------------------------------------------------------------------------
struct Read {
  std::vector<int> arg;               // winning index per head (graph output order)
  std::vector<float> conf;            // winning share of the summed probability, 0..1
  int crops = 0;
  spec::Plate plate;
  std::vector<std::vector<float>> probs;   // per-head probabilities, for the frame-to-frame tracker
};

// Graph outputs in declaration order = head order (region, class_num_01..03, hiragana,
// plate_num_01..04, [plate_kind, legible]). A 9-output legacy model simply has the last two
// missing; decode() then reports '?' for them rather than inventing a value.
inline std::vector<std::string> ocr_head_names(const onx::Graph& g) {
  std::vector<std::string> names;
  for (auto& o : g.outputs) names.push_back(o.name);
  if (names.empty() && !g.nodes.empty()) names.push_back(g.nodes.back().output[0]);
  return names;
}

inline const std::vector<float>& default_margins() {
  static const std::vector<float> m = {-0.06f, -0.03f, 0.0f, 0.03f, 0.06f, 0.10f};
  return m;
}

inline Read read_plate(const onx::Graph& ocr, const spec::Spec& sp, const unsigned char* rgb,
                       int W, int H, float x0, float y0, float x1, float y1,
                       const std::vector<float>& margins) {
  std::vector<std::string> hn = ocr_head_names(ocr);
  std::vector<std::vector<double>> sum(hn.size());
  for (float m : margins) {
    std::vector<float> in = crop_to_input(rgb, W, H, x0, y0, x1, y1, m);
    Tensor xt = from_data({1, 3, 128, 128}, std::move(in));
    auto vals = onx::run_onnx(ocr, xt);
    for (size_t h = 0; h < hn.size(); ++h) {
      const std::vector<float>& p = vals.at(hn[h])->data;   // already softmaxed by the graph
      if (sum[h].empty()) sum[h].assign(p.size(), 0.0);
      for (size_t i = 0; i < p.size(); ++i) sum[h][i] += p[i];
    }
  }
  Read r;
  r.crops = (int)margins.size();
  for (size_t h = 0; h < hn.size(); ++h) {
    int best = 0; double bv = -1, tot = 0;
    for (size_t i = 0; i < sum[h].size(); ++i) { tot += sum[h][i]; if (sum[h][i] > bv) { bv = sum[h][i]; best = (int)i; } }
    r.arg.push_back(best);
    r.conf.push_back(tot > 0 ? (float)(bv / tot) : 0.f);
    std::vector<float> pv(sum[h].size());
    for (size_t i = 0; i < sum[h].size(); ++i) pv[i] = (float)(tot > 0 ? sum[h][i] / tot : 0.0);
    r.probs.push_back(std::move(pv));      // normalised, so frames contribute equally to a track
  }
  r.plate = spec::decode(sp, r.arg);
  return r;
}

inline Read read_plate_single(const onx::Graph& ocr, const spec::Spec& sp, const unsigned char* rgb,
                             int W, int H, float x0, float y0, float x1, float y1) {
  static const std::vector<float> one = {0.f};
  return read_plate(ocr, sp, rgb, W, H, x0, y0, x1, y1, one);
}

inline Read read_plate_tta(const onx::Graph& ocr, const spec::Spec& sp, const unsigned char* rgb,
                          int W, int H, float x0, float y0, float x1, float y1) {
  return read_plate(ocr, sp, rgb, W, H, x0, y0, x1, y1, default_margins());
}

// ---- stage 2: corner regression + rectification ---------------------------------------------
struct CornerCfg {
  int in_px = 64;
  float expand = 0.25f;          // how much context around the box the corner net sees
  float margin = 0.06f;          // border kept around the plate in the rectified crop
  // Measured 2026-08-19: rectification is worth it when the box comes from a detector on a full
  // scene (real close-up black plate: region confidence 0.42 -> 0.85) and it HURTS when the box is
  // small or the image is already a tight plate crop (bus photo, 73 px box: 横浜 0.96 -> 佐賀 0.52;
  // alpr_jp pre-cropped plates: 92.7% -> 83.3%). The corner net only ever saw plates with real
  // surroundings, so a box with nothing around it is out of its distribution. Hence a size gate
  // instead of a blanket "always rectify".
  float min_box_px = 90.f;
  int refine = 0;                // extra cascade passes of the corner net (refine_corners)
  // Net for the refine passes (nullptr = the same one). The pair that works: a net trained on wide
  // angles for the first look, the original near-frontal specialist for the refinement, since after
  // one warp the plate it sees IS near-frontal.
  const onx::Graph* refine_net = nullptr;
};

inline bool refine_corners(const onx::Graph& corner, const unsigned char* rgb, int W, int H,
                           const CornerCfg& cfg, float c[8]);

// Predict the plate's 4 corners (image pixels, TL/TR/BR/BL) from a detector box.
inline bool predict_corners(const onx::Graph& corner, const unsigned char* rgb, int W, int H,
                           const Box& b, const CornerCfg& cfg, float out[8]) {
  if ((b.x2 - b.x1) < cfg.min_box_px) return false;      // too small: the box crop is better
  float win[4];
  std::vector<float> in = corner_input(rgb, W, H, b.x1, b.y1, b.x2, b.y2, cfg.expand, cfg.in_px, win);
  Tensor x = from_data({1, 3, cfg.in_px, cfg.in_px}, std::move(in));
  auto vals = onx::run_onnx(corner, x);
  std::string name = corner.outputs.empty() ? std::string() : corner.outputs[0].name;
  if (name.empty() || !vals.count(name)) return false;
  const std::vector<float>& p = vals.at(name)->data;
  if (p.size() < 8) return false;
  for (int i = 0; i < 4; ++i) {                        // normalised crop coords -> image pixels
    out[2 * i] = win[0] + p[2 * i] * (win[2] - win[0]);
    out[2 * i + 1] = win[1] + p[2 * i + 1] * (win[3] - win[1]);
  }
  for (int k = 0; k < cfg.refine; ++k) {
    float c2[8];
    std::copy(out, out + 8, c2);
    if (!refine_corners(cfg.refine_net ? *cfg.refine_net : corner, rgb, W, H, cfg, c2)) break;
    std::copy(c2, c2 + 8, out);
  }
  return true;
}

// Cascade refinement: re-run the corner net on a view that is already rectified by the previous
// estimate, then map its answer back through the same homography.
//
// Why: the corner net was trained on axis-aligned box crops of plates within roll +-8 / yaw +-35 /
// pitch +-20. A plate rolled 25 degrees is outside that, so the first estimate is only roughly right
// — but roughly right is enough to undo most of the rotation. The second view shows a near-frontal
// plate framed exactly like a training crop (plate filling the inner 1/(1+2*expand) of the window),
// which is the net's home ground, so its answer is sharp. Each pass costs one 64x64 forward (~1 ms).
inline bool refine_corners(const onx::Graph& corner, const unsigned char* rgb, int W, int H,
                           const CornerCfg& cfg, float c[8]) {
  const int S = cfg.in_px;
  const double e = cfg.expand;
  const double lo = e / (1 + 2 * e) * S, hi = (1 + e) / (1 + 2 * e) * S;
  const double dx[4] = {lo, hi, hi, lo}, dy[4] = {lo, lo, hi, hi};
  double sx[4], sy[4], Hm[9];
  for (int i = 0; i < 4; ++i) { sx[i] = c[2 * i]; sy[i] = c[2 * i + 1]; }
  if (!solve_h_dst_to_src(dx, dy, sx, sy, Hm)) return false;
  auto map = [&](double x, double y, double& u, double& v) {
    const double w = Hm[6] * x + Hm[7] * y + Hm[8];
    if (std::fabs(w) < 1e-12) return false;
    u = (Hm[0] * x + Hm[1] * y + Hm[2]) / w;
    v = (Hm[3] * x + Hm[4] * y + Hm[5]) / w;
    return true;
  };
  std::vector<float> in((size_t)3 * S * S, 0.f);
  auto px = [&](int yy, int xx, int ch) {
    yy = std::clamp(yy, 0, H - 1); xx = std::clamp(xx, 0, W - 1);
    return (float)rgb[((size_t)yy * W + xx) * 3 + ch];
  };
  for (int y = 0; y < S; ++y)
    for (int x = 0; x < S; ++x) {
      double u, v;
      if (!map(x + 0.5, y + 0.5, u, v)) continue;
      u -= 0.5; v -= 0.5;
      const int xi = (int)std::floor(u), yi = (int)std::floor(v);
      const float fx = (float)(u - xi), fy = (float)(v - yi);
      for (int ch = 0; ch < 3; ++ch) {
        const float val = px(yi, xi, ch) * (1 - fx) * (1 - fy) + px(yi, xi + 1, ch) * fx * (1 - fy)
                        + px(yi + 1, xi, ch) * (1 - fx) * fy + px(yi + 1, xi + 1, ch) * fx * fy;
        in[(size_t)((ch * S + y) * S + x)] = val / 255.f;
      }
    }
  Tensor xt = from_data({1, 3, S, S}, std::move(in));
  auto vals = onx::run_onnx(corner, xt);
  std::string name = corner.outputs.empty() ? std::string() : corner.outputs[0].name;
  if (name.empty() || !vals.count(name)) return false;
  const std::vector<float>& p = vals.at(name)->data;
  if (p.size() < 8) return false;
  float nc[8];
  for (int i = 0; i < 4; ++i) {
    double u, v;
    if (!map(p[2 * i] * S, p[2 * i + 1] * S, u, v)) return false;
    nc[2 * i] = (float)u; nc[2 * i + 1] = (float)v;
  }
  for (int i = 0; i < 8; ++i) c[i] = nc[i];
  return true;
}

// ---- rotation search -------------------------------------------------------------------------
// The detector was trained with +-5 degree rotation augmentation and plates within roll +-8, and it
// shows: measured with `jlpr rot-test`, it finds 100% of plates up to 20 degrees of roll, 88% at 30,
// and 3% at 45. Rather than retrain it, look again at the frame rotated by +-angle: a plate rolled
// 45 degrees is a 15-degree plate in the frame rotated by -30. Whatever is found there gets its
// corners in that frame (where the corner net also sees a near-upright plate) and the corners are
// mapped back, so everything after — warp, read, JSON — works in original image coordinates.
struct PlateHit {
  Box box;            // axis-aligned bbox of `quad` in the original image (for NMS / display)
  float quad[8];      // TL TR BR BL in the original image
  bool cornered;      // quad came from the corner net (else: the detector box, mapped back)
  float angle;        // degrees the frame was rotated by when this was found
  Box rbox;           // the detector box in that rotated frame (== box when angle is 0)
};

// Rotate the frame by `deg` (counter-clockwise on screen) about its centre onto a canvas big enough
// to hold all of it. `back` maps canvas pixels to original pixels: p_orig = A * p_rot + t.
inline std::vector<unsigned char> rotate_frame(const unsigned char* rgb, int W, int H, float deg,
                                               int& OW, int& OH, double back[6]) {
  const double r = deg * 3.14159265358979 / 180.0, c = std::cos(r), s = std::sin(r);
  OW = (int)std::ceil(std::fabs(W * c) + std::fabs(H * s));
  OH = (int)std::ceil(std::fabs(W * s) + std::fabs(H * c));
  const double icx = W * 0.5, icy = H * 0.5, ocx = OW * 0.5, ocy = OH * 0.5;
  // on screen (y down) a CCW turn by r is x' = c*x + s*y, y' = -s*x + c*y; its inverse is the transpose
  back[0] = c;  back[1] = -s; back[2] = icx - c * ocx + s * ocy;
  back[3] = s;  back[4] = c;  back[5] = icy - s * ocx - c * ocy;
  std::vector<unsigned char> out((size_t)OW * OH * 3, 114);
  for (int y = 0; y < OH; ++y)
    for (int x = 0; x < OW; ++x) {
      const double u = back[0] * (x + 0.5) + back[1] * (y + 0.5) + back[2] - 0.5;
      const double v = back[3] * (x + 0.5) + back[4] * (y + 0.5) + back[5] - 0.5;
      if (u < -0.5 || v < -0.5 || u > W - 0.5 || v > H - 0.5) continue;
      const int xi = (int)std::floor(u), yi = (int)std::floor(v);
      const float fx = (float)(u - xi), fy = (float)(v - yi);
      for (int ch = 0; ch < 3; ++ch) {
        auto px = [&](int yy, int xx) {
          yy = std::clamp(yy, 0, H - 1); xx = std::clamp(xx, 0, W - 1);
          return (float)rgb[((size_t)yy * W + xx) * 3 + ch];
        };
        const float val = px(yi, xi) * (1 - fx) * (1 - fy) + px(yi, xi + 1) * fx * (1 - fy)
                        + px(yi + 1, xi) * (1 - fx) * fy + px(yi + 1, xi + 1) * fx * fy;
        out[((size_t)y * OW + x) * 3 + ch] = (unsigned char)std::clamp(val + 0.5f, 0.f, 255.f);
      }
    }
  return out;
}

inline float box_iou(const Box& a, const Box& b) {
  const float ix = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float iy = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float in = ix * iy;
  const float un = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - in;
  return un > 0 ? in / un : 0.f;
}

// Detect (and, if a corner net is given, corner) every plate, searching the listed rotations.
// angles = {0} is exactly the old behaviour. Duplicates found at several angles are merged by NMS
// on the mapped-back bbox, keeping the highest detector score. `only_if_empty`: stop as soon as an
// angle finds something — the cheap mode for slow targets (WASM), where an upright plate must not
// pay for the search; listed first, angle 0 then costs exactly what it did before.
inline std::vector<PlateHit> find_plates(const onx::Graph& det, const onx::Graph* corner,
                                         const unsigned char* rgb, int W, int H, const DetCfg& dcfg,
                                         const CornerCfg& ccfg, const std::vector<float>& angles,
                                         bool only_if_empty = false) {
  std::vector<PlateHit> all;
  for (float ang : angles) {
    if (only_if_empty && !all.empty()) break;
    int RW = W, RH = H;
    double back[6] = {1, 0, 0, 0, 1, 0};
    std::vector<unsigned char> rot;
    const unsigned char* im = rgb;
    if (ang != 0.f) { rot = rotate_frame(rgb, W, H, ang, RW, RH, back); im = rot.data(); }
    for (const Box& b : detect_plates(det, im, RW, RH, dcfg)) {
      PlateHit h;
      h.angle = ang;
      h.rbox = b;
      h.cornered = corner && predict_corners(*corner, im, RW, RH, b, ccfg, h.quad);
      if (!h.cornered) {
        const float q[8] = {b.x1, b.y1, b.x2, b.y1, b.x2, b.y2, b.x1, b.y2};
        std::copy(q, q + 8, h.quad);
      }
      float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
      for (int i = 0; i < 4; ++i) {
        const double x = h.quad[2 * i], y = h.quad[2 * i + 1];
        h.quad[2 * i] = (float)(back[0] * x + back[1] * y + back[2]);
        h.quad[2 * i + 1] = (float)(back[3] * x + back[4] * y + back[5]);
        x0 = std::min(x0, h.quad[2 * i]); x1 = std::max(x1, h.quad[2 * i]);
        y0 = std::min(y0, h.quad[2 * i + 1]); y1 = std::max(y1, h.quad[2 * i + 1]);
      }
      // Upright finds report the detector's own box, exactly as before the search existed (JSON,
      // tracker and parity all key on it); rotated finds report the bbox of the mapped-back quad.
      if (ang == 0.f) h.box = b;
      else
        h.box = {std::clamp(x0, 0.f, (float)W - 1), std::clamp(y0, 0.f, (float)H - 1),
                 std::clamp(x1, 0.f, (float)W - 1), std::clamp(y1, 0.f, (float)H - 1), b.score};
      all.push_back(h);
    }
  }
  std::sort(all.begin(), all.end(), [](const PlateHit& a, const PlateHit& b) { return a.box.score > b.box.score; });
  std::vector<PlateHit> keep;
  for (const PlateHit& h : all) {
    bool dup = false;
    for (const PlateHit& k : keep) if (box_iou(h.box, k.box) > 0.45f) { dup = true; break; }
    if (!dup) keep.push_back(h);
  }
  return keep;
}

inline Read read_plate_warped(const onx::Graph& ocr, const spec::Spec& sp, const unsigned char* rgb,
                             int W, int H, const float corners[8], const CornerCfg& cfg);

// Read one hit: rectified when the corner net placed it; otherwise the old box crop (+TTA) for an
// upright find, or the rotated box's own quad (margin 0: the detector box already has its border).
inline Read read_hit(const onx::Graph& ocr, const spec::Spec& sp, const unsigned char* rgb, int W, int H,
                     const PlateHit& h, const CornerCfg& ccfg, bool tta) {
  if (h.cornered) return read_plate_warped(ocr, sp, rgb, W, H, h.quad, ccfg);
  // Not cornered: the old box crop (+ margin TTA), taken in the frame the plate was found in, where
  // it stands nearly upright. Measured with rot-test this beats every corner-net variant, so it is
  // the reading path for rotated finds too, not just a fallback.
  const unsigned char* im = rgb;
  int RW = W, RH = H;
  std::vector<unsigned char> rot;
  if (h.angle != 0.f) {
    double back[6];
    rot = rotate_frame(rgb, W, H, h.angle, RW, RH, back);
    im = rot.data();
  }
  const Box& b = h.rbox;
  return tta ? read_plate_tta(ocr, sp, im, RW, RH, b.x1, b.y1, b.x2, b.y2)
             : read_plate_single(ocr, sp, im, RW, RH, b.x1, b.y1, b.x2, b.y2);
}

// Read a plate through the rectifier: one forward pass, no TTA — the framing is now fixed by the
// corners instead of being averaged over six guesses.
inline Read read_plate_warped(const onx::Graph& ocr, const spec::Spec& sp, const unsigned char* rgb,
                             int W, int H, const float corners[8], const CornerCfg& cfg) {
  std::vector<std::string> hn = ocr_head_names(ocr);
  std::vector<float> in = warp_to_input(rgb, W, H, corners, cfg.margin, 128);
  Tensor xt = from_data({1, 3, 128, 128}, std::move(in));
  auto vals = onx::run_onnx(ocr, xt);
  Read r;
  r.crops = 1;
  for (size_t h = 0; h < hn.size(); ++h) {
    const std::vector<float>& p = vals.at(hn[h])->data;
    int best = 0;
    double tot = 0;
    for (size_t i = 0; i < p.size(); ++i) { tot += p[i]; if (p[i] > p[best]) best = (int)i; }
    r.arg.push_back(best);
    r.conf.push_back(tot > 0 ? (float)(p[best] / tot) : 0.f);
    std::vector<float> pv(p.size());
    for (size_t i = 0; i < p.size(); ++i) pv[i] = (float)(tot > 0 ? p[i] / tot : 0.0);
    r.probs.push_back(std::move(pv));
  }
  r.plate = spec::decode(sp, r.arg);
  return r;
}

// ---- result serialisation -------------------------------------------------------------------
// One JSON shape for every consumer: the CLI (`jlpr detect --json`), the WASM module and the
// Python side (tools/infer.py). tools/parity/infer.py diffs the C++ and Python JSON, so the field
// names and the number formatting live in exactly one place.
inline std::string json_escape(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += (char)c;
  }
  return o;
}

inline std::string plates_json(const std::vector<Box>& boxes, const std::vector<Read>& reads) {
  std::string o = "{\"plates\":[";
  char num[256];
  for (size_t i = 0; i < boxes.size() && i < reads.size(); ++i) {
    const Box& b = boxes[i];
    const Read& r = reads[i];
    snprintf(num, sizeof num, "%s{\"box\":[%.1f,%.1f,%.1f,%.1f],\"det\":%.3f,\"crops\":%d,",
             i ? "," : "", b.x1, b.y1, b.x2, b.y2, b.score, r.crops);
    o += num;
    o += "\"text\":\"" + json_escape(r.plate.text) + "\",";
    o += "\"region\":\"" + json_escape(r.plate.region) + "\",";
    o += "\"cls\":\"" + json_escape(r.plate.cls) + "\",";
    o += "\"hira\":\"" + json_escape(r.plate.hira) + "\",";
    o += "\"num\":\"" + json_escape(r.plate.disp) + "\",";
    o += "\"kind\":\"" + json_escape(r.plate.kind) + "\",";
    o += "\"arg\":[";
    for (size_t k = 0; k < r.arg.size(); ++k) { snprintf(num, sizeof num, "%s%d", k ? "," : "", r.arg[k]); o += num; }
    o += "],\"conf\":[";
    for (size_t k = 0; k < r.conf.size(); ++k) { snprintf(num, sizeof num, "%s%.4f", k ? "," : "", r.conf[k]); o += num; }
    o += "]}";
  }
  o += "]}";
  return o;
}

}  // namespace jl
