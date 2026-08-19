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
};

// Predict the plate's 4 corners (image pixels, TL/TR/BR/BL) from a detector box.
inline bool predict_corners(const onx::Graph& corner, const unsigned char* rgb, int W, int H,
                           const Box& b, const CornerCfg& cfg, float out[8]) {
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
  return true;
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
