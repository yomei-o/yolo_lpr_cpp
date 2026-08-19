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
#include "crop.hpp"
#include "spec.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace jl {

// ---- detector -------------------------------------------------------------------------------
// The interim detector is lpr_cpp's ReLU plate yolox-tiny (8 classes, class 7 = plate) at 416.
// It is ONNX-fed, so its head already carries probabilities: logits=false. A yolov8 nc=1
// detector replaces it at M6 (see RESUME) — the box interface here does not change.
struct DetCfg {
  int imgsz = 416;
  int plate_class = 7;
  float conf = 0.15f;                 // low on purpose: black/yellow plates score ~1/4 of white
  float nms = 0.45f;
  bool logits = false;
  std::vector<std::string> heads = {"/head/Concat_output_0", "/head/Concat_1_output_0",
                                    "/head/Concat_2_output_0"};
  std::vector<int64_t> strides = {8, 16, 32};
  int64_t nc = 8;
};

struct Box { float x1, y1, x2, y2, score; };

inline std::vector<Box> detect_plates(const onx::Graph& det, const unsigned char* rgb, int W, int H,
                                      const DetCfg& cfg, std::vector<Det>* all_out = nullptr) {
  float scale = 1.f;
  Tensor x = letterbox_bgr(rgb, W, H, cfg.imgsz, scale);
  std::set<std::string> stop(cfg.heads.begin(), cfg.heads.end());
  auto vals = onx::run_onnx(det, x, stop);
  std::vector<Tensor> raw;
  for (auto& h : cfg.heads) raw.push_back(vals.at(h));
  std::vector<Det> dets = yolox_detect(raw, cfg.strides, cfg.nc, cfg.conf, cfg.nms, cfg.logits);
  if (all_out) *all_out = dets;
  std::vector<Box> out;
  for (const Det& d : dets) {
    if (d.cls != cfg.plate_class) continue;
    Box b{d.x1 / scale, d.y1 / scale, d.x2 / scale, d.y2 / scale, d.score};
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
