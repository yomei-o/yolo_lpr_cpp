// WASM entry point — the whole pipeline in one module: the page fetches the two .onnx files and
// the label spec, hands over the bytes, then pushes RGBA frames in. Same headers as the CLI, so
// the browser runs the same graphs and the same decode as `jlpr detect`.
//
// build: sh build/emcc.sh wasm/jlpr_wasm.cpp -o wasm/jlpr.js
#include "spec.hpp"
#include "pipeline.hpp"
#include "track.hpp"
#include "overlay.hpp"
#include <emscripten/emscripten.h>
#include <string>
#include <vector>

static onx::Graph g_det, g_ocr, g_corner;
static bool g_det_ok = false, g_ocr_ok = false, g_corner_ok = false;
static spec::Spec g_spec;
static bool g_spec_ok = false;
static std::string g_result = "{}";
static std::vector<jl::Box> g_last_boxes;
static std::vector<std::vector<std::vector<float>>> g_last_probs;
static jl::Tracker g_tracker;
static jl::TrackerCfg g_tcfg;
static int g_rot_search = 0;      // 1 = also search frames rotated +-30 deg when nothing is found
static int g_refine = 0;          // extra corner-net passes (pipeline.hpp refine_corners)

// What the last jl_run / jl_track_step found, kept for jl_draw.
struct DrawItem { jl::Box box; std::string label; bool settled; };
static std::vector<DrawItem> g_draw;

static void remember_plates(const std::vector<jl::Box>& boxes, const std::vector<jl::Read>& reads) {
  g_draw.clear();
  for (size_t i = 0; i < boxes.size() && i < reads.size(); ++i)
    g_draw.push_back({boxes[i], ov::number_label(reads[i].plate), false});
}

extern "C" {

// Tilted / rotated plates (README 「回転・斜め」). Off by default so jl_run behaves as before.
EMSCRIPTEN_KEEPALIVE void jl_set_rotation(int rot_search, int refine) {
  g_rot_search = rot_search;
  g_refine = refine;
}

EMSCRIPTEN_KEEPALIVE int jl_load_det(const unsigned char* buf, int len) {
  g_det = onx::parse_onnx(buf, (size_t)len);
  g_det_ok = !g_det.nodes.empty();
  return g_det_ok ? (int)g_det.nodes.size() : -1;
}

EMSCRIPTEN_KEEPALIVE int jl_load_ocr(const unsigned char* buf, int len) {
  g_ocr = onx::parse_onnx(buf, (size_t)len);
  g_ocr_ok = !g_ocr.nodes.empty();
  return g_ocr_ok ? (int)g_ocr.nodes.size() : -1;
}

// The 4-corner regressor (M6). With it loaded, a plate is read from ONE rectified crop instead of
// six margins — the margin sweep only ever existed because the framing was unknown.
EMSCRIPTEN_KEEPALIVE int jl_load_corner(const unsigned char* buf, int len) {
  g_corner = onx::parse_onnx(buf, (size_t)len);
  g_corner_ok = !g_corner.nodes.empty();
  return g_corner_ok ? (int)g_corner.nodes.size() : -1;
}

EMSCRIPTEN_KEEPALIVE int jl_load_spec(const char* text) {
  g_spec = spec::parse(std::string(text));
  g_spec_ok = !g_spec.groups.empty();
  return g_spec_ok ? (int)g_spec.groups.size() : -1;
}

// Run on an RGBA frame (w*h*4, canvas order). tta=0 reads one crop, 1 reads the margin spread.
// box_* < 0 means "detect"; otherwise the detector is skipped and that box is read directly
// (the page's 「枠を手で指定」 path, which is how a hand-held close-up gets read at all).
// det_kind: 0 = the interim YOLOX graph
//           1 = a [1,4+nc,N] head whose boxes are xyxy (PlateYOLO-JP with the NMS tail cut off)
//           2 = the same shape but cxcywh (a plain Ultralytics export of our own yolov8n)
// Getting 1 vs 2 wrong does not crash: it produces boxes of plausible size in the wrong places,
// which is exactly the kind of failure a demo hides. Hence an explicit parameter, no guessing.
EMSCRIPTEN_KEEPALIVE int jl_run(const unsigned char* rgba, int w, int h, float conf, int tta,
                                float bx0, float by0, float bx1, float by1, int det_kind) {
  g_draw.clear();
  if (!g_ocr_ok || !g_spec_ok) { g_result = "{\"error\":\"models not loaded\"}"; return -1; }
  std::vector<unsigned char> rgb((size_t)w * h * 3);
  for (size_t i = 0, n = (size_t)w * h; i < n; ++i) {
    rgb[i * 3 + 0] = rgba[i * 4 + 0];
    rgb[i * 3 + 1] = rgba[i * 4 + 1];
    rgb[i * 3 + 2] = rgba[i * 4 + 2];
  }

  std::vector<jl::Box> boxes;
  std::vector<jl::Read> reads;
  jl::CornerCfg ccfg;
  ccfg.refine = g_refine;
  if (bx0 < 0 && g_rot_search && g_det_ok) {
    // Tilted-plate path: upright first, the rotated frames only when the upright pass found nothing
    // (an upright plate costs exactly what it did before).
    jl::DetCfg cfg;
    cfg.conf = conf;
    if (det_kind >= 1) {
      cfg.kind = jl::DetKind::V8;
      cfg.nc = 1;
      cfg.plate_class = 0;
      cfg.imgsz = 0;
      cfg.v8_fmt = (det_kind == 2) ? BoxFmt::CXCYWH : BoxFmt::XYXY;
    }
    static const std::vector<float> angles = {0.f, 30.f, -30.f};
    for (const jl::PlateHit& hh : jl::find_plates(g_det, g_corner_ok ? &g_corner : nullptr, rgb.data(), w, h,
                                                  cfg, ccfg, angles, true)) {
      boxes.push_back(hh.box);
      reads.push_back(jl::read_hit(g_ocr, g_spec, rgb.data(), w, h, hh, ccfg, tta != 0));
    }
    g_last_boxes = boxes;
    g_last_probs.clear();
    for (const jl::Read& r : reads) g_last_probs.push_back(r.probs);
    g_result = jl::plates_json(boxes, reads);
    remember_plates(boxes, reads);
    return (int)boxes.size();
  }
  if (bx0 >= 0) {
    boxes.push_back({bx0, by0, bx1, by1, 1.0f});
  } else {
    if (!g_det_ok) { g_result = "{\"error\":\"detector not loaded\"}"; return -1; }
    jl::DetCfg cfg;
    cfg.conf = conf;
    if (det_kind >= 1) {
      cfg.kind = jl::DetKind::V8;
      cfg.nc = 1;
      cfg.plate_class = 0;
      cfg.imgsz = 0;                                  // taken from the graph's declared input
      cfg.v8_fmt = (det_kind == 2) ? BoxFmt::CXCYWH : BoxFmt::XYXY;
    }
    boxes = jl::detect_plates(g_det, rgb.data(), w, h, cfg);
  }

  g_last_boxes = boxes;
  g_last_probs.clear();
  for (const jl::Box& b : boxes) {
    float corners[8];
    if (g_corner_ok && jl::predict_corners(g_corner, rgb.data(), w, h, b, ccfg, corners)) {
      reads.push_back(jl::read_plate_warped(g_ocr, g_spec, rgb.data(), w, h, corners, ccfg));
    } else {
      reads.push_back(tta ? jl::read_plate_tta(g_ocr, g_spec, rgb.data(), w, h, b.x1, b.y1, b.x2, b.y2)
                          : jl::read_plate_single(g_ocr, g_spec, rgb.data(), w, h, b.x1, b.y1, b.x2, b.y2));
    }
  }
  for (const jl::Read& r : reads) g_last_probs.push_back(r.probs);
  std::string o = jl::plates_json(boxes, reads);
  g_result = o;
  remember_plates(boxes, reads);
  return (int)boxes.size();
}

EMSCRIPTEN_KEEPALIVE const char* jl_result() { return g_result.c_str(); }

EMSCRIPTEN_KEEPALIVE void jl_track_reset() { g_tracker.reset(); }

// One live frame: detect, read, fold into the tracks. The reading of a track improves as it is seen
// for longer, because the per-head probabilities are summed over its frames (see track.hpp).
EMSCRIPTEN_KEEPALIVE int jl_track_step(const unsigned char* rgba, int w, int h, float conf, int tta,
                                      int det_kind) {
  int n = jl_run(rgba, w, h, conf, tta, -1, -1, -1, -1, det_kind);
  if (n < 0) return n;
  // jl_run already filled g_last_boxes/g_last_reads for this frame
  g_tracker.update(g_last_boxes, g_last_probs, g_spec, g_tcfg);
  g_result = jl::tracks_json(g_tracker, g_tcfg);
  g_draw.clear();
  for (const jl::Track& t : g_tracker.tracks())
    g_draw.push_back({t.box, ov::number_label(t.plate), t.stable >= g_tcfg.stable_needed});
  return (int)g_tracker.tracks().size();
}

EMSCRIPTEN_KEEPALIVE const char* jl_tracks() { return g_result.c_str(); }

// Draw the last result into an RGBA buffer (w*h*4): a frame around each plate and its 4-digit
// number above it. clear=1 first makes the buffer fully transparent — the page lays that layer over
// the live camera preview; clear=0 draws straight onto a frame. Red = read, green = tracker settled.
// Returns the number of plates drawn.
EMSCRIPTEN_KEEPALIVE int jl_draw(unsigned char* rgba, int w, int h, int clear) {
  if (clear) std::fill(rgba, rgba + (size_t)w * h * 4, (unsigned char)0);
  const ov::Canvas cv{rgba, w, h, 4};
  for (const DrawItem& d : g_draw)
    ov::plate(cv, d.box.x1, d.box.y1, d.box.x2, d.box.y2, d.label, d.settled ? ov::kGreen : ov::kRed);
  return (int)g_draw.size();
}

}  // extern "C"
