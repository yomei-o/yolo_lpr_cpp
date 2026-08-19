// WASM entry point — the whole pipeline in one module: the page fetches the two .onnx files and
// the label spec, hands over the bytes, then pushes RGBA frames in. Same headers as the CLI, so
// the browser runs the same graphs and the same decode as `jlpr detect`.
//
// build: sh build/emcc.sh wasm/jlpr_wasm.cpp -o wasm/jlpr.js
#include "spec.hpp"
#include "pipeline.hpp"
#include <emscripten/emscripten.h>
#include <string>
#include <vector>

static onx::Graph g_det, g_ocr, g_corner;
static bool g_det_ok = false, g_ocr_ok = false, g_corner_ok = false;
static spec::Spec g_spec;
static bool g_spec_ok = false;
static std::string g_result = "{}";

extern "C" {

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
  if (!g_ocr_ok || !g_spec_ok) { g_result = "{\"error\":\"models not loaded\"}"; return -1; }
  std::vector<unsigned char> rgb((size_t)w * h * 3);
  for (size_t i = 0, n = (size_t)w * h; i < n; ++i) {
    rgb[i * 3 + 0] = rgba[i * 4 + 0];
    rgb[i * 3 + 1] = rgba[i * 4 + 1];
    rgb[i * 3 + 2] = rgba[i * 4 + 2];
  }

  std::vector<jl::Box> boxes;
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

  std::vector<jl::Read> reads;
  jl::CornerCfg ccfg;
  for (const jl::Box& b : boxes) {
    float corners[8];
    if (g_corner_ok && jl::predict_corners(g_corner, rgb.data(), w, h, b, ccfg, corners)) {
      reads.push_back(jl::read_plate_warped(g_ocr, g_spec, rgb.data(), w, h, corners, ccfg));
    } else {
      reads.push_back(tta ? jl::read_plate_tta(g_ocr, g_spec, rgb.data(), w, h, b.x1, b.y1, b.x2, b.y2)
                          : jl::read_plate_single(g_ocr, g_spec, rgb.data(), w, h, b.x1, b.y1, b.x2, b.y2));
    }
  }
  std::string o = jl::plates_json(boxes, reads);
  g_result = o;
  return (int)boxes.size();
}

EMSCRIPTEN_KEEPALIVE const char* jl_result() { return g_result.c_str(); }

}  // extern "C"
