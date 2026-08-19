// jlpr — the one CLI for this project (C++ side). Mirrors tools/jlpr.py subcommand for subcommand;
// whatever one can do, the other must be able to do too (see README "Python と C++ の対等性").
//
//   jlpr labels     [--spec spec/labels.txt] [--dump | --emit-header <out.hpp>] [--vectors N] [--seed S]
//   jlpr export     --ocr <ref_dir> --out models/plate_ocr.onnx    (weights.bin + manifest -> ONNX)
//   jlpr parity-ocr --ocr <onnx> --ref <ref_dir>                   (vs the original ONNX fixture)
//   jlpr detect     --img <file> [--det <onnx>] [--ocr <onnx>] [--out out.png] [--conf f] [--single]
//   jlpr gen | train | val                                          (later milestones)
//
// build: sh build/gcc.sh pure/jlpr.cpp -o jlpr.exe   |   sh build/cc.sh pure/jlpr.cpp -o jlpr.exe
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "spec.hpp"
#include "onnx_export_lpr.hpp"
#include "pipeline.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

static void out_raw(const std::string& s) { fwrite(s.data(), 1, s.size(), stdout); }

static std::string arg_of(int argc, char** argv, const std::string& key, const std::string& def) {
  for (int i = 2; i + 1 < argc; ++i) if (key == argv[i]) return argv[i + 1];
  return def;
}
static bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 2; i < argc; ++i) if (key == argv[i]) return true;
  return false;
}

// C string literal escaping, written without backslash literals (0x5c = backslash, 0x22 = quote)
// so the generator itself stays easy to grep.
static const char BS = (char)0x5c;
static const char DQ = (char)0x22;

static std::string escape_line(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    if (c == (unsigned char)DQ || c == (unsigned char)BS) { o += BS; o += (char)c; }
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "%cx%02x", BS, c); o += b; }
    else o += (char)c;
  }
  return o;
}

static int cmd_labels(int argc, char** argv) {
  std::string spec_path = arg_of(argc, argv, "--spec", "spec/labels.txt");
  spec::Spec sp = spec::load(spec_path);
  int nvec = std::atoi(arg_of(argc, argv, "--vectors", "8").c_str());
  uint64_t seed = strtoull(arg_of(argc, argv, "--seed", "12345").c_str(), nullptr, 10);

  std::string hdr = arg_of(argc, argv, "--emit-header", "");
  if (!hdr.empty()) {
    std::ifstream f(spec_path, std::ios::binary);
    std::ostringstream ss; ss << f.rdbuf();
    std::string text = ss.str();
    std::ostringstream o;
    o << "// GENERATED from " << spec_path << " -- do not edit."
      << " (jlpr labels --emit-header | python tools/jlpr.py labels --emit-header)\n"
      << "#pragma once\n"
      << "inline const char* SPEC_LABELS_TXT =\n";
    size_t i = 0;
    while (i < text.size()) {
      size_t e = text.find('\n', i);
      if (e == std::string::npos) e = text.size();
      std::string line = text.substr(i, e - i);
      if (!line.empty() && line.back() == '\r') line.pop_back();
      o << "  " << DQ << escape_line(line) << BS << "n" << DQ << "\n";
      i = e + 1;
    }
    o << "  ;\n";
    std::string s = o.str();
    std::ofstream of(hdr, std::ios::binary);
    of.write(s.data(), s.size());
    printf("wrote %s (%zu bytes)\n", hdr.c_str(), s.size());
    return 0;
  }

  if (has_flag(argc, argv, "--dump") || argc == 2) {
    out_raw(spec::canonical_dump(sp));
    out_raw(spec::decode_vectors_dump(sp, seed, nvec));
    return 0;
  }
  printf("usage: jlpr labels [--spec f] [--dump] [--emit-header out.hpp] [--vectors N] [--seed S]\n");
  return 1;
}

static int cmd_export(int argc, char** argv) {
  std::string ref = arg_of(argc, argv, "--ocr", "");
  std::string out = arg_of(argc, argv, "--out", "models/plate_ocr.onnx");
  if (ref.empty()) {
    printf("usage: jlpr export --ocr <dir with manifest.txt + weights.bin> --out <onnx>\n");
    return 1;
  }
  lprx::export_lpr_onnx(ref, out);
  return 0;
}

// Parity against the fixture that came from the ORIGINAL ONNX (input.bin -> refout.bin, produced
// by onnxruntime). This is what proves the exported graph is the same function, not just loadable.
static int cmd_parity_ocr(int argc, char** argv) {
  std::string onnx = arg_of(argc, argv, "--ocr", "models/plate_ocr.onnx");
  std::string ref = arg_of(argc, argv, "--ref", "");
  if (ref.empty()) { printf("usage: jlpr parity-ocr --ocr <onnx> --ref <dir with input.bin refout.bin>\n"); return 1; }
  if (ref.back() != '/' && ref.back() != '\\') ref += '/';

  onx::Graph g = onx::load_onnx(onnx);
  std::vector<std::string> hn = jl::ocr_head_names(g);
  std::ifstream fi(ref + "input.bin", std::ios::binary);
  if (!fi) { printf("missing %sinput.bin\n", ref.c_str()); return 1; }
  std::vector<float> xin((size_t)3 * 128 * 128);
  fi.read((char*)xin.data(), (std::streamsize)xin.size() * 4);
  std::ifstream fo(ref + "refout.bin", std::ios::binary);
  if (!fo) { printf("missing %srefout.bin\n", ref.c_str()); return 1; }

  auto vals = onx::run_onnx(g, from_data({1, 3, 128, 128}, xin));
  float worst = 0; int argmatch = 0;
  for (size_t h = 0; h < hn.size(); ++h) {
    const std::vector<float>& got = vals.at(hn[h])->data;
    std::vector<float> want(got.size());
    fo.read((char*)want.data(), (std::streamsize)want.size() * 4);
    float w = 0; int ag = 0, aw = 0; float mg = -1e30f, mw = -1e30f;
    for (size_t i = 0; i < got.size(); ++i) {
      w = std::max(w, std::fabs(got[i] - want[i]));
      if (got[i] > mg) { mg = got[i]; ag = (int)i; }
      if (want[i] > mw) { mw = want[i]; aw = (int)i; }
    }
    worst = std::max(worst, w);
    argmatch += (ag == aw);
    printf("  %-22s dim %3zu  worst %.2e  argmax onnx=%d ref=%d%s\n", hn[h].c_str(), got.size(), w,
           ag, aw, ag == aw ? "" : "   <-- MISMATCH");
  }
  printf("exported ONNX vs original-ONNX fixture: worst %.3e   argmax %d/%zu   %s\n",
         worst, argmatch, hn.size(), (worst < 1e-3f && argmatch == (int)hn.size()) ? "MATCH" : "MISMATCH");
  return (worst < 1e-3f && argmatch == (int)hn.size()) ? 0 : 1;
}

static void draw_box(unsigned char* im, int W, int H, int x1, int y1, int x2, int y2,
                     unsigned char r, unsigned char g, unsigned char b, int t = 3) {
  auto put = [&](int a, int c) {
    if (a < 0 || c < 0 || a >= W || c >= H) return;
    unsigned char* q = &im[((size_t)c * W + a) * 3];
    q[0] = r; q[1] = g; q[2] = b;
  };
  for (int k = 0; k < t; ++k) {
    for (int a = x1; a <= x2; ++a) { put(a, y1 + k); put(a, y2 - k); }
    for (int c = y1; c <= y2; ++c) { put(x1 + k, c); put(x2 - k, c); }
  }
}

// Dump a decoded frame as raw RGBA (little-endian int32 w, h, then w*h*4 bytes). The node and
// browser tests need the *same pixels* the CLI saw, and they have no image decoder.
static int cmd_rgba(int argc, char** argv) {
  std::string img = arg_of(argc, argv, "--img", "");
  std::string out = arg_of(argc, argv, "--out", "");
  if (img.empty() || out.empty()) { printf("usage: jlpr rgba --img <file> --out <file.rgba>\n"); return 1; }
  int W = 0, H = 0, C = 0;
  unsigned char* im = stbi_load(img.c_str(), &W, &H, &C, 4);
  if (!im) { printf("cannot load %s\n", img.c_str()); return 1; }
  std::ofstream f(out, std::ios::binary);
  int32_t hdr[2] = {W, H};
  f.write((const char*)hdr, 8);
  f.write((const char*)im, (std::streamsize)W * H * 4);
  printf("wrote %s (%dx%d RGBA)\n", out.c_str(), W, H);
  stbi_image_free(im);
  return 0;
}

static int cmd_detect(int argc, char** argv) {
  std::string img = arg_of(argc, argv, "--img", "");
  std::string det_p = arg_of(argc, argv, "--det", "models/plate_det.onnx");
  std::string ocr_p = arg_of(argc, argv, "--ocr", "models/plate_ocr.onnx");
  std::string spec_p = arg_of(argc, argv, "--spec", "spec/labels.txt");
  std::string outp = arg_of(argc, argv, "--out", "");
  bool single = has_flag(argc, argv, "--single");
  bool as_json = has_flag(argc, argv, "--json");   // machine-readable, same shape as the WASM/Python output
  jl::DetCfg cfg;
  cfg.conf = (float)atof(arg_of(argc, argv, "--conf", "0.15").c_str());
  cfg.imgsz = std::atoi(arg_of(argc, argv, "--imgsz", "416").c_str());
  if (img.empty()) {
    printf("usage: jlpr detect --img <file> [--det onnx] [--ocr onnx] [--out png] [--conf f] [--single] [--json]\n");
    return 1;
  }

  int W = 0, H = 0, C = 0;
  unsigned char* im = stbi_load(img.c_str(), &W, &H, &C, 3);
  if (!im) { printf("cannot load %s\n", img.c_str()); return 1; }
  spec::Spec sp = spec::load(spec_p);
  onx::Graph det = onx::load_onnx(det_p);
  onx::Graph ocr = onx::load_onnx(ocr_p);
  if (!as_json) printf("%s %dx%d   det=%s ocr=%s\n", img.c_str(), W, H, det_p.c_str(), ocr_p.c_str());

  std::vector<Det> all;
  std::vector<jl::Box> boxes = jl::detect_plates(det, im, W, H, cfg, &all);
  std::vector<jl::Read> reads;
  if (!as_json)
    printf("plates: %zu (conf>=%.2f, %zu raw detections over all classes)\n", boxes.size(), cfg.conf, all.size());

  for (size_t i = 0; i < boxes.size(); ++i) {
    const jl::Box& b = boxes[i];
    jl::Read r = single ? jl::read_plate_single(ocr, sp, im, W, H, b.x1, b.y1, b.x2, b.y2)
                        : jl::read_plate_tta(ocr, sp, im, W, H, b.x1, b.y1, b.x2, b.y2);
    reads.push_back(r);
    if (!as_json) {
      printf("  [%zu] box (%.0f,%.0f)-(%.0f,%.0f) det %.2f  crops %d\n", i, b.x1, b.y1, b.x2, b.y2,
             b.score, r.crops);
      out_raw("       " + r.plate.text + "\n");
      printf("       conf");
      for (size_t h = 0; h < r.conf.size(); ++h) printf(" %.2f", r.conf[h]);
      printf("\n");
    }
    draw_box(im, W, H, (int)b.x1, (int)b.y1, (int)b.x2, (int)b.y2, 255, 60, 60);
  }
  if (as_json) out_raw(jl::plates_json(boxes, reads) + "\n");
  if (!outp.empty()) {
    stbi_write_png(outp.c_str(), W, H, 3, im, W * 3);
    if (!as_json) printf("wrote %s\n", outp.c_str());
  }
  stbi_image_free(im);
  return boxes.empty() ? 1 : 0;
}

int main(int argc, char** argv) {
#ifdef _WIN32
  _setmode(_fileno(stdout), _O_BINARY);      // keep dumps byte-identical with the Python side
#endif
  if (argc < 2) {
    printf("jlpr — Japanese license plate pipeline (C++)\n"
           "  jlpr labels     [--dump|--emit-header out.hpp]\n"
           "  jlpr export     --ocr <ref_dir> --out <onnx>\n"
           "  jlpr parity-ocr --ocr <onnx> --ref <ref_dir>\n"
           "  jlpr detect     --img <file> [--det onnx] [--ocr onnx] [--out png] [--conf f] [--single] [--json]\n"
           "  jlpr rgba       --img <file> --out <file.rgba>\n"
           "  jlpr gen|train|val   (not implemented yet)\n");
    return 1;
  }
  std::string cmd = argv[1];
  if (cmd == "labels") return cmd_labels(argc, argv);
  if (cmd == "export") return cmd_export(argc, argv);
  if (cmd == "parity-ocr") return cmd_parity_ocr(argc, argv);
  if (cmd == "detect") return cmd_detect(argc, argv);
  if (cmd == "rgba") return cmd_rgba(argc, argv);
  printf("jlpr: '%s' is not implemented yet\n", cmd.c_str());
  return 1;
}
