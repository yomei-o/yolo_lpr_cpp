// jlpr — the one CLI for this project (C++ side). Mirrors tools/jlpr.py subcommand for subcommand;
// whatever one can do, the other must be able to do too (see README "Python と C++ の対等性").
//
//   jlpr labels  [--spec spec/labels.txt] [--dump | --emit-header <out.hpp>] [--vectors N] [--seed S]
//   jlpr export  --ocr <ref_dir> --out models/plate_ocr.onnx      (weights.bin + manifest -> ONNX)
//   jlpr detect  --img <file> [--det models/plate_det.onnx] [--ocr models/plate_ocr.onnx] [--out out.png]
//   jlpr gen | train | val                                         (later milestones)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif
#include "spec.hpp"

static void out_raw(const std::string& s) { fwrite(s.data(), 1, s.size(), stdout); }

static std::string arg_of(int argc, char** argv, const std::string& key, const std::string& def) {
  for (int i = 2; i + 1 < argc; ++i) if (key == argv[i]) return argv[i + 1];
  return def;
}
static bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 2; i < argc; ++i) if (key == argv[i]) return true;
  return false;
}

// C string literal escaping, written without backslash literals in the source of this file
// (0x5c = backslash, 0x22 = double quote) so the generator itself stays easy to grep.
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
    // Embedded copy of the spec for builds without a filesystem (WASM). Byte-identical whether
    // emitted by C++ or Python: both write the same text.
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

int main(int argc, char** argv) {
#ifdef _WIN32
  _setmode(_fileno(stdout), _O_BINARY);      // keep dumps byte-identical with the Python side
#endif
  if (argc < 2) {
    printf("jlpr — Japanese license plate pipeline (C++)\n"
           "  jlpr labels [--dump|--emit-header out.hpp]\n"
           "  jlpr export --ocr <ref_dir> --out <onnx>\n"
           "  jlpr detect --img <file> [--det <onnx>] [--ocr <onnx>] [--out out.png]\n"
           "  jlpr gen|train|val   (not implemented yet)\n");
    return 1;
  }
  std::string cmd = argv[1];
  if (cmd == "labels") return cmd_labels(argc, argv);
  printf("jlpr: '%s' is not implemented yet\n", cmd.c_str());
  return 1;
}
