// jlpr — the one CLI for this project (C++ side). Mirrors tools/jlpr.py subcommand for subcommand;
// whatever one can do, the other must be able to do too (see README "Python と C++ の対等性").
//
//   jlpr labels     [--spec spec/labels.txt] [--dump | --emit-header <out.hpp>] [--vectors N] [--seed S]
//   jlpr export     --ocr <ref_dir> --out models/plate_ocr.onnx    (weights.bin + manifest -> ONNX)
//   jlpr parity-ocr --ocr <onnx> --ref <ref_dir>                   (vs the original ONNX fixture)
//   jlpr detect     --img <file> [--det <onnx>] [--ocr <onnx>] [--out out.png] [--conf f] [--single]
//   jlpr gen | gen-det                                             (synthetic crops / frames)
//   jlpr train      --model ocr|det|corner ...                     (all three stages train here)
//   jlpr val        [--model det] ...                              (recognizer / detector metrics)
//
// build: sh build/gcc.sh pure/jlpr.cpp -o jlpr.exe   |   sh build/cc.sh pure/jlpr.cpp -o jlpr.exe
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "spec.hpp"
#include "onnx_export_lpr.hpp"
#include "pipeline.hpp"
#include "gen_render.hpp"
#include "gen_det.hpp"
#include "onnx_train.hpp"
#include "train_ocr.hpp"
#include "train_det.hpp"
#include "eval_det.hpp"
#include "train_corner.hpp"
#include <filesystem>
#include <functional>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
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

// mkdir -p: create every missing component. Creating only the leaf silently fails the first time a
// generator is pointed at data/synth on a fresh checkout, and the error looks like a permissions
// problem instead of a missing parent.
static void make_dir(const std::string& d) {
  std::string acc;
  for (size_t i = 0; i <= d.size(); ++i) {
    if (i == d.size() || d[i] == '/' || d[i] == (char)0x5c) {   // 0x5c = backslash
      if (!acc.empty() && acc != "." && acc != "..") {
#ifdef _WIN32
        _mkdir(acc.c_str());
#else
        mkdir(acc.c_str(), 0755);
#endif
      }
    }
    if (i < d.size()) acc += d[i];
  }
}

static int cmd_gen(int argc, char** argv) {
  std::string out = arg_of(argc, argv, "--out", "data/synth");
  std::string spec_path = arg_of(argc, argv, "--spec", "spec/labels.txt");
  std::string font_dir = arg_of(argc, argv, "--fonts", "fonts");
  int count = std::atoi(arg_of(argc, argv, "--count", "16").c_str());
  int start = std::atoi(arg_of(argc, argv, "--start", "0").c_str());
  int out_px = std::atoi(arg_of(argc, argv, "--out-px", "192").c_str());
  uint64_t seed = strtoull(arg_of(argc, argv, "--seed", "12345").c_str(), nullptr, 10);
  bool meta_only = has_flag(argc, argv, "--meta-only");
  bool clean = has_flag(argc, argv, "--clean");    // no degradation (debug / curriculum)
  // --region <n> pins the region name, `sweep` cycles all of them, `a-b` cycles an inclusive range
  // (133-137 = the 2025 additions, for oversampling names the real data has none of)
  std::string region_arg = arg_of(argc, argv, "--region", "");
  int region_pin = region_arg.empty() ? -1 : (region_arg == "sweep" ? -2 : std::atoi(region_arg.c_str()));
  int region_lo = -1, region_hi = -1;
  if (region_arg.find('-') != std::string::npos && region_arg != "sweep") {
    size_t d = region_arg.find('-');
    region_lo = std::atoi(region_arg.substr(0, d).c_str());
    region_hi = std::atoi(region_arg.substr(d + 1).c_str());
    region_pin = -3;
  }
  bool tex_dump = has_flag(argc, argv, "--tex-dump");   // also write the flat plate art (debug)
  bool quiet = has_flag(argc, argv, "--quiet");

  spec::Spec sp = spec::load(spec_path);
  // The font list is drawn from even in --meta-only mode, so the rng stream (and therefore the
  // meta dump) is identical whether or not images are rendered.
  std::string only_font = arg_of(argc, argv, "--font", "");
  // --fonts-strict: only the faces spec/fonts.txt lists (reproducible datasets on any machine)
  std::string fmanifest;
  if (has_flag(argc, argv, "--fonts-strict")) {
    size_t sl = spec_path.find_last_of("/\\");
    fmanifest = (sl == std::string::npos ? std::string() : spec_path.substr(0, sl + 1)) + "fonts.txt";
  }
  std::vector<std::string> font_paths = gen::font_files(font_dir, only_font, fmanifest);
  std::vector<std::string> font_names;
  for (const std::string& f : font_paths) font_names.push_back(f.substr(f.find_last_of("/\\") + 1));
  if (!quiet) {
    std::string d = gen::font_manifest_diff(spec_path, font_names);
    if (!d.empty()) printf("%s%c", d.c_str(), 0x0a);
  }
  if (font_paths.empty()) {
    printf("no fonts in %s — run: python tools/fetch_fonts.py --include-system\n", font_dir.c_str());
    return 1;
  }
  std::vector<gen::Font> fonts;
  if (!meta_only) {
    for (const std::string& f : font_paths) {
      gen::Font fo;
      if (gen::load_font(f, fo)) fonts.push_back(std::move(fo));
      else printf("warn: cannot load font %s\n", f.c_str());
    }
    if (fonts.empty()) {
      printf("no fonts in %s — run: python tools/fetch_fonts.py --include-system\n", font_dir.c_str());
      return 1;
    }
  }

  make_dir(out);
  std::string mode = start > 0 ? "ab" : "wb";
  FILE* fl = fopen((out + "/labels.txt").c_str(), mode.c_str());
  FILE* fc = fopen((out + "/corners.txt").c_str(), mode.c_str());
  FILE* fm = fopen((out + "/meta.txt").c_str(), mode.c_str());
  if (!fl || !fc || !fm) { printf("cannot write into %s\n", out.c_str()); return 1; }

  // Samples are independent (each has its own rng stream), so this is embarrassingly parallel.
  // Text output is collected per index and written in order afterwards, so the files are identical
  // whether the build has OpenMP or not.
  std::vector<std::string> ml_v((size_t)count), ll_v((size_t)count), cl_v((size_t)count);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int i = start; i < start + count; ++i) {
    // Per-sample stream so any range can be generated independently (spec/gen.md).
    Rng rng(seed ^ ((uint64_t)i * 0x9E3779B97F4A7C15ull));
    gen::Params p = gen::sample(rng, sp);
    // --region pins (or sweeps) the region name. Applied after sample() so the draw order stays as
    // spec/gen.md documents it; used to make crops of names the real data has none of.
    if (region_pin == -2) p.region = i % sp.head("region").n;
    else if (region_pin == -3) p.region = region_lo + i % (region_hi - region_lo + 1);
    else if (region_pin >= 0) p.region = region_pin;
    if (clean) gen::make_clean(p);
    int font_idx = (int)rng.below((uint64_t)font_names.size());   // draw #29 (spec/gen.md)
    char name[64];
    snprintf(name, sizeof name, "plate%06d.png", i);

    ml_v[(size_t)(i - start)] = gen::meta_line(name, p, sp, font_names[font_idx]);
    ll_v[(size_t)(i - start)] = gen::labels_line(name, p);

    if (meta_only) continue;

    if (tex_dump) {
      Rng trng(seed ^ ((uint64_t)i * 0x9E3779B97F4A7C15ull));   // same stream, texture only
      gen::Params tp = gen::sample(trng, sp);
      (void)trng.below((uint64_t)font_names.size());
      gen::Img tex = gen::plate_texture(tp, sp, fonts, font_idx, trng);
      char tn[64];
      snprintf(tn, sizeof tn, "tex%06d.png", i);
      stbi_write_png((out + "/" + tn).c_str(), tex.w, tex.h, 3, tex.d.data(), tex.w * 3);
    }
    gen::Rendered R = gen::render(p, sp, fonts, font_idx, out_px, rng);
    stbi_write_png((out + "/" + name).c_str(), R.crop.w, R.crop.h, 3, R.crop.d.data(), R.crop.w * 3);
    char cl[256];
    snprintf(cl, sizeof cl, "%s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %s\n", name,
             R.corners[0], R.corners[1], R.corners[2], R.corners[3], R.corners[4], R.corners[5],
             R.corners[6], R.corners[7], R.font.c_str());
    cl_v[(size_t)(i - start)] = cl;
    if (!quiet && (i - start) < 12) {
      std::string t = gen::plate_text(p, sp);
      printf("  %s  %-28s px=%.0f yaw=%.0f pitch=%.0f blur=%.2f legible=%d font=%s\n", name,
             t.c_str(), p.plate_px, p.yaw, p.pitch, p.blur, p.legible ? 1 : 0, R.font.c_str());
    }
  }
  for (int k = 0; k < count; ++k) {                 // write in index order, threads or not
    fwrite(ml_v[(size_t)k].data(), 1, ml_v[(size_t)k].size(), fm);
    fwrite(ll_v[(size_t)k].data(), 1, ll_v[(size_t)k].size(), fl);
    if (!cl_v[(size_t)k].empty()) fwrite(cl_v[(size_t)k].data(), 1, cl_v[(size_t)k].size(), fc);
  }
  fclose(fl); fclose(fc); fclose(fm);
  printf("%s %d samples into %s (out_px=%d, seed=%llu)\n", meta_only ? "meta for" : "wrote", count,
         out.c_str(), out_px, (unsigned long long)seed);
  return 0;
}


// jlpr gen-det — detection training data (full frames, plates at 3%..95% of the frame width).
static int cmd_gen_det(int argc, char** argv) {
  std::string out = arg_of(argc, argv, "--out", "data/det");
  std::string spec_path = arg_of(argc, argv, "--spec", "spec/labels.txt");
  std::string font_dir = arg_of(argc, argv, "--fonts", "fonts");
  std::string bg_dir = arg_of(argc, argv, "--bg", "");
  std::string real_dir = arg_of(argc, argv, "--real-plates", "");
  int real_pct = std::atoi(arg_of(argc, argv, "--real-pct", "50").c_str());
  std::string only_font = arg_of(argc, argv, "--font", "");
  int count = std::atoi(arg_of(argc, argv, "--count", "16").c_str());
  int start = std::atoi(arg_of(argc, argv, "--start", "0").c_str());
  int imgsz = std::atoi(arg_of(argc, argv, "--imgsz", "640").c_str());
  uint64_t seed = strtoull(arg_of(argc, argv, "--seed", "12345").c_str(), nullptr, 10);
  bool meta_only = has_flag(argc, argv, "--meta-only");
  bool quiet = has_flag(argc, argv, "--quiet");

  spec::Spec sp = spec::load(spec_path);
  // --fonts-strict: only the faces spec/fonts.txt lists (reproducible datasets on any machine)
  std::string fmanifest;
  if (has_flag(argc, argv, "--fonts-strict")) {
    size_t sl = spec_path.find_last_of("/\\");
    fmanifest = (sl == std::string::npos ? std::string() : spec_path.substr(0, sl + 1)) + "fonts.txt";
  }
  std::vector<std::string> font_paths = gen::font_files(font_dir, only_font, fmanifest);
  if (font_paths.empty()) { printf("no fonts in %s\n", font_dir.c_str()); return 1; }

  // Background pool: real photos make the negatives and the context far more honest than a gradient.
  // Sorted so the index draw means the same file in both languages.
  std::vector<std::string> bgs;
  if (!bg_dir.empty()) {
    trn::list_images_recursive(bg_dir, bgs);
    std::sort(bgs.begin(), bgs.end());
    if (!quiet) printf("background pool: %zu files from %s\n", bgs.size(), bg_dir.c_str());
  }

  // Real plate photos to paste (searched recursively): the detector needs the actual typeface,
  // lighting and dirt, not our drawing of them. Trained on drawn plates only it reached mAP50 0.99
  // on synthetic frames and still scored 0.1 on the real bus photo. Sorted, so the index draw picks
  // the same file in both languages.
  std::vector<std::string> reals;
  if (!real_dir.empty()) {
    // comma-separated list of directories, so "the plate folders but not the negatives" is expressible
    size_t pos = 0;
    while (pos <= real_dir.size()) {
      size_t comma = real_dir.find(',', pos);
      std::string one = real_dir.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      if (!one.empty()) trn::list_images_recursive(one, reals);
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    std::sort(reals.begin(), reals.end());
    if (!quiet) printf("real plate pool: %zu files from %s (%d%% of plates)\n", reals.size(),
                       real_dir.c_str(), real_pct);
  }

  std::vector<gen::Font> fonts;
  if (!meta_only) {
    for (const std::string& f : font_paths) {
      gen::Font fo;
      if (gen::load_font(f, fo)) fonts.push_back(std::move(fo));
    }
    if (fonts.empty()) { printf("no usable font in %s\n", font_dir.c_str()); return 1; }
  }

  make_dir(out);
  make_dir(out + "/images");
  make_dir(out + "/labels");
  std::string mode = start > 0 ? "ab" : "wb";
  FILE* fm = fopen((out + "/meta.txt").c_str(), mode.c_str());
  FILE* fc = fopen((out + "/corners.txt").c_str(), mode.c_str());
  if (!fm || !fc) { printf("cannot write into %s\n", out.c_str()); return 1; }

  std::vector<std::string> mv((size_t)count), cv((size_t)count);
  int kept_total = 0, neg_total = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+ : kept_total, neg_total)
#endif
  for (int i = start; i < start + count; ++i) {
    Rng rng(seed ^ ((uint64_t)i * 0x9E3779B97F4A7C15ull) ^ 0xD1B54A32D192ED03ull);
    gen::DetSample d = gen::sample_det(rng, sp, (int)bgs.size(), (int)font_paths.size(),
                                       (int)reals.size(), real_pct);
    gen::apply_share(d, imgsz);
    gen::place_det(d, imgsz);
    char name[64];
    snprintf(name, sizeof name, "det%06d.png", i);
    std::string bg_name = "synth";
    if (d.bg_real && !bgs.empty()) {
      std::string b = bgs[(size_t)d.bg_idx];
      bg_name = b.substr(b.find_last_of("/\\") + 1);
    }
    mv[(size_t)(i - start)] = gen::det_meta(name, d, sp, bg_name, imgsz);
    cv[(size_t)(i - start)] = gen::det_corners(name, d, imgsz);
    for (const gen::DetPlate& pl : d.plates) kept_total += pl.keep ? 1 : 0;
    neg_total += (d.n_plates == 0) ? 1 : 0;

    if (meta_only) continue;

    gen::Img canvas(imgsz, imgsz, {0, 0, 0});
    bool have_bg = false;
    if (d.bg_real && !bgs.empty()) {
      int bw = 0, bh = 0, bc = 0;
      std::vector<unsigned char> blob = trn::read_file(bgs[(size_t)d.bg_idx]);
      unsigned char* bi = blob.empty() ? nullptr
          : stbi_load_from_memory(blob.data(), (int)blob.size(), &bw, &bh, &bc, 3);
      if (bi) {                                   // centre-crop to square, then scale to imgsz
        int side = std::min(bw, bh);
        int ox = (bw - side) / 2, oy = (bh - side) / 2;
        for (int y = 0; y < imgsz; ++y)
          for (int x = 0; x < imgsz; ++x) {
            int sx = ox + (int)((x + 0.5f) * side / imgsz);
            int sy = oy + (int)((y + 0.5f) * side / imgsz);
            sx = std::clamp(sx, 0, bw - 1); sy = std::clamp(sy, 0, bh - 1);
            const unsigned char* q = &bi[((size_t)sy * bw + sx) * 3];
            unsigned char* p2 = canvas.at(x, y);
            p2[0] = q[0]; p2[1] = q[1]; p2[2] = q[2];
          }
        stbi_image_free(bi);
        have_bg = true;
      }
    }
    if (!have_bg) gen::synth_background(canvas, d.bg_hue, d.bg_dark, rng);

    for (gen::DetPlate& pl : d.plates) {
      gen::Img tex;
      bool got_real = false;
      if (pl.use_real && !reals.empty()) {
        int rw = 0, rh = 0, rc = 0;
        std::vector<unsigned char> rblob = trn::read_file(reals[(size_t)pl.real_idx % reals.size()]);
        unsigned char* ri = rblob.empty() ? nullptr
            : stbi_load_from_memory(rblob.data(), (int)rblob.size(), &rw, &rh, &rc, 3);
        if (ri) {
          tex.w = rw; tex.h = rh; tex.d.assign(ri, ri + (size_t)rw * rh * 3);
          stbi_image_free(ri);
          got_real = true;
        }
      }
      if (!got_real) tex = gen::plate_texture(pl.p, sp, fonts, pl.font_idx % (int)fonts.size(), rng);
      gen::paste_textured_quad(canvas, tex, pl.quad);
    }

    gen::Params ph;                               // image-level degradation
    ph.brightness = d.brightness; ph.contrast = d.contrast; ph.warm = d.warm; ph.noise = d.noise;
    gen::photometric(canvas, ph, rng);
    gen::gaussian_blur(canvas, (float)d.blur);
    gen::motion_blur(canvas, (float)d.motion);

    stbi_write_png((out + "/images/" + name).c_str(), canvas.w, canvas.h, 3, canvas.d.data(), canvas.w * 3);
    std::string lab = gen::det_labels(d, imgsz);
    std::string lp = out + "/labels/" + std::string(name).substr(0, strlen(name) - 4) + ".txt";
    FILE* lf = fopen(lp.c_str(), "wb");
    if (lf) { fwrite(lab.data(), 1, lab.size(), lf); fclose(lf); }
  }
  for (int k = 0; k < count; ++k) {
    fwrite(mv[(size_t)k].data(), 1, mv[(size_t)k].size(), fm);
    if (!cv[(size_t)k].empty()) fwrite(cv[(size_t)k].data(), 1, cv[(size_t)k].size(), fc);
  }
  fclose(fm); fclose(fc);
  printf("%s %d frames into %s (imgsz=%d, %d plates kept, %d empty frames, seed=%llu)\n",
         meta_only ? "meta for" : "wrote", count, out.c_str(), imgsz, kept_total, neg_total,
         (unsigned long long)seed);
  return 0;
}


// Detector evaluation over a YOLO-format directory: the product path (full graph, decode tail, NMS)
// scored with pure/eval_det.hpp. Shared by `jlpr val --model det` and by the in-training validation,
// so "the number the trainer chased" and "the number the CLI reports" cannot drift apart.
struct DetEval {
  evd::Report r;
  std::vector<int> tp_b, gt_b;
  int fp_bucket = 0, n_det_at_conf = 0, empty_frames = 0, fp_empty = 0, n_gt = 0;
  size_t frames = 0;
};

static DetEval eval_det_dir(const onx::Graph& det, jl::DetCfg cfg, const std::string& data, int limit,
                            float conf, float iou_thr, bool progress) {
  DetEval e;
  e.tp_b.assign(evd::buckets().size(), 0);
  e.gt_b.assign(evd::buckets().size(), 0);
  std::vector<std::string> files;
  trn::list_images_recursive(data + "/images", files);
  std::sort(files.begin(), files.end());
  if (limit > 0 && (int)files.size() > limit) files.resize((size_t)limit);
  e.frames = files.size();
  const std::vector<float> thr = evd::iou_thresholds();
  std::vector<evd::Scored> all;
  for (size_t k = 0; k < files.size(); ++k) {
    int W = 0, H = 0, C = 0;
    std::vector<unsigned char> blob = trn::read_file(files[k]);
    unsigned char* im = blob.empty() ? nullptr
                                     : stbi_load_from_memory(blob.data(), (int)blob.size(), &W, &H, &C, 3);
    if (!im) { printf("cannot read %s\n", files[k].c_str()); continue; }
    std::string stem = files[k].substr(files[k].find_last_of("/\\") + 1);
    stem = stem.substr(0, stem.find_last_of('.'));
    std::vector<evd::GtBox> gts;
    {
      std::vector<unsigned char> lb = trn::read_file(data + "/labels/" + stem + ".txt");
      std::istringstream ss(std::string(lb.begin(), lb.end()));
      std::string line;
      while (std::getline(ss, line)) {
        std::istringstream ls(line);
        int c; float xc, yc, w, h;
        if (!(ls >> c >> xc >> yc >> w >> h)) continue;
        if (w <= 0 || h <= 0) continue;
        gts.push_back({(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H, w});
      }
    }
    std::vector<jl::Box> boxes = jl::detect_plates(det, im, W, H, cfg);
    stbi_image_free(im);
    std::vector<evd::DetBox> dets, strong;
    for (const jl::Box& b : boxes) {
      dets.push_back({b.x1, b.y1, b.x2, b.y2, b.score});
      if (b.score >= conf) strong.push_back(dets.back());
    }
    e.n_gt += (int)gts.size();
    e.n_det_at_conf += (int)strong.size();
    if (gts.empty()) { ++e.empty_frames; e.fp_empty += (int)strong.size(); }
    evd::match_frame(dets, gts, thr, all);                            // confidence-ranked, for mAP
    evd::bucket_match(strong, gts, iou_thr, e.tp_b, e.gt_b, e.fp_bucket);   // per-gt best IoU
    if (progress && (k + 1) % 100 == 0) { printf("  %zu/%zu\n", k + 1, files.size()); fflush(stdout); }
  }
  e.r = evd::summarize(all, e.n_gt, conf);
  return e;
}

// jlpr train --model det --gradcheck — the analytic gradient of the fused v8 loss against a central
// difference, on small random heads. Cheap, needs no model and no data, and it is the only thing
// standing between "the loss looks plausible" and "the loss is differentiated correctly": the
// assignment is discrete, so a wrong sign in the CIoU or DFL chain still trains *something*.
static int cmd_train_det_gradcheck(int argc, char** argv) {
  uint64_t seed = strtoull(arg_of(argc, argv, "--seed", "7").c_str(), nullptr, 10);
  const int imgsz = 64, B = 2, nc = 1, reg = 16;
  const int64_t hw[3] = {8, 4, 2};
  Rng rng(seed);
  auto randn = [&]() {                                   // Box-Muller off the project's splitmix64
    double u1 = std::max(1e-12, rng.unit()), u2 = rng.unit();
    return (float)(std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * u2));
  };
  std::vector<Tensor> bx, cs;
  std::vector<float> strides;
  for (int l = 0; l < 3; ++l) {
    Tensor b = make_tensor({B, 4 * reg, hw[l], hw[l]}, true);
    Tensor c = make_tensor({B, nc, hw[l], hw[l]}, true);
    for (float& v : b->data) v = randn();
    for (float& v : c->data) v = randn() - 2.f;          // scores start low, as in a real head
    bx.push_back(b); cs.push_back(c);
    strides.push_back((float)imgsz / (float)hw[l]);
  }
  std::vector<std::vector<std::array<float, 5>>> gts(B);
  gts[0].push_back({0, 6, 10, 40, 30});
  gts[0].push_back({0, 30, 34, 60, 52});
  gts[1].push_back({0, 2, 2, 20, 14});
  det::LossCfg cfg;
  det::LossOut rep;
  Tensor loss = det::v8_loss(bx, cs, strides, gts, cfg, &rep);
  backward(loss);
  printf("gradcheck: loss %.6f (box %.4f cls %.4f dfl %.4f, %d fg anchors)\n", rep.total, rep.box,
         rep.cls, rep.dfl, rep.fg);

  auto num_grad = [&](Tensor t, int64_t i, float h) {
    const float keep = t->data[i];
    det::LossOut r1, r2;
    t->data[i] = keep + h;
    Tensor l1 = det::v8_loss(bx, cs, strides, gts, cfg, &r1);
    t->data[i] = keep - h;
    Tensor l2 = det::v8_loss(bx, cs, strides, gts, cfg, &r2);
    t->data[i] = keep;
    const float d = (l1->data[0] - l2->data[0]) / (2 * h);
    free_graph(l1); free_graph(l2);
    return d;
  };
  double worst = 0, worst_a = 0, worst_n = 0;
  int checked = 0, skipped = 0;
  for (int k = 0; k < 3; ++k)
    for (int which = 0; which < 2; ++which) {
      Tensor t = which ? cs[k] : bx[k];
      for (int s = 0; s < 25; ++s) {
        const int64_t i = (int64_t)rng.below((uint64_t)t->numel());
        const float a = t->grad[i];
        float n = num_grad(t, i, 1e-2f);
        // A finite difference across an assignment flip is meaningless (the loss is piecewise
        // smooth, not smooth), so a disagreeing pair is re-tested at a tenth of the step: if the
        // two agree there, the wide step straddled a discontinuity, not a bug.
        double rel = std::fabs(a - n) / std::max(1e-4f, std::max(std::fabs(a), std::fabs(n)));
        if (rel > 1e-2) {
          const float n2 = num_grad(t, i, 1e-3f);
          double rel2 = std::fabs(a - n2) / std::max(1e-4f, std::max(std::fabs(a), std::fabs(n2)));
          if (rel2 < rel) { rel = rel2; n = n2; }
          if (rel > 1e-2) { ++skipped; continue; }
        }
        ++checked;
        if (rel > worst) { worst = rel; worst_a = a; worst_n = n; }
      }
    }
  free_graph(loss);
  printf("gradcheck: %d entries, worst relative error %.3e (analytic %.6f vs numeric %.6f)%s\n",
         checked, worst, worst_a, worst_n, skipped ? "" : "");
  if (skipped) printf("gradcheck: %d entries sat on an assignment flip and were not compared\n", skipped);
  printf("gradcheck: %s\n", worst < 1e-2 ? "PASS" : "FAIL");
  return worst < 1e-2 ? 0 : 1;
}

// jlpr train --model det — fine-tune the yolov8 nc=1 detector ONNX in place (pure/train_det.hpp).
// The loss is pinned to Ultralytics' (tools/parity/train_det.py); everything around it — SGD with
// warmup and a decaying schedule, EMA, mosaic/flip/hsv/affine augmentation, close_mosaic, freezing,
// validation-driven checkpoint selection — is the same recipe tools/train_det.py asks Ultralytics for,
// so a run here is a run there minus the speed.
static int cmd_train_det(int argc, char** argv) {
  if (has_flag(argc, argv, "--gradcheck")) return cmd_train_det_gradcheck(argc, argv);
  std::string onnx_in = arg_of(argc, argv, "--init", "models/plate_det_v8n_320.onnx");
  std::string data = arg_of(argc, argv, "--data", "");
  std::string valdir = arg_of(argc, argv, "--val", "");
  std::string out = arg_of(argc, argv, "--export", "");
  std::string out_best = arg_of(argc, argv, "--export-best", "");
  std::string fixture = arg_of(argc, argv, "--dump-fixture", "");
  std::string optim = arg_of(argc, argv, "--optim", "sgd");
  int imgsz = std::atoi(arg_of(argc, argv, "--imgsz", "0").c_str());
  int steps = std::atoi(arg_of(argc, argv, "--steps", "30").c_str());
  int epochs = std::atoi(arg_of(argc, argv, "--epochs", "0").c_str());
  int batch = std::atoi(arg_of(argc, argv, "--batch", "2").c_str());
  int limit = std::atoi(arg_of(argc, argv, "--limit", "0").c_str());
  int freeze = std::atoi(arg_of(argc, argv, "--freeze", "0").c_str());
  int val_every = std::atoi(arg_of(argc, argv, "--val-every", "0").c_str());
  int val_limit = std::atoi(arg_of(argc, argv, "--val-limit", "0").c_str());
  float lr0 = (float)atof(arg_of(argc, argv, "--lr", optim == "sgd" ? "0.002" : "1e-4").c_str());
  float lrf = (float)atof(arg_of(argc, argv, "--lrf", "0.01").c_str());
  float momentum = (float)atof(arg_of(argc, argv, "--momentum", "0.937").c_str());
  float warm_mom = (float)atof(arg_of(argc, argv, "--warmup-momentum", "0.8").c_str());
  float wd = (float)atof(arg_of(argc, argv, "--weight-decay", "0.0005").c_str());
  float ema_decay = (float)atof(arg_of(argc, argv, "--ema-decay", "0.9999").c_str());
  float close_mosaic = (float)atof(arg_of(argc, argv, "--close-mosaic", "0.1").c_str());
  uint64_t seed = strtoull(arg_of(argc, argv, "--seed", "1234").c_str(), nullptr, 10);
  bool cos_lr = has_flag(argc, argv, "--cos-lr");
  bool no_aug = has_flag(argc, argv, "--no-aug");
  bool no_ema = has_flag(argc, argv, "--no-ema");
  bool dump_loss = has_flag(argc, argv, "--dump-loss");

  det::AugCfg aug;
  aug.mosaic = (float)atof(arg_of(argc, argv, "--mosaic", "1.0").c_str());
  aug.fliplr = (float)atof(arg_of(argc, argv, "--fliplr", "0.5").c_str());
  aug.degrees = (float)atof(arg_of(argc, argv, "--degrees", "5.0").c_str());
  aug.translate = (float)atof(arg_of(argc, argv, "--translate", "0.15").c_str());
  aug.scale = (float)atof(arg_of(argc, argv, "--scale", "0.6").c_str());
  aug.hsv_h = (float)atof(arg_of(argc, argv, "--hsv-h", "0.015").c_str());
  aug.hsv_s = (float)atof(arg_of(argc, argv, "--hsv-s", "0.6").c_str());
  aug.hsv_v = (float)atof(arg_of(argc, argv, "--hsv-v", "0.4").c_str());

  if (data.empty()) { printf("jlpr train --model det: pass --data <dir with images/ and labels/>\n"); return 1; }
  std::vector<det::Item> items = det::read_yolo(data);
  if (limit > 0 && (int)items.size() > limit) items.resize((size_t)limit);
  if (items.empty()) { printf("no images under %s/images\n", data.c_str()); return 1; }
  size_t nbox = 0;
  for (const det::Item& it : items) nbox += it.boxes.size();
  const int per_epoch = std::max(1, (int)((items.size() + batch - 1) / batch));
  if (epochs > 0) steps = epochs * per_epoch;
  // Ultralytics warms up over 3 epochs, floored at 100 iterations; a short smoke run must not spend
  // itself entirely in the warmup, so it is also capped at a fifth of the run.
  int warmup = std::atoi(arg_of(argc, argv, "--warmup", "-1").c_str());
  if (warmup < 0) warmup = std::min(std::max(100, 3 * per_epoch), std::max(1, steps / 5));
  // close_mosaic: a fraction of the run (0.1 = the last tenth), or a step count if > 1.
  const int mosaic_off_at = close_mosaic > 1.f ? steps - (int)close_mosaic
                                              : steps - (int)(close_mosaic * steps);

  onx::Graph g = onx::load_onnx(onnx_in);
  det::HeadNames hn;
  std::string why;
  if (!det::find_v8_heads(g, hn, &why)) { printf("%s: %s\n", onnx_in.c_str(), why.c_str()); return 1; }
  if (imgsz <= 0) {
    int iw = 0, ih = 0;
    jl::graph_input_hw(g, iw, ih);
    imgsz = iw > 0 ? iw : 320;
  }
  // Only the six head convs feed the loss, so only the subgraph behind them holds parameters:
  // the DFL projection and the decode tail must stay exactly as exported (see weight_initializers).
  std::set<std::string> needed(hn.box.begin(), hn.box.end());
  needed.insert(hn.cls.begin(), hn.cls.end());
  onx::Trainable t = onx::make_trainable(g, false, needed);

  // --freeze N: hold the first N `model.<i>.` modules still, the same way Ultralytics' --freeze does.
  // The tensors stay in the graph (and in write_back); they just leave the optimiser.
  std::vector<Tensor> opt_params;
  int frozen = 0;
  for (size_t i = 0; i < t.params.size(); ++i) {
    bool hold = false;
    if (freeze > 0) {
      const std::string& n = t.param_names[i];
      if (n.rfind("model.", 0) == 0) {
        const size_t dot = n.find('.', 6);
        const int mod = std::atoi(n.substr(6, dot - 6).c_str());
        hold = mod < freeze;
      }
    }
    if (hold) ++frozen; else opt_params.push_back(t.params[i]);
  }

  if (!dump_loss) {
    printf("%s: %zu trainable tensors (%d frozen), %zu parameters\n", onnx_in.c_str(),
           t.params.size(), frozen, onx::param_count(t));
    printf("heads: box %s ... (%zu levels), cls %s ...\n", hn.box[0].c_str(), hn.box.size(),
           hn.cls[0].c_str());
    printf("data: %zu images, %zu boxes, imgsz %d, batch %d (%d/epoch), %d steps%s\n", items.size(),
           nbox, imgsz, batch, per_epoch, steps, epochs > 0 ? " (from --epochs)" : "");
    printf("optim: %s lr %g -> %g (%s), warmup %d, momentum %g, wd %g, EMA %s\n", optim.c_str(), lr0,
           lr0 * lrf, cos_lr ? "cosine" : "linear", warmup, momentum, wd, no_ema ? "off" : "on");
    printf("aug: %s\n", no_aug ? "off (--no-aug)" : "mosaic/flip/hsv/affine, mosaic closes at step "
                                                    "(see --close-mosaic)");
  }

  // --check-aug: does the augmentation move the boxes with the pixels? Build augmented batches and
  // run the *already trained* detector over them: if the affine/mosaic/flip maths were wrong, the
  // labels would no longer sit on the plates and the recall would collapse. Cheap, and it tests the
  // one thing a loss-parity test cannot see (the loss happily trains on wrong boxes).
  int check_aug = std::atoi(arg_of(argc, argv, "--check-aug", "0").c_str());
  if (check_aug > 0) {
    jl::DetCfg ccfg;
    ccfg.kind = jl::DetKind::V8;
    ccfg.nc = 1;
    ccfg.plate_class = 0;
    ccfg.imgsz = imgsz;
    ccfg.conf = 0.25f;
    ccfg.nms = 0.45f;
    ccfg.v8_fmt = (arg_of(argc, argv, "--fmt", "cxcywh") == "xyxy") ? BoxFmt::XYXY : BoxFmt::CXCYWH;
    for (int pass = 0; pass < 2; ++pass) {              // 0 = plain resize, 1 = augmented
      Rng crng(seed);
      int gt_n = 0, hit = 0;
      double iou_sum = 0;
      for (int k = 0; k < check_aug; ++k) {
        std::vector<int> idx;
        for (int b = 0; b < batch; ++b) idx.push_back((int)crng.below((uint64_t)items.size()));
        det::Batch ba = pass ? det::make_batch_aug(items, idx, imgsz, crng, aug, true)
                             : det::make_batch(items, idx, imgsz);
        for (int b = 0; b < batch; ++b) {
          std::vector<unsigned char> rgb((size_t)imgsz * imgsz * 3);
          for (int c = 0; c < 3; ++c)
            for (int y = 0; y < imgsz; ++y)
              for (int x = 0; x < imgsz; ++x)
                rgb[((size_t)y * imgsz + x) * 3 + c] = (unsigned char)std::lround(
                    255.f * ba.x->data[((size_t)b * 3 + c) * imgsz * imgsz + (size_t)y * imgsz + x]);
          std::vector<jl::Box> got = jl::detect_plates(g, rgb.data(), imgsz, imgsz, ccfg);
          for (const auto& q : ba.gts[(size_t)b]) {
            ++gt_n;
            evd::GtBox gt{q[1], q[2], q[3], q[4], 0};
            float best = 0;
            for (const jl::Box& d : got) best = std::max(best, evd::iou_xyxy(gt, {d.x1, d.y1, d.x2, d.y2, d.score}));
            iou_sum += best;
            if (best >= 0.5f) ++hit;
          }
        }
      }
      printf("check-aug [%s]: %d labels, the shipped detector finds %.1f%% of them (mean best IoU %.3f)\n",
             pass ? "augmented" : "plain resize", gt_n, 100.0 * hit / std::max(1, gt_n),
             iou_sum / std::max(1, gt_n));
    }
    return 0;
  }

  SGD sgd(opt_params, lr0, momentum, wd, true);
  Adam adam(opt_params, lr0, 0.9f, 0.999f, 1e-8f, wd, true);
  const bool use_sgd = optim != "adam";
  Ema ema(t.params, ema_decay);
  Rng rng(seed);
  det::LossCfg cfg;

  jl::DetCfg vcfg;                       // validation runs the product path: full graph + NMS
  vcfg.kind = jl::DetKind::V8;
  vcfg.nc = 1;
  vcfg.plate_class = 0;
  vcfg.imgsz = imgsz;
  vcfg.conf = 0.001f;
  vcfg.nms = 0.45f;
  vcfg.v8_fmt = (arg_of(argc, argv, "--fmt", "cxcywh") == "xyxy") ? BoxFmt::XYXY : BoxFmt::CXCYWH;

  std::map<std::string, std::vector<float>> best_snap;
  float best_map = -1.f;
  int best_step = 0;
  auto snapshot = [&]() {
    std::map<std::string, std::vector<float>> s;
    for (const auto& kv : t.init) s[kv.first] = kv.second->data;
    return s;
  };
  auto restore = [&](const std::map<std::string, std::vector<float>>& s) {
    for (const auto& kv : s) {
      auto it = t.init.find(kv.first);
      if (it != t.init.end()) it->second->data = kv.second;
    }
  };
  // Validation and export both want the EMA weights, not the live ones (measured everywhere in the
  // YOLO literature; the live weights at step k are noisier than their average).
  auto with_ema = [&](const std::function<void()>& fn) {
    if (!no_ema) ema.swap();
    fn();
    if (!no_ema) ema.swap();
  };
  auto validate = [&](int step) {
    if (valdir.empty()) return;
    with_ema([&] {
      onx::write_back(t);
      DetEval e = eval_det_dir(t.g, vcfg, valdir, val_limit, 0.25f, 0.5f, false);
      printf("  val @%d: mAP50 %.4f  mAP50-95 %.4f  P %.3f R %.3f  (%d gt over %zu frames)\n", step,
             e.r.map50, e.r.map5095, e.r.precision, e.r.recall, e.n_gt, e.frames);
      if (e.r.map5095 > best_map) {
        best_map = e.r.map5095;
        best_step = step;
        best_snap = snapshot();          // inside with_ema: this saves the averaged weights
        printf("    <- best\n");
      }
    });
    fflush(stdout);
  };

  for (int step = 0; step < steps; ++step) {
    // schedule: linear warmup (momentum too, as Ultralytics does), then linear or cosine decay to
    // lr0*lrf. Both are computed per step rather than per epoch because this trainer counts steps.
    const float x = (float)step / (float)std::max(1, steps);
    float lr = cos_lr ? lr0 * (lrf + (1 - lrf) * 0.5f * (1 + std::cos(3.14159265358979f * x)))
                      : lr0 * ((1 - x) * (1 - lrf) + lrf);
    float mom = momentum;
    if (step < warmup) {
      const float w = (float)(step + 1) / (float)warmup;
      lr *= w;
      mom = warm_mom + (momentum - warm_mom) * w;
    }
    sgd.lr = lr; sgd.momentum = mom; adam.lr = lr;

    // With fewer images than the batch size, every step sees all of them in order: that is the
    // "does this train at all" case, and a random sampler would hide the answer behind which images
    // the draw happened to pick (measured: batch 2 out of 8 frames swings the loss by 2x per step,
    // purely because frames carry 0-3 plates).
    std::vector<int> idx;
    if ((int)items.size() <= batch) for (size_t i = 0; i < items.size(); ++i) idx.push_back((int)i);
    else for (int b = 0; b < batch; ++b) idx.push_back((int)rng.below((uint64_t)items.size()));
    det::Batch ba = no_aug ? det::make_batch(items, idx, imgsz)
                           : det::make_batch_aug(items, idx, imgsz, rng, aug, step < mosaic_off_at);
    det::LossOut rep;
    std::vector<Tensor> bxs, css;
    Tensor loss = det::forward_loss(t, hn, ba.x, ba.gts, imgsz, cfg, &rep, &bxs, &css);
    if (use_sgd) sgd.zero_grad(); else adam.zero_grad();
    backward(loss);
    if (dump_loss) printf("%d %.6f %.6f %.6f %.6f\n", step, rep.total, rep.box, rep.cls, rep.dfl);
    else printf("step %d: loss %.4f (box %.4f cls %.4f dfl %.4f) fg %d tss %.2f lr %.2e\n", step,
                rep.total, rep.box, rep.cls, rep.dfl, rep.fg, rep.tss, lr);
    fflush(stdout);
    if (!fixture.empty() && step == 0) {
      // Step 0's head tensors, gts, loss and gradients — the input tools/parity/train_det.py hands
      // to ultralytics' own v8DetectionLoss so both sides differentiate the *same* numbers.
      FILE* f = fopen(fixture.c_str(), "wb");
      if (f) {
        auto wi = [&](int32_t v) { fwrite(&v, 4, 1, f); };
        fwrite("JLPRDET1", 1, 8, f);
        wi((int32_t)bxs.size()); wi((int32_t)bxs[0]->shape[0]);
        wi((int32_t)css[0]->shape[1]); wi((int32_t)(bxs[0]->shape[1] / 4)); wi((int32_t)imgsz);
        for (size_t l = 0; l < bxs.size(); ++l) wi((int32_t)bxs[l]->shape[2]);   // square feature maps
        for (const Tensor& b : bxs) fwrite(b->data.data(), 4, (size_t)b->numel(), f);
        for (const Tensor& c : css) fwrite(c->data.data(), 4, (size_t)c->numel(), f);
        int32_t ngt = 0;
        for (const auto& v : ba.gts) ngt += (int32_t)v.size();
        wi(ngt);
        for (size_t b = 0; b < ba.gts.size(); ++b)
          for (const auto& q : ba.gts[b]) { wi((int32_t)b); fwrite(q.data(), 4, 5, f); }
        float parts[4] = {rep.total, rep.box, rep.cls, rep.dfl};
        fwrite(parts, 4, 4, f);
        for (const Tensor& b : bxs) fwrite(b->grad.data(), 4, (size_t)b->numel(), f);
        for (const Tensor& c : css) fwrite(c->grad.data(), 4, (size_t)c->numel(), f);
        fclose(f);
        printf("wrote %s (head tensors, gts, loss and gradients of step 0)\n", fixture.c_str());
      }
    }
    if (use_sgd) sgd.step(); else adam.step();
    if (!no_ema) ema.update();
    free_graph(loss);
    if (val_every > 0 && (step + 1) % val_every == 0) validate(step + 1);
  }
  if (!valdir.empty() && (val_every <= 0 || steps % val_every != 0)) validate(steps);   // final

  if (!out_best.empty() && !best_snap.empty()) {
    std::map<std::string, std::vector<float>> live = snapshot();
    restore(best_snap);
    onx::write_back(t);
    onx::save_onnx(t.g, out_best);
    printf("wrote %s (best val: mAP50-95 %.4f at step %d)\n", out_best.c_str(), best_map, best_step);
    restore(live);
  }
  if (!out.empty()) {
    with_ema([&] {
      onx::write_back(t);
      onx::save_onnx(t.g, out);
    });
    printf("wrote %s%s\n", out.c_str(), no_ema ? "" : " (EMA weights)");
  }
  return 0;
}

// jlpr train --model corner — train the 4-corner regressor (pure/train_corner.hpp). Mirrors
// tools/train_corner.py, down to the order of the random draws, so the two see the same batch.
static int cmd_train_corner(int argc, char** argv) {
  std::string synth = arg_of(argc, argv, "--synth", "");
  std::string valdir = arg_of(argc, argv, "--val", "");
  std::string init = arg_of(argc, argv, "--init", "random");
  std::string out = arg_of(argc, argv, "--export", "");
  std::string fixture = arg_of(argc, argv, "--dump-fixture", "");
  int width = std::atoi(arg_of(argc, argv, "--width", "24").c_str());
  int steps = std::atoi(arg_of(argc, argv, "--steps", "2000").c_str());
  int batch = std::atoi(arg_of(argc, argv, "--batch", "64").c_str());
  float lr0 = (float)atof(arg_of(argc, argv, "--lr", "1e-3").c_str());
  float wd = (float)atof(arg_of(argc, argv, "--weight-decay", "1e-4").c_str());
  float beta = (float)atof(arg_of(argc, argv, "--huber-beta", "0.02").c_str());
  float jitter = (float)atof(arg_of(argc, argv, "--jitter", "0.04").c_str());
  float expand_lo = (float)atof(arg_of(argc, argv, "--expand-lo", "0.05").c_str());
  float expand_hi = (float)atof(arg_of(argc, argv, "--expand-hi", "0.45").c_str());
  float clip = (float)atof(arg_of(argc, argv, "--clip", "5.0").c_str());
  int eval_every = std::atoi(arg_of(argc, argv, "--eval-every", "250").c_str());
  uint64_t seed = strtoull(arg_of(argc, argv, "--seed", "77").c_str(), nullptr, 10);
  bool dump_loss = has_flag(argc, argv, "--dump-loss");   // parity: the per-step loss only

  if (synth.empty()) {
    printf("usage: jlpr train --model corner --synth <dir with corners.txt> [--val dir] "
           "[--init random|onnx] [--width N] [--steps N] [--batch N] [--lr f] [--export onnx]\n");
    return 1;
  }
  std::vector<crn::Item> train = crn::read_corners(synth);
  if (train.empty()) { printf("no corners.txt in %s (generate with `jlpr gen`)\n", synth.c_str()); return 1; }
  std::vector<crn::Item> val = valdir.empty() ? train : crn::read_corners(valdir);

  onx::Graph g = (init == "random") ? crn::build_graph(width, seed) : onx::load_onnx(init);
  onx::Trainable t = onx::make_trainable(g);
  if (!dump_loss)
    printf("corner training: %zu crops (val %zu), init %s, %zu tensors / %zu parameters\n",
           train.size(), val.size(), init.c_str(), t.params.size(), onx::param_count(t));

  Adam opt(t.params, lr0, 0.9f, 0.999f, 1e-8f, wd, true);        // AdamW (decoupled), as in torch
  Rng rng(seed);
  double run = -1;
  for (int step = 1; step <= steps; ++step) {
    // the Python schedule verbatim: linear warmup over 100 steps, multiplied by a cosine over the run
    const float lr = lr0 * std::min(1.f, (float)step / 100.f) *
                     (0.5f * (1.f + std::cos(3.14159265358979f * (float)step / (float)steps)));
    opt.lr = lr;
    crn::Batch ba = crn::make_batch(train, rng, batch, jitter, expand_lo, expand_hi);
    std::map<std::string, Tensor> vals = onx::run_onnx(t.g, ba.x, {}, &t.init, true);
    Tensor pred = vals.at(t.g.outputs[0].name);
    Tensor loss = crn::smooth_l1(pred, ba.y, beta);
    opt.zero_grad();
    backward(loss);
    crn::clip_grad_norm(t.params, clip);
    if (!fixture.empty() && step == 1) {
      // step 1's batch, loss and every parameter gradient — what tools/parity/train_corner.py
      // replays through the PyTorch CornerNet. Written after clip_grad_norm and before opt.step(),
      // so the weights are still the ones in --init and the gradients are the ones about to be used.
      FILE* f = fopen(fixture.c_str(), "wb");
      if (f) {
        auto wi = [&](int32_t v) { fwrite(&v, 4, 1, f); };
        fwrite("JLPRCRN1", 1, 8, f);
        wi(batch); wi(crn::IN_PX); wi((int32_t)t.params.size());
        fwrite(ba.x->data.data(), 4, (size_t)ba.x->numel(), f);
        fwrite(ba.y.data(), 4, ba.y.size(), f);
        float lv = loss->data[0];
        fwrite(&lv, 4, 1, f);
        for (size_t i = 0; i < t.params.size(); ++i) {
          wi((int32_t)t.param_names[i].size());
          fwrite(t.param_names[i].data(), 1, t.param_names[i].size(), f);
          wi((int32_t)t.params[i]->numel());
          fwrite(t.params[i]->grad.data(), 4, (size_t)t.params[i]->numel(), f);
        }
        fclose(f);
        if (!dump_loss) printf("wrote %s (batch, loss and %zu parameter gradients of step 1)\n",
                               fixture.c_str(), t.params.size());
      }
    }
    opt.step();
    const double lv = loss->data[0];
    run = run < 0 ? lv : 0.9 * run + 0.1 * lv;
    free_graph(loss);
    if (dump_loss) printf("%d %.8f\n", step, lv);
    else if (step % 50 == 0 || step == 1)
      printf("  step %5d/%d  loss %.5f  lr %.2e\n", step, steps, run, lr);
    if (!dump_loss && eval_every > 0 && step % eval_every == 0) {
      printf("  eval @%d: mean corner error %.2f%% of plate width\n", step,
             100 * crn::eval_error(t, val));
      fflush(stdout);
    }
    fflush(stdout);
  }
  if (!dump_loss)
    printf("final: mean corner error %.2f%% of plate width (target <= 1.5%%)\n",
           100 * crn::eval_error(t, val, 800));
  if (!out.empty()) {
    onx::write_back(t);                      // weights *and* the BN running stats this trained
    onx::save_onnx(t.g, out);
    printf("wrote %s\n", out.c_str());
  }
  return 0;
}

// jlpr train --model ocr — fine-tune the recognizer ONNX in place (see pure/train_ocr.hpp).
static int cmd_train(int argc, char** argv) {
  std::string model = arg_of(argc, argv, "--model", "ocr");
  if (model == "det") return cmd_train_det(argc, argv);
  if (model == "corner") return cmd_train_corner(argc, argv);
  if (model != "ocr") { printf("jlpr train: --model must be ocr, det or corner\n"); return 1; }
  std::string onnx_in = arg_of(argc, argv, "--init", "models/plate_ocr.onnx");
  std::string spec_path = arg_of(argc, argv, "--spec", "spec/labels.txt");
  std::string synth = arg_of(argc, argv, "--synth", "");
  std::string alpr = arg_of(argc, argv, "--alpr", "");
  std::string out = arg_of(argc, argv, "--export", "");            // best real hold-out checkpoint
  std::string out_last = arg_of(argc, argv, "--export-last", "");  // the final step
  std::string out_balanced = arg_of(argc, argv, "--export-balanced", "");
  double select_margin = atof(arg_of(argc, argv, "--select-margin", "1.0").c_str());
  int steps = std::atoi(arg_of(argc, argv, "--steps", "100").c_str());
  int batch = std::atoi(arg_of(argc, argv, "--batch", "8").c_str());
  float lr = (float)atof(arg_of(argc, argv, "--lr", "3e-4").c_str());
  float bb_mult = (float)atof(arg_of(argc, argv, "--backbone-lr-mult", "0.1").c_str());
  double real_w = atof(arg_of(argc, argv, "--real-weight", "0.35").c_str());
  int eval_every = std::atoi(arg_of(argc, argv, "--eval-every", "0").c_str());
  int eval_limit = std::atoi(arg_of(argc, argv, "--eval-limit", "144").c_str());
  uint64_t seed = strtoull(arg_of(argc, argv, "--seed", "1234").c_str(), nullptr, 10);
  bool synth_region = has_flag(argc, argv, "--synth-region");
  bool dump_loss = has_flag(argc, argv, "--dump-loss");   // parity: print the per-step loss only

  spec::Spec sp = spec::load(spec_path);

  std::vector<trn::Item> synth_items, alpr_items;
  std::vector<int> alpr_train, alpr_val;
  if (!alpr.empty()) {
    alpr_items = trn::read_alpr(alpr, sp);
    trn::split_holdout(alpr_items.size(), 0.2, alpr_train, alpr_val);
  }
  // regions the real training split covers -> the rest are taught from synthetic (see read_synth)
  std::set<int> uncovered;
  if (!alpr_items.empty()) {
    std::set<int> covered;
    for (int i : alpr_train) covered.insert(alpr_items[(size_t)i].heads[0]);
    for (int r = 0; r < sp.head("region").n; ++r) if (!covered.count(r)) uncovered.insert(r);
    if (!dump_loss)
      printf("real data covers %zu of %d regions; synthetic teaches the other %zu\n", covered.size(),
             sp.head("region").n, uncovered.size());
  }
  if (!synth.empty()) synth_items = trn::read_synth(synth, synth_region, uncovered);

  onx::Graph g = onx::load_onnx(onnx_in);
  // Bring the shipped graph up to the spec (region 133->138). The appended classes start inert at -10
  // unless something will actually train them, in which case 0 — see onx::widen_heads.
  std::map<std::string, int> want;
  for (const spec::Group* gr : sp.of_kind("head")) want[gr->name] = gr->n;
  float new_bias = uncovered.empty() ? -10.f : 0.f;
  std::string nb = arg_of(argc, argv, "--new-class-bias", "");
  if (!nb.empty()) new_bias = (float)atof(nb.c_str());
  std::vector<std::string> widened = onx::widen_heads(g, want, new_bias);
  onx::Trainable t = onx::make_trainable(g);
  if (!dump_loss) {
    printf("%s: %zu trainable tensors, %zu parameters, %zu heads\n", onnx_in.c_str(),
           t.params.size(), onx::param_count(t), t.heads.size());
    if (!widened.empty()) {
      printf("widened to the spec (new-class bias %.1f):", new_bias);
      for (const std::string& w : widened) printf(" %s", w.c_str());
      printf("\n");
    }
  }
  if (synth_items.empty() && alpr_items.empty()) {
    printf("no data: pass --synth <dir> and/or --alpr <root>\n");
    return 1;
  }
  if (!dump_loss) {
    char reg_note[96] = " (region masked)";
    if (synth_region) snprintf(reg_note, sizeof reg_note, " (region taught for every name)");
    else if (!uncovered.empty())
      snprintf(reg_note, sizeof reg_note, " (region taught for the %zu names real data lacks)",
               uncovered.size());
    printf("synthetic %zu crops%s, real %zu crops (%zu train / %zu hold-out)\n", synth_items.size(),
           reg_note, alpr_items.size(), alpr_train.size(), alpr_val.size());
  }

  // A held-out slice of the synthetic crops, used only to watch the region names real data lacks.
  std::vector<int> synth_eval;
  if (!synth_items.empty()) {
    for (size_t i = 0; i < synth_items.size(); i += 1 + synth_items.size() / 200)
      synth_eval.push_back((int)i);
  }
  trn::Snapshot best, bal;
  double best_acc = -1.0, bal_acc = -1.0, bal_syn = -1.0;
  int best_step = 0, bal_step = 0;

  std::vector<const std::vector<trn::Item>*> sets;
  std::vector<const std::vector<int>*> pools;
  std::vector<double> weights;
  std::vector<std::pair<double, double>> margins;
  std::vector<bool> photo;
  if (!synth_items.empty()) {
    sets.push_back(&synth_items); pools.push_back(nullptr);
    weights.push_back(1.0 - real_w); margins.push_back({-0.03, 0.12}); photo.push_back(false);
  }
  if (!alpr_items.empty()) {
    sets.push_back(&alpr_items); pools.push_back(&alpr_train);
    weights.push_back(real_w); margins.push_back({-0.04, 0.12}); photo.push_back(true);
  }

  // head tensors get lr, the pretrained backbone gets lr*mult (same split as the Python trainer)
  std::vector<Tensor> head_params, bb_params;
  for (size_t i = 0; i < t.params.size(); ++i) {
    const std::string& n = t.param_names[i];
    (n.rfind("head", 0) == 0 ? head_params : bb_params).push_back(t.params[i]);
  }
  Adam opt_head(head_params, lr), opt_bb(bb_params, lr * bb_mult);
  if (!dump_loss)
    printf("param groups: %zu head tensors at lr, %zu backbone tensors at lr*%.2f, BN stats frozen\n",
           head_params.size(), bb_params.size(), bb_mult);

  size_t region_head = 0;
  for (size_t h = 0; h < t.heads.size(); ++h)
    if (t.heads[h].rfind("region", 0) == 0) region_head = h;

  if (!alpr_val.empty() && !dump_loss) {
    int n = 0;
    double acc = trn::eval_region(t, alpr_items, alpr_val, region_head, 0, &n);
    printf("step 0 (this ONNX): real hold-out region top1 %.1f%% over %d crops\n", 100 * acc, n);
  }

  Rng rng(seed);
  double run_loss = -1;
  for (int step = 1; step <= steps; ++step) {
    float cur = step <= 1 ? lr : lr;                        // constant lr keeps the parity simple
    opt_head.lr = cur;
    opt_bb.lr = cur * bb_mult;
    trn::Batch b = trn::make_batch(sets, pools, weights, margins, photo, batch, rng);
    auto vals = onx::forward(t, b.x);
    Tensor loss = onx::multihead_ce(vals, t, b.labels, b.mask);
    if (!loss) continue;
    for (Tensor& p : t.params) std::fill(p->grad.begin(), p->grad.end(), 0.f);
    backward(loss);
    opt_head.step();
    opt_bb.step();
    double lv = loss->data[0];
    run_loss = run_loss < 0 ? lv : 0.9 * run_loss + 0.1 * lv;
    free_graph(loss);
    if (dump_loss) printf("step %d loss %.6f\n", step, lv);
    else if (step % 5 == 0 || step == 1) printf("  step %4d/%d  loss %7.3f\n", step, steps, run_loss);
    if (!dump_loss && eval_every && step % eval_every == 0 && !alpr_val.empty()) {
      int n = 0;
      double acc = trn::eval_region(t, alpr_items, alpr_val, region_head, eval_limit, &n);
      printf("  eval @%d: real region %.1f%% (%d)", step, 100 * acc, n);
      // The synthetic region score is the only view of the names real data has none of. Same routine,
      // different item list: synthetic items carry a region label too.
      double syn = -1.0;
      if (!synth_eval.empty()) {
        int sn = 0;
        syn = trn::eval_region(t, synth_items, synth_eval, region_head, eval_limit, &sn);
        printf("  synth region %.1f%% (%d)", 100 * syn, sn);
      }
      if (acc > best_acc) {
        best_acc = acc; best_step = step; best = trn::snapshot(t);
        printf("  <- best");
      }
      if (syn > bal_syn && acc >= best_acc - select_margin / 100.0) {
        bal_syn = syn; bal_acc = acc; bal_step = step; bal = trn::snapshot(t);
        printf("  <- balanced");
      }
      printf("%c", 0x0a);
    }
  }
  if (!alpr_val.empty() && !dump_loss) {
    int n = 0;
    double acc = trn::eval_region(t, alpr_items, alpr_val, region_head, 0, &n);
    printf("final: real hold-out region top1 %.1f%% over %d crops\n", 100 * acc, n);
    if (acc > best_acc) { best_acc = acc; best_step = steps; best = trn::snapshot(t); }
  }

  // Export order matters: write the alternates first, then leave the graph holding the model that
  // `--export` names, so a caller that only looks at one file gets the intended one.
  auto save_as = [&](const std::string& path, const trn::Snapshot& snap, const char* what,
                     int at_step, double racc, double sacc) {
    if (path.empty() || snap.empty()) return;
    trn::Snapshot keep = trn::snapshot(t);
    trn::restore(t, snap);
    onx::write_back(t);
    onx::save_onnx(t.g, path);
    if (sacc >= 0)
      printf("wrote %s (%s: step %d, real %.1f%% / synth region %.1f%%)\n", path.c_str(), what,
             at_step, 100 * racc, 100 * sacc);
    else
      printf("wrote %s (%s: step %d, real %.1f%%)\n", path.c_str(), what, at_step, 100 * racc);
    trn::restore(t, keep);
  };
  if (!out_last.empty()) {
    onx::write_back(t);
    onx::save_onnx(t.g, out_last);
    printf("wrote %s (final step)\n", out_last.c_str());
  }
  save_as(out_balanced, bal, "balanced", bal_step, bal_acc, bal_syn);
  if (!out.empty()) {
    if (!best.empty()) save_as(out, best, "best real hold-out", best_step, best_acc, -1.0);
    else {
      onx::write_back(t);
      onx::save_onnx(t.g, out);
      printf("wrote %s\n", out.c_str());
    }
  }
  return 0;
}


// jlpr val --model det — the C++ side of tools/eval_det.py, plus the mAP that used to need
// Ultralytics. The metrics and their two matchings live in pure/eval_det.hpp; the work is in
// eval_det_dir() above, which the trainer's in-run validation calls too.
static int cmd_val_det(int argc, char** argv) {
  std::string data = arg_of(argc, argv, "--data", "");
  std::string det_p = arg_of(argc, argv, "--det", "models/plate_det_v8n_320.onnx");
  int limit = std::atoi(arg_of(argc, argv, "--limit", "0").c_str());
  int imgsz = std::atoi(arg_of(argc, argv, "--imgsz", "0").c_str());
  float conf = (float)atof(arg_of(argc, argv, "--conf", "0.25").c_str());
  float iou_thr = (float)atof(arg_of(argc, argv, "--iou", "0.5").c_str());
  float conf_lo = (float)atof(arg_of(argc, argv, "--conf-lo", "0.001").c_str());
  bool as_json = has_flag(argc, argv, "--json");
  if (data.empty()) {
    printf("usage: jlpr val --model det --data <dir with images/ labels/> [--det onnx] "
           "[--fmt xyxy|cxcywh] [--conf f] [--iou f] [--imgsz N] [--limit N] [--json]\n");
    return 1;
  }
  jl::DetCfg cfg;
  cfg.kind = jl::DetKind::V8;
  cfg.nc = 1;
  cfg.plate_class = 0;
  cfg.imgsz = imgsz;
  cfg.conf = std::min(conf, conf_lo);        // one pass at the low threshold: NMS cannot let a
  cfg.nms = (float)atof(arg_of(argc, argv, "--nms", "0.45").c_str());   // weak box suppress a strong
  cfg.v8_fmt = (arg_of(argc, argv, "--fmt", "cxcywh") == "xyxy") ? BoxFmt::XYXY : BoxFmt::CXCYWH;

  onx::Graph det = onx::load_onnx(det_p);
  DetEval e = eval_det_dir(det, cfg, data, limit, conf, iou_thr, !as_json);
  if (e.frames == 0) { printf("no images under %s/images\n", data.c_str()); return 1; }
  const evd::Report& r = e.r;
  if (as_json) {
    printf("{\"frames\":%zu,\"gt\":%d,\"map50\":%.6f,\"map50_95\":%.6f,\"precision\":%.6f,"
           "\"recall\":%.6f,\"f1\":%.6f,\"tp\":%d,\"fp\":%d,\"buckets\":[",
           e.frames, e.n_gt, r.map50, r.map5095, r.precision, r.recall, r.f1, r.tp, r.fp);
    for (size_t b = 0; b < e.gt_b.size(); ++b)
      printf("%s{\"lo\":%.2f,\"hi\":%.2f,\"gt\":%d,\"found\":%d}", b ? "," : "", evd::buckets()[b][0],
             evd::buckets()[b][1], e.gt_b[b], e.tp_b[b]);
    printf("],\"bucket_fp\":%d,\"empty_frames\":%d,\"fp_on_empty\":%d}\n", e.fp_bucket, e.empty_frames,
           e.fp_empty);
    return 0;
  }
  printf("\n%zu frames, conf>=%.2f, IoU>=%.2f, model=%s\n", e.frames, conf, iou_thr, det_p.c_str());
  printf("%-16s %8s %8s %8s\n", "plate share", "GT", "found", "recall");
  for (size_t b = 0; b < e.gt_b.size(); ++b) {
    if (!e.gt_b[b]) continue;
    char name[32];
    snprintf(name, sizeof name, "%.0f-%.0f%%", evd::buckets()[b][0] * 100,
             std::min(1.f, evd::buckets()[b][1]) * 100);
    printf("%-16s %8d %8d %7.1f%%\n", name, e.gt_b[b], e.tp_b[b], 100.0 * e.tp_b[b] / e.gt_b[b]);
  }
  int tot_g = 0, tot_t = 0;
  for (size_t b = 0; b < e.gt_b.size(); ++b) { tot_g += e.gt_b[b]; tot_t += e.tp_b[b]; }
  printf("%-16s %8d %8d %7.1f%%\n", "all", tot_g, tot_t, 100.0 * tot_t / std::max(1, tot_g));
  printf("false positives: %d (of %d detections); on the %d plate-free frames: %d\n", e.fp_bucket,
         e.n_det_at_conf, e.empty_frames, e.fp_empty);
  printf("mAP50 %.4f  mAP50-95 %.4f   (P %.4f  R %.4f  F1 %.4f at conf %.2f, TP %d FP %d FN %d)\n",
         r.map50, r.map5095, r.precision, r.recall, r.f1, conf, r.tp, r.fp, r.fn);
  return 0;
}

// jlpr val — the C++ side of tools/eval_ocr.py: region top-1/top-3 on real crops, per-head and
// whole-plate on generated ones. Same crops, same reading modes (1 crop / 6-crop TTA / rectified by
// the corner net), same numbers — so a model can be judged with no Python anywhere in the loop.
static int cmd_val(int argc, char** argv) {
  if (arg_of(argc, argv, "--model", "ocr") == "det") return cmd_val_det(argc, argv);
  std::string ocr_p = arg_of(argc, argv, "--ocr", "models/plate_ocr_v2.onnx");
  std::string spec_p = arg_of(argc, argv, "--spec", "spec/labels.txt");
  std::string data = arg_of(argc, argv, "--data", "");
  std::string kind = arg_of(argc, argv, "--kind", "alpr");
  std::string corner_p = arg_of(argc, argv, "--corner", "");
  int limit = std::atoi(arg_of(argc, argv, "--limit", "0").c_str());
  float margin = (float)atof(arg_of(argc, argv, "--margin", "0.03").c_str());
  bool tta = has_flag(argc, argv, "--tta");
  bool holdout = has_flag(argc, argv, "--holdout");
  if (data.empty()) {
    printf("usage: jlpr val --data <dir> [--kind alpr|synth] [--ocr onnx] [--corner onnx] [--tta] "
           "[--margin f] [--limit N] [--holdout]\n");
    return 1;
  }
  spec::Spec sp = spec::load(spec_p);
  onx::Graph g = onx::load_onnx(ocr_p);
  onx::Graph corner;
  const bool use_corner = !corner_p.empty();
  if (use_corner) corner = onx::load_onnx(corner_p);
  std::vector<std::string> heads = jl::ocr_head_names(g);

  std::vector<trn::Item> items = (kind == "synth") ? trn::read_synth(data, true)
                                                  : trn::read_alpr(data, sp);
  if (items.empty()) { printf("no labelled crops under %s (kind=%s)\n", data.c_str(), kind.c_str()); return 1; }
  std::vector<int> idxs;
  if (kind == "alpr" && holdout) {
    std::vector<int> tr;
    trn::split_holdout(items.size(), 0.2, tr, idxs);   // the same split the trainer uses
  } else {
    for (size_t i = 0; i < items.size(); ++i) idxs.push_back((int)i);
  }
  if (limit > 0 && (int)idxs.size() > limit) idxs.resize((size_t)limit);

  const size_t nheads = heads.size();
  std::vector<int> per_head(nheads, 0), per_head_n(nheads, 0);
  std::map<std::string, std::array<int, 2>> by_cat;     // category -> {crops, region correct}
  int n = 0, full_ok = 0, top3 = 0, legible_n = 0, legible_full = 0;
  double conf_sum = 0, conf_right = 0;
  const std::vector<float> one_margin = {margin};
  for (int id : idxs) {
    const trn::Item& it = items[(size_t)id];
    int W = 0, H = 0, C = 0;
    std::vector<unsigned char> blob = trn::read_file(it.path);
    unsigned char* im = blob.empty() ? nullptr
                                     : stbi_load_from_memory(blob.data(), (int)blob.size(), &W, &H, &C, 3);
    if (!im) continue;
    // The file *is* the plate crop unless the generator recorded a box for it.
    jl::Box box{0.f, 0.f, (float)W, (float)H, 1.f};
    if (it.have_box) { box.x1 = it.bx0; box.y1 = it.by0; box.x2 = it.bx1; box.y2 = it.by1; }
    jl::Read rd;
    bool warped = false;
    if (use_corner) {
      float c[8];
      jl::CornerCfg ccfg;
      if (jl::predict_corners(corner, im, W, H, box, ccfg, c)) {
        rd = jl::read_plate_warped(g, sp, im, W, H, c, ccfg);
        warped = true;
      }
    }
    if (!warped)
      rd = jl::read_plate(g, sp, im, W, H, box.x1, box.y1, box.x2, box.y2,
                          tta ? jl::default_margins() : one_margin);
    stbi_image_free(im);

    bool all = true;
    for (size_t h = 0; h < nheads && h < it.heads.size(); ++h) {
      if (it.mask[h] <= 0.5f) continue;
      ++per_head_n[h];
      const bool ok = rd.arg[h] == it.heads[h];
      per_head[h] += ok ? 1 : 0;
      if (h < 9) all = all && ok;
    }
    if (!rd.probs.empty()) {
      // region top-3: the same "is the right name even in the running" question eval_ocr.py asks
      const std::vector<float>& p = rd.probs[0];
      int r1 = 0, r2 = -1, r3 = -1;
      for (size_t i = 1; i < p.size(); ++i) if (p[i] > p[(size_t)r1]) r1 = (int)i;
      for (size_t i = 0; i < p.size(); ++i) {
        if ((int)i == r1) continue;
        if (r2 < 0 || p[i] > p[(size_t)r2]) { r3 = r2; r2 = (int)i; }
        else if (r3 < 0 || p[i] > p[(size_t)r3]) r3 = (int)i;
      }
      conf_sum += p[(size_t)r1];
      if (it.mask[0] > 0.5f) {
        const int want = it.heads[0];
        if (want == r1) { conf_right += p[(size_t)r1]; }
        if (want == r1 || want == r2 || want == r3) ++top3;
      }
    }
    full_ok += all ? 1 : 0;
    if (it.heads.size() > 10 && it.mask[10] > 0.5f && it.heads[10] == 1) {   // labelled legible
      ++legible_n;
      legible_full += all ? 1 : 0;
    }
    if (kind == "alpr" && !it.key.empty()) {
      const std::string cat = it.key.substr(0, it.key.find('/'));
      std::array<int, 2>& c = by_cat[cat];
      c[0] += 1;
      c[1] += (it.mask[0] > 0.5f && rd.arg[0] == it.heads[0]) ? 1 : 0;
    }
    ++n;
    if (n % 200 == 0) { printf("  %d/%zu\n", n, idxs.size()); fflush(stdout); }
  }

  const char* mode = use_corner ? "corner-rectified" : (tta ? "6-crop TTA" : "1 crop");
  printf("\n%s: %d crops from %s (%s, margin %.2f)\n", ocr_p.c_str(), n, data.c_str(), mode, margin);
  if (kind == "alpr") {
    printf("region top1 %.1f%%   top3 %.1f%%   mean conf %.3f (correct only %.3f)%s\n",
           100.0 * per_head[0] / std::max(1, per_head_n[0]), 100.0 * top3 / std::max(1, n),
           conf_sum / std::max(1, n), conf_right / std::max(1, per_head[0]),
           holdout ? "   (hold-out split)" : "");
    for (const auto& kv : by_cat)
      printf("  %-16s %5d crops  top1 %5.1f%%\n", kv.first.c_str(), kv.second[0],
             100.0 * kv.second[1] / std::max(1, kv.second[0]));
  } else {
    printf("whole plate %.1f%%", 100.0 * full_ok / std::max(1, n));
    if (legible_n) printf("   (legible only %.1f%% of %d)", 100.0 * legible_full / legible_n, legible_n);
    printf("\nper head:");
    for (size_t h = 0; h < nheads && h < 9; ++h)
      printf(" %s %.0f%%", heads[h].substr(0, 6).c_str(),
             100.0 * per_head[h] / std::max(1, per_head_n[h]));
    printf("\n");
  }
  return 0;
}

// jlpr pseudo-label — the C++ side of tools/pseudo_label.py: label real photos with an existing
// detector so the synthetic set gets some real backgrounds to sit on.
//
// Why it exists: our composites paste plates onto whatever we have, so a detector trained on them is
// strong at close-ups and weak at 5% of the frame, and the borrowed PlateYOLO-JP is the mirror image.
// It is a good teacher exactly where we are bad. The labels are only as good as the teacher, which is
// why the output is *mixed with* synthetic data rather than used alone.
static int cmd_pseudo_label(int argc, char** argv) {
  std::string src = arg_of(argc, argv, "--src", "");
  std::string out = arg_of(argc, argv, "--out", "");
  std::string det_p = arg_of(argc, argv, "--det", "models/plate_det_pyj320.onnx");
  int imgsz = std::atoi(arg_of(argc, argv, "--imgsz", "640").c_str());
  int limit = std::atoi(arg_of(argc, argv, "--limit", "0").c_str());
  float conf = (float)atof(arg_of(argc, argv, "--conf", "0.5").c_str());
  bool keep_empty = has_flag(argc, argv, "--keep-empty");
  if (src.empty() || out.empty()) {
    printf("usage: jlpr pseudo-label --src <dir of photos> --out <yolo dir> [--det onnx] "
           "[--fmt xyxy|cxcywh] [--imgsz N] [--conf f] [--limit N] [--keep-empty]\n");
    return 1;
  }
  jl::DetCfg cfg;
  cfg.kind = jl::DetKind::V8;
  cfg.nc = 1;
  cfg.plate_class = 0;
  cfg.imgsz = std::atoi(arg_of(argc, argv, "--det-imgsz", "0").c_str());
  cfg.conf = conf;
  cfg.nms = 0.45f;
  cfg.v8_fmt = (arg_of(argc, argv, "--fmt", "xyxy") == "cxcywh") ? BoxFmt::CXCYWH : BoxFmt::XYXY;

  std::vector<std::string> files;
  trn::list_images_recursive(src, files);
  std::sort(files.begin(), files.end());
  if (limit > 0 && (int)files.size() > limit) files.resize((size_t)limit);
  if (files.empty()) { printf("no images under %s\n", src.c_str()); return 1; }
  make_dir(out + "/images");
  make_dir(out + "/labels");
  onx::Graph det = onx::load_onnx(det_p);

  int kept = 0, boxes_total = 0, empty = 0;
  for (size_t n = 0; n < files.size(); ++n) {
    int W = 0, H = 0, C = 0;
    std::vector<unsigned char> blob = trn::read_file(files[n]);
    unsigned char* im = blob.empty() ? nullptr
                                     : stbi_load_from_memory(blob.data(), (int)blob.size(), &W, &H, &C, 3);
    if (!im) continue;
    std::vector<jl::Box> boxes = jl::detect_plates(det, im, W, H, cfg);
    if (boxes.empty() && !keep_empty) { ++empty; stbi_image_free(im); continue; }
    char name[64];
    snprintf(name, sizeof name, "real%05d", kept);
    // The frame is written at --imgsz so the normalised labels stay valid whatever the source size.
    Tensor sized = jl::resize_rgb01(im, W, H, imgsz, imgsz);
    stbi_image_free(im);
    std::vector<unsigned char> pix((size_t)imgsz * imgsz * 3);
    for (int y = 0; y < imgsz; ++y)
      for (int x = 0; x < imgsz; ++x)
        for (int c = 0; c < 3; ++c)
          pix[((size_t)y * imgsz + x) * 3 + c] = (unsigned char)std::lround(
              255.f * std::min(1.f, std::max(0.f, sized->data[((size_t)c * imgsz + y) * imgsz + x])));
    stbi_write_png((out + "/images/" + name + ".png").c_str(), imgsz, imgsz, 3, pix.data(), imgsz * 3);
    std::string lines;
    for (const jl::Box& b : boxes) {
      char buf[128];
      snprintf(buf, sizeof buf, "0 %.6f %.6f %.6f %.6f\n", (b.x1 + b.x2) * 0.5f / W,
               (b.y1 + b.y2) * 0.5f / H, (b.x2 - b.x1) / W, (b.y2 - b.y1) / H);
      lines += buf;
      ++boxes_total;
    }
    trn::write_file(out + "/labels/" + name + ".txt", lines.data(), lines.size());
    ++kept;
    if ((n + 1) % 100 == 0) {
      printf("  %zu/%zu  kept %d  boxes %d  empty %d\n", n + 1, files.size(), kept, boxes_total, empty);
      fflush(stdout);
    }
  }
  printf("\nwrote %d images and %d boxes into %s (skipped %d with no confident detection)\n", kept,
         boxes_total, out.c_str(), empty);
  printf("teacher: %s at conf>=%.2f — these labels are only as good as that model, which is why they\n"
         "are mixed with synthetic data rather than used alone.\n", det_p.c_str(), conf);
  return 0;
}

// jlpr check-regions — the C++ side of tools/check_regions.py: can the recogniser name every region
// on the list at all? Renders come from `jlpr gen --region sweep`, so this is an easy test (same
// fonts as training); it answers "is this class reachable", not "how good is it in the wild".
static int cmd_check_regions(int argc, char** argv) {
  std::string data = arg_of(argc, argv, "--data", "");
  std::string ocr_p = arg_of(argc, argv, "--ocr", "models/plate_ocr_v2.onnx");
  std::string spec_p = arg_of(argc, argv, "--spec", "spec/labels.txt");
  std::string alpr = arg_of(argc, argv, "--alpr", "");    // which names real data covers
  int limit = std::atoi(arg_of(argc, argv, "--limit", "0").c_str());
  bool tta = has_flag(argc, argv, "--tta");
  if (data.empty()) {
    printf("usage: jlpr check-regions --data <dir from `jlpr gen --region sweep`> [--ocr onnx] "
           "[--alpr <root>] [--tta] [--limit N]\n");
    return 1;
  }
  spec::Spec sp = spec::load(spec_p);
  onx::Graph g = onx::load_onnx(ocr_p);
  std::vector<trn::Item> items = trn::read_synth(data, true);
  if (items.empty()) { printf("no labelled crops under %s\n", data.c_str()); return 1; }
  if (limit > 0 && (int)items.size() > limit) items.resize((size_t)limit);

  std::set<int> covered;                                  // region ids real data has examples of
  if (!alpr.empty())
    for (const trn::Item& it : trn::read_alpr(alpr, sp)) covered.insert(it.heads[0]);

  const spec::Group& reg = sp.head("region");
  std::vector<int> seen(reg.n, 0), ok(reg.n, 0);
  const std::vector<float> one = {0.03f};
  for (const trn::Item& it : items) {
    int W = 0, H = 0, C = 0;
    std::vector<unsigned char> blob = trn::read_file(it.path);
    unsigned char* im = blob.empty() ? nullptr
                                     : stbi_load_from_memory(blob.data(), (int)blob.size(), &W, &H, &C, 3);
    if (!im) continue;
    float x0 = 0, y0 = 0, x1 = (float)W, y1 = (float)H;
    if (it.have_box) { x0 = it.bx0; y0 = it.by0; x1 = it.bx1; y1 = it.by1; }
    jl::Read rd = jl::read_plate(g, sp, im, W, H, x0, y0, x1, y1,
                                 tta ? jl::default_margins() : one);
    stbi_image_free(im);
    const int want = it.heads[0];
    if (want < 0 || want >= reg.n) continue;
    ++seen[(size_t)want];
    if (rd.arg[0] == want) ++ok[(size_t)want];
  }
  int tot = 0, hit = 0, new_tot = 0, new_hit = 0, cov_tot = 0, cov_hit = 0, unc_tot = 0, unc_hit = 0;
  std::vector<std::string> misses;
  for (int r = 0; r < reg.n; ++r) {
    if (!seen[(size_t)r]) continue;
    tot += seen[(size_t)r];
    hit += ok[(size_t)r];
    // the five names added in 2025 sit at the end of spec/labels.txt (133..137 in the 138-name list)
    const bool is_new = r >= reg.n - 5;
    if (is_new) { new_tot += seen[(size_t)r]; new_hit += ok[(size_t)r]; }
    if (!alpr.empty()) {
      if (covered.count(r)) { cov_tot += seen[(size_t)r]; cov_hit += ok[(size_t)r]; }
      else { unc_tot += seen[(size_t)r]; unc_hit += ok[(size_t)r]; }
    }
    if (!ok[(size_t)r]) misses.push_back(reg.tok[(size_t)r]);
  }
  printf("\n%s on %d renders of %d region names (%s)\n", ocr_p.c_str(), tot,
         (int)std::count_if(seen.begin(), seen.end(), [](int v) { return v > 0; }),
         tta ? "6-crop TTA" : "1 crop");
  printf("readable overall: %.1f%%\n", 100.0 * hit / std::max(1, tot));
  if (new_tot) printf("  2025 additions (last 5 names): %d/%d\n", new_hit, new_tot);
  if (!alpr.empty()) {
    printf("  names real data covers   : %.1f%% of %d\n", 100.0 * cov_hit / std::max(1, cov_tot), cov_tot);
    printf("  names real data lacks    : %.1f%% of %d\n", 100.0 * unc_hit / std::max(1, unc_tot), unc_tot);
  }
  if (!misses.empty()) {
    printf("never read correctly (%zu):", misses.size());
    for (size_t i = 0; i < misses.size() && i < 40; ++i) printf(" %s", misses[i].c_str());
    printf("%s\n", misses.size() > 40 ? " ..." : "");
  }
  return 0;
}

// jlpr context-test — the C++ side of tools/context_test.py: the same photo, only the size of the
// window cut around the plate changes. This is the measurement that showed the borrowed detectors
// only fire on plates that come with a car around them (0.83 at 8% of the frame, nothing past 30%),
// and it is how a new detector's usable range gets stated in numbers instead of adjectives.
static int cmd_context_test(int argc, char** argv) {
  std::string img = arg_of(argc, argv, "--img", "");
  std::string det_p = arg_of(argc, argv, "--det", "models/plate_det_v8n_320.onnx");
  std::string shares_s = arg_of(argc, argv, "--shares", "0.05,0.08,0.12,0.20,0.30,0.45,0.60,0.80,0.95");
  float bx0 = (float)atof(arg_of(argc, argv, "--x1", "-1").c_str());
  float by0 = (float)atof(arg_of(argc, argv, "--y1", "-1").c_str());
  float bx1 = (float)atof(arg_of(argc, argv, "--x2", "-1").c_str());
  float by1 = (float)atof(arg_of(argc, argv, "--y2", "-1").c_str());
  float conf = (float)atof(arg_of(argc, argv, "--conf", "0.10").c_str());
  if (img.empty()) {
    printf("usage: jlpr context-test --img <file> [--det onnx] [--fmt xyxy|cxcywh] "
           "[--x1 f --y1 f --x2 f --y2 f] [--shares a,b,c] [--conf f]\n"
           "  without --x1..--y2 the plate is located by running the detector on the whole frame\n");
    return 1;
  }
  jl::DetCfg cfg;
  cfg.kind = jl::DetKind::V8;
  cfg.nc = 1;
  cfg.plate_class = 0;
  cfg.imgsz = std::atoi(arg_of(argc, argv, "--imgsz", "0").c_str());
  cfg.conf = conf;
  cfg.nms = 0.45f;
  // Box layout is declared, not sniffed — the trap this tool itself fell into once: reading cxcywh
  // as xyxy puts a plausible-looking box in the wrong place and the run reports "no detections".
  cfg.v8_fmt = (arg_of(argc, argv, "--fmt", "cxcywh") == "xyxy") ? BoxFmt::XYXY : BoxFmt::CXCYWH;

  int W = 0, H = 0, C = 0;
  std::vector<unsigned char> blob = trn::read_file(img);
  unsigned char* im = blob.empty() ? nullptr
                                   : stbi_load_from_memory(blob.data(), (int)blob.size(), &W, &H, &C, 3);
  if (!im) { printf("cannot read %s\n", img.c_str()); return 1; }
  onx::Graph det = onx::load_onnx(det_p);
  if (bx0 < 0) {
    std::vector<jl::Box> b = jl::detect_plates(det, im, W, H, cfg);
    if (b.empty()) { printf("no plate found in the full frame; pass --x1..--y2 to say where it is\n"); return 1; }
    bx0 = b[0].x1; by0 = b[0].y1; bx1 = b[0].x2; by1 = b[0].y2;
    printf("plate located at (%.0f,%.0f)-(%.0f,%.0f), det %.2f\n", bx0, by0, bx1, by1, b[0].score);
  }
  const float pcx = (bx0 + bx1) / 2, pcy = (by0 + by1) / 2, pw = bx1 - bx0;
  std::vector<float> shares;
  {
    std::stringstream ss(shares_s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) shares.push_back((float)atof(tok.c_str()));
  }
  // The window is 4:3 and is always resampled to 640x480, exactly as tools/context_test.py does:
  // the detector must see the same framing at every share, or the sweep measures the resize instead
  // of the context.
  const int out_w = std::atoi(arg_of(argc, argv, "--out-w", "640").c_str());
  const int out_h = (int)std::lround(out_w * 3.0 / 4.0);
  printf("%-12s %-14s %8s %6s\n", "plate share", "plate px @out", "det", "IoU");
  for (float sh : shares) {
    const float win_w = pw / std::max(0.01f, sh), win_h = win_w * 3.f / 4.f;
    const float x0 = pcx - win_w / 2, y0 = pcy - win_h / 2;
    std::vector<unsigned char> crop((size_t)out_w * out_h * 3, 114);
    for (int y = 0; y < out_h; ++y)
      for (int x = 0; x < out_w; ++x) {
        const float sx = x0 + (x + 0.5f) * (win_w / out_w) - 0.5f;
        const float sy = y0 + (y + 0.5f) * (win_h / out_h) - 0.5f;
        for (int c = 0; c < 3; ++c)
          crop[((size_t)y * out_w + x) * 3 + c] =
              (unsigned char)std::lround(std::min(255.f, std::max(0.f, det::bl_sample(im, W, H, sx, sy, c))));
      }
    std::vector<jl::Box> got = jl::detect_plates(det, crop.data(), out_w, out_h, cfg);
    const float sx = out_w / win_w, sy = out_h / win_h;
    evd::GtBox want{(bx0 - x0) * sx, (by0 - y0) * sy, (bx1 - x0) * sx, (by1 - y0) * sy, sh};
    float best = 0, best_iou = 0;
    for (const jl::Box& d : got) {
      const float v = evd::iou_xyxy(want, {d.x1, d.y1, d.x2, d.y2, d.score});
      if (v > best_iou) { best_iou = v; best = d.score; }
    }
    char det_s[16];
    if (best_iou > 0.3f) snprintf(det_s, sizeof det_s, "%.2f", best);
    else snprintf(det_s, sizeof det_s, "%s", "none");
    printf("%-11.0f%% %-14.0f %8s %6.2f\n", sh * 100, want.x2 - want.x1, det_s, best_iou);
  }
  stbi_image_free(im);
  return 0;
}

// jlpr strip-nms — the C++ side of tools/strip_nms.py: cut an Ultralytics-style detector ONNX just
// before its NMS tail. An export made with nms=True ends in NonMaxSuppression / GatherND / ScatterND
// / NonZero / Where, which a small hand-written interpreter has no business implementing and does not
// need to: the tensor feeding the NMS already holds pixel-space boxes and sigmoided scores, and
// pure/infer_v8.hpp does the threshold + greedy NMS in 40 lines.
static int cmd_strip_nms(int argc, char** argv) {
  std::string src = arg_of(argc, argv, "--in", "");
  std::string dst = arg_of(argc, argv, "--out", "");
  std::string want = arg_of(argc, argv, "--out-tensor", "");
  if (src.empty() || dst.empty()) {
    printf("usage: jlpr strip-nms --in <onnx> --out <onnx> [--out-tensor NAME]\n"
           "  with no --out-tensor the last Concat before the tail is used\n");
    return 1;
  }
  onx::Graph g = onx::load_onnx(src);
  if (want.empty()) {
    for (const onx::Node& n : g.nodes)
      if (n.op_type == "Concat" && !n.output.empty()) want = n.output[0];
    if (want.empty()) { printf("%s has no Concat to cut at; pass --out-tensor\n", src.c_str()); return 1; }
  }
  // keep exactly the nodes that produce `want`, and only the initializers those nodes read
  std::map<std::string, const onx::Node*> prod;
  for (const onx::Node& n : g.nodes)
    for (const std::string& o : n.output) prod[o] = &n;
  if (!prod.count(want)) { printf("%s: no node produces '%s'\n", src.c_str(), want.c_str()); return 1; }
  std::set<const onx::Node*> keep;
  std::set<std::string> used;
  std::vector<std::string> stack{want};
  while (!stack.empty()) {
    const std::string name = stack.back();
    stack.pop_back();
    used.insert(name);
    auto it = prod.find(name);
    if (it == prod.end() || keep.count(it->second)) continue;
    keep.insert(it->second);
    for (const std::string& in : it->second->input) if (!in.empty()) stack.push_back(in);
  }
  onx::Graph out;
  out.opset = g.opset;
  out.inputs = g.inputs;
  for (const onx::Node& n : g.nodes) if (keep.count(&n)) out.nodes.push_back(n);
  for (const onx::Tensor64& t : g.init_f) if (used.count(t.name)) out.init_f.push_back(t);
  for (const onx::IntsTensor& t : g.init_i) if (used.count(t.name)) out.init_i.push_back(t);
  out.outputs.push_back({want, {}});
  onx::save_onnx(out, dst);
  std::map<std::string, int> ops;
  for (const onx::Node& n : out.nodes) ops[n.op_type]++;
  printf("cut at %s: %zu -> %zu nodes, %zu float initializers\n", want.c_str(), g.nodes.size(),
         out.nodes.size(), out.init_f.size());
  printf("ops:");
  for (const auto& kv : ops) printf(" %s(%d)", kv.first.c_str(), kv.second);
  printf("\nwrote %s\n", dst.c_str());
  return 0;
}

// jlpr recolor-test — the C++ side of tools/recolor_test.py: the same photo and the same geometry,
// with only the plate's base and text colours swapped for the four Japanese schemes. It exists
// because "the detector is weak on black plates" was a claim that needed a measurement; the answer
// (this method does NOT reproduce a colour bias, and the recogniser reads all four) is in RESUME.
static int cmd_recolor_test(int argc, char** argv) {
  std::string img = arg_of(argc, argv, "--img", "");
  std::string det_p = arg_of(argc, argv, "--det", "models/plate_det_v8n_320.onnx");
  std::string ocr_p = arg_of(argc, argv, "--ocr", "models/plate_ocr_v7_bal.onnx");
  std::string spec_p = arg_of(argc, argv, "--spec", "spec/labels.txt");
  std::string save = arg_of(argc, argv, "--save-dir", "");
  float bx0 = (float)atof(arg_of(argc, argv, "--x1", "-1").c_str());
  float by0 = (float)atof(arg_of(argc, argv, "--y1", "-1").c_str());
  float bx1 = (float)atof(arg_of(argc, argv, "--x2", "-1").c_str());
  float by1 = (float)atof(arg_of(argc, argv, "--y2", "-1").c_str());
  float conf = (float)atof(arg_of(argc, argv, "--conf", "0.05").c_str());
  if (img.empty()) {
    printf("usage: jlpr recolor-test --img <file> [--det onnx] [--ocr onnx] [--fmt xyxy|cxcywh] "
           "[--x1 f --y1 f --x2 f --y2 f] [--conf f] [--save-dir d]\n");
    return 1;
  }
  jl::DetCfg cfg;
  cfg.kind = jl::DetKind::V8;
  cfg.nc = 1;
  cfg.plate_class = 0;
  cfg.imgsz = std::atoi(arg_of(argc, argv, "--imgsz", "0").c_str());
  cfg.conf = conf;
  cfg.nms = 0.45f;
  cfg.v8_fmt = (arg_of(argc, argv, "--fmt", "cxcywh") == "xyxy") ? BoxFmt::XYXY : BoxFmt::CXCYWH;

  int W = 0, H = 0, C = 0;
  std::vector<unsigned char> blob = trn::read_file(img);
  unsigned char* im = blob.empty() ? nullptr
                                   : stbi_load_from_memory(blob.data(), (int)blob.size(), &W, &H, &C, 3);
  if (!im) { printf("cannot read %s\n", img.c_str()); return 1; }
  onx::Graph det = onx::load_onnx(det_p);
  onx::Graph ocr = onx::load_onnx(ocr_p);
  spec::Spec sp = spec::load(spec_p);
  if (bx0 < 0) {
    jl::DetCfg find = cfg;
    find.conf = 0.25f;
    std::vector<jl::Box> b = jl::detect_plates(det, im, W, H, find);
    if (b.empty()) { printf("no plate on the original; pass --x1..--y2\n"); return 1; }
    bx0 = b[0].x1; by0 = b[0].y1; bx1 = b[0].x2; by1 = b[0].y2;
    printf("plate box from the original: (%.0f,%.0f)-(%.0f,%.0f)\n", bx0, by0, bx1, by1);
  }
  struct Pal { const char* name; unsigned char bg[3], fg[3]; };
  const Pal pals[4] = {
    {"\xE7\x99\xBD\xE5\x9C\xB0\xE7\xB7\x91\xE5\xAD\x97 (\xE8\x87\xAA\xE5\xAE\xB6\xE7\x94\xA8)",   {250, 250, 248}, {16, 90, 60}},
    {"\xE7\xB7\x91\xE5\x9C\xB0\xE7\x99\xBD\xE5\xAD\x97 (\xE4\xBA\x8B\xE6\xA5\xAD\xE7\x94\xA8)",   {16, 90, 60},    {250, 250, 248}},
    {"\xE9\xBB\x84\xE5\x9C\xB0\xE9\xBB\x92\xE5\xAD\x97 (\xE8\xBB\xBD\xE8\x87\xAA\xE5\xAE\xB6\xE7\x94\xA8)", {240, 205, 20},  {25, 25, 25}},
    {"\xE9\xBB\x92\xE5\x9C\xB0\xE9\xBB\x84\xE5\xAD\x97 (\xE8\xBB\xBD\xE4\xBA\x8B\xE6\xA5\xAD\xE7\x94\xA8)", {25, 25, 25},    {240, 205, 20}},
  };
  const int x1 = std::max(0, (int)std::lround(bx0)), y1 = std::max(0, (int)std::lround(by0));
  const int x2 = std::min(W, (int)std::lround(bx1)), y2 = std::min(H, (int)std::lround(by1));
  if (x2 <= x1 || y2 <= y1) { printf("the box is empty after clipping\n"); return 1; }
  // luminance inside the plate, min-max normalised on the 2nd/98th percentile (shading and bleed are
  // preserved: only the two endpoint colours change)
  std::vector<float> lum((size_t)(x2 - x1) * (y2 - y1));
  for (int y = y1; y < y2; ++y)
    for (int x = x1; x < x2; ++x) {
      const unsigned char* p = im + ((size_t)y * W + x) * 3;
      lum[(size_t)(y - y1) * (x2 - x1) + (x - x1)] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    }
  std::vector<float> srt = lum;
  std::sort(srt.begin(), srt.end());
  const float lo = srt[(size_t)(0.02 * (srt.size() - 1))], hi = srt[(size_t)(0.98 * (srt.size() - 1))];

  printf("\n%-28s %8s   %s\n", "plate colours", "det", "reading (given box)");
  for (const Pal& pal : pals) {
    std::vector<unsigned char> rgb(im, im + (size_t)W * H * 3);
    for (int y = y1; y < y2; ++y)
      for (int x = x1; x < x2; ++x) {
        const float t = std::min(1.f, std::max(0.f, (lum[(size_t)(y - y1) * (x2 - x1) + (x - x1)] - lo) /
                                                        std::max(1e-6f, hi - lo)));
        for (int c = 0; c < 3; ++c)
          rgb[((size_t)y * W + x) * 3 + c] =
              (unsigned char)std::lround(pal.fg[c] + (pal.bg[c] - pal.fg[c]) * t);
      }
    std::vector<jl::Box> got = jl::detect_plates(det, rgb.data(), W, H, cfg);
    jl::Read rd = jl::read_plate_tta(ocr, sp, rgb.data(), W, H, bx0, by0, bx1, by1);
    char det_s[16];
    if (!got.empty()) snprintf(det_s, sizeof det_s, "%.3f", got[0].score);
    else snprintf(det_s, sizeof det_s, "%s", "none");
    printf("%-28s %8s   %s (region %.2f)\n", pal.name, det_s, rd.plate.text.c_str(), rd.conf[0]);
    if (!save.empty()) {
      make_dir(save);
      static int k = 0;
      char path[512];
      snprintf(path, sizeof path, "%s/recolor%d.png", save.c_str(), k++);
      stbi_write_png(path, W, H, 3, rgb.data(), W * 3);
    }
  }
  stbi_image_free(im);
  return 0;
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
    if (!trn::write_file(hdr, s.data(), s.size())) {   // UTF-8 path safe: the parity test writes into
      printf("cannot write %s\n", hdr.c_str());        // a tempdir, which here sits under 大谷陽明/
      return 1;
    }
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
  std::string det_p = arg_of(argc, argv, "--det", "models/plate_det_pyj320.onnx");
  std::string ocr_p = arg_of(argc, argv, "--ocr", "models/plate_ocr_v2.onnx");
  std::string corner_p = arg_of(argc, argv, "--corner", "");   // empty = box crop + margin TTA
  std::string spec_p = arg_of(argc, argv, "--spec", "spec/labels.txt");
  std::string outp = arg_of(argc, argv, "--out", "");
  bool single = has_flag(argc, argv, "--single");
  bool as_json = has_flag(argc, argv, "--json");   // machine-readable, same shape as the WASM/Python output
  std::string kind = arg_of(argc, argv, "--det-kind", "v8");
  jl::DetCfg cfg;
  cfg.conf = (float)atof(arg_of(argc, argv, "--conf", "0.30").c_str());
  if (kind == "v8") {          // plate-only YOLOv8/v11/v12 head, size taken from the graph
    cfg.kind = jl::DetKind::V8;
    cfg.nc = 1;
    cfg.plate_class = 0;
    cfg.imgsz = std::atoi(arg_of(argc, argv, "--imgsz", "0").c_str());
    // Box layout is declared, not sniffed: PlateYOLO-JP cut before its NMS gives xyxy, a plain
    // Ultralytics export gives cxcywh. Guessing wrong yields plausible boxes in the wrong place.
    cfg.v8_fmt = (arg_of(argc, argv, "--fmt", "xyxy") == "cxcywh") ? BoxFmt::CXCYWH : BoxFmt::XYXY;
  } else {
    cfg.imgsz = std::atoi(arg_of(argc, argv, "--imgsz", "416").c_str());
  }
  if (img.empty()) {
    printf("usage: jlpr detect --img <file> [--det onnx] [--ocr onnx] [--out png] [--corner onnx] [--det-kind v8|yolox] [--fmt xyxy|cxcywh] [--conf f] [--single] [--json]\n");
    return 1;
  }

  int W = 0, H = 0, C = 0;
  unsigned char* im = stbi_load(img.c_str(), &W, &H, &C, 3);
  if (!im) { printf("cannot load %s\n", img.c_str()); return 1; }
  spec::Spec sp = spec::load(spec_p);
  onx::Graph det = onx::load_onnx(det_p);
  onx::Graph ocr = onx::load_onnx(ocr_p);
  onx::Graph corner;
  bool have_corner = false;
  if (!corner_p.empty()) { corner = onx::load_onnx(corner_p); have_corner = !corner.nodes.empty(); }
  if (!as_json) printf("%s %dx%d   det=%s ocr=%s\n", img.c_str(), W, H, det_p.c_str(), ocr_p.c_str());

  std::vector<Det> all;
  std::vector<jl::Box> boxes = jl::detect_plates(det, im, W, H, cfg, &all);
  std::vector<jl::Read> reads;
  if (!as_json)
    printf("plates: %zu (conf>=%.2f, %zu raw detections over all classes)\n", boxes.size(), cfg.conf, all.size());

  for (size_t i = 0; i < boxes.size(); ++i) {
    const jl::Box& b = boxes[i];
    jl::Read r;
    float corners[8];
    jl::CornerCfg ccfg;
    if (have_corner && jl::predict_corners(corner, im, W, H, b, ccfg, corners)) {
      r = jl::read_plate_warped(ocr, sp, im, W, H, corners, ccfg);   // rectified: 1 forward pass
    } else {
      r = single ? jl::read_plate_single(ocr, sp, im, W, H, b.x1, b.y1, b.x2, b.y2)
                 : jl::read_plate_tta(ocr, sp, im, W, H, b.x1, b.y1, b.x2, b.y2);
    }
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
  // Windows hands main() ANSI-converted arguments, so a path like ../_alpr/自家用 arrives as CP932
  // bytes and every UTF-8 conversion downstream turns it into garbage ("real plate pool: 0 files").
  // Take the real command line instead and re-encode it as UTF-8, which is what the rest of this
  // program assumes everywhere.
  std::vector<std::string> utf8_args;
  std::vector<char*> utf8_argv;
  {
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv) {
      for (int i = 0; i < wargc; ++i) utf8_args.push_back(trn::from_w(wargv[i]));
      LocalFree(wargv);
      for (std::string& a : utf8_args) utf8_argv.push_back(a.data());
      argc = (int)utf8_argv.size();
      argv = utf8_argv.data();
    }
  }
#endif
  if (argc < 2) {
    printf("jlpr — Japanese license plate pipeline (C++)\n"
           "  jlpr labels     [--dump|--emit-header out.hpp]\n"
           "  jlpr export     --ocr <ref_dir> --out <onnx>\n"
           "  jlpr parity-ocr --ocr <onnx> --ref <ref_dir>\n"
           "  jlpr detect     --img <file> [--det onnx] [--ocr onnx] [--out png] [--corner onnx] [--det-kind v8|yolox] [--fmt xyxy|cxcywh] [--conf f] [--single] [--json]\n"
           "  jlpr rgba       --img <file> --out <file.rgba>\n"
           "  jlpr gen|gen-det [--out dir] [--count N] [--imgsz S]\n"
           "  jlpr train      --model ocr --synth <dir> --alpr <root> [--export onnx]\n"
           "  jlpr train      --model det --data <yolo dir> [--init onnx] [--steps N] [--batch N] [--lr f]\n"
           "                              [--limit N] [--export onnx] [--gradcheck] [--dump-fixture bin]\n"
           "  jlpr train      --model corner --synth <dir> [--init random|onnx] [--width N] [--export onnx]\n"
           "  jlpr val        --data <dir> [--kind alpr|synth] [--holdout]\n"
           "  jlpr val        --model det --data <yolo dir> [--det onnx] [--fmt xyxy|cxcywh] [--conf f]\n"
           "  jlpr pseudo-label --src <dir> --out <yolo dir> [--det onnx] [--conf f]\n"
           "  jlpr check-regions --data <region sweep dir> [--ocr onnx] [--alpr root] [--tta]\n"
           "  jlpr context-test  --img <file> [--det onnx] [--shares a,b,c]\n"
           "  jlpr recolor-test  --img <file> [--det onnx] [--ocr onnx]\n"
           "  jlpr strip-nms     --in <onnx> --out <onnx> [--out-tensor NAME]\n");
    return 1;
  }
  std::string cmd = argv[1];
  if (cmd == "labels") return cmd_labels(argc, argv);
  if (cmd == "export") return cmd_export(argc, argv);
  if (cmd == "parity-ocr") return cmd_parity_ocr(argc, argv);
  if (cmd == "detect") return cmd_detect(argc, argv);
  if (cmd == "rgba") return cmd_rgba(argc, argv);
  if (cmd == "gen") return cmd_gen(argc, argv);
  if (cmd == "gen-det") return cmd_gen_det(argc, argv);
  if (cmd == "train") return cmd_train(argc, argv);
  if (cmd == "val") return cmd_val(argc, argv);
  if (cmd == "pseudo-label") return cmd_pseudo_label(argc, argv);
  if (cmd == "check-regions") return cmd_check_regions(argc, argv);
  if (cmd == "context-test") return cmd_context_test(argc, argv);
  if (cmd == "strip-nms") return cmd_strip_nms(argc, argv);
  if (cmd == "recolor-test") return cmd_recolor_test(argc, argv);
  printf("jlpr: '%s' is not implemented yet\n", cmd.c_str());
  return 1;
}
