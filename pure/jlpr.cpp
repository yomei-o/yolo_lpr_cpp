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
#include <filesystem>
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
  bool tex_dump = has_flag(argc, argv, "--tex-dump");   // also write the flat plate art (debug)
  bool quiet = has_flag(argc, argv, "--quiet");

  spec::Spec sp = spec::load(spec_path);
  // The font list is drawn from even in --meta-only mode, so the rng stream (and therefore the
  // meta dump) is identical whether or not images are rendered.
  std::string only_font = arg_of(argc, argv, "--font", "");
  std::vector<std::string> font_paths = gen::font_files(font_dir, only_font);
  std::vector<std::string> font_names;
  for (const std::string& f : font_paths) font_names.push_back(f.substr(f.find_last_of("/\\") + 1));
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
  std::vector<std::string> font_paths = gen::font_files(font_dir, only_font);
  if (font_paths.empty()) { printf("no fonts in %s\n", font_dir.c_str()); return 1; }

  // Background pool: real photos make the negatives and the context far more honest than a gradient.
  // Sorted so the index draw means the same file in both languages.
  std::vector<std::string> bgs;
  if (!bg_dir.empty()) {
    std::error_code ec;
    for (auto& e : std::filesystem::directory_iterator(bg_dir, ec)) {
      if (!e.is_regular_file()) continue;
      std::string x = e.path().string();
      std::string low = x;
      for (char& c : low) c = (char)tolower(c);
      if (low.size() > 4 && (low.rfind(".jpg") == low.size() - 4 || low.rfind(".png") == low.size() - 4 ||
                             low.rfind(".jpeg") == low.size() - 5))
        bgs.push_back(x);
    }
    std::sort(bgs.begin(), bgs.end());
    if (!quiet) printf("background pool: %zu files from %s\n", bgs.size(), bg_dir.c_str());
  }

  // Real plate photos to paste (searched recursively): the detector needs the actual typeface,
  // lighting and dirt, not our drawing of them. Trained on drawn plates only it reached mAP50 0.99
  // on synthetic frames and still scored 0.1 on the real bus photo. Sorted, so the index draw picks
  // the same file in both languages.
  std::vector<std::string> reals;
  if (!real_dir.empty()) {
    std::error_code ec2;
    for (auto& e : std::filesystem::recursive_directory_iterator(real_dir, ec2)) {
      if (!e.is_regular_file()) continue;
      std::string x = e.path().string();
      std::string low = x;
      for (char& c : low) c = (char)tolower(c);
      if (low.size() > 4 && (low.rfind(".jpg") == low.size() - 4 || low.rfind(".png") == low.size() - 4 ||
                             low.rfind(".jpeg") == low.size() - 5))
        reals.push_back(x);
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
      unsigned char* bi = stbi_load(bgs[(size_t)d.bg_idx].c_str(), &bw, &bh, &bc, 3);
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
        unsigned char* ri = stbi_load(reals[(size_t)pl.real_idx % reals.size()].c_str(), &rw, &rh, &rc, 3);
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


// jlpr train --model ocr — fine-tune the recognizer ONNX in place (see pure/train_ocr.hpp).
static int cmd_train(int argc, char** argv) {
  std::string model = arg_of(argc, argv, "--model", "ocr");
  if (model != "ocr") { printf("jlpr train: only --model ocr is implemented (M5)\n"); return 1; }
  std::string onnx_in = arg_of(argc, argv, "--init", "models/plate_ocr.onnx");
  std::string spec_path = arg_of(argc, argv, "--spec", "spec/labels.txt");
  std::string synth = arg_of(argc, argv, "--synth", "");
  std::string alpr = arg_of(argc, argv, "--alpr", "");
  std::string out = arg_of(argc, argv, "--export", "");
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
  onx::Graph g = onx::load_onnx(onnx_in);
  onx::Trainable t = onx::make_trainable(g);
  if (!dump_loss)
    printf("%s: %zu trainable tensors, %zu parameters, %zu heads\n", onnx_in.c_str(),
           t.params.size(), onx::param_count(t), t.heads.size());

  std::vector<trn::Item> synth_items, alpr_items;
  std::vector<int> alpr_train, alpr_val;
  if (!synth.empty()) synth_items = trn::read_synth(synth, synth_region);
  if (!alpr.empty()) {
    alpr_items = trn::read_alpr(alpr, sp);
    trn::split_holdout(alpr_items.size(), 0.2, alpr_train, alpr_val);
  }
  if (synth_items.empty() && alpr_items.empty()) {
    printf("no data: pass --synth <dir> and/or --alpr <root>\n");
    return 1;
  }
  if (!dump_loss)
    printf("synthetic %zu crops%s, real %zu crops (%zu train / %zu hold-out)\n", synth_items.size(),
           synth_region ? " (region taught)" : " (region masked)", alpr_items.size(),
           alpr_train.size(), alpr_val.size());

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
      printf("  eval @%d: real region %.1f%% (%d)\n", step, 100 * acc, n);
    }
  }
  if (!alpr_val.empty() && !dump_loss) {
    int n = 0;
    double acc = trn::eval_region(t, alpr_items, alpr_val, region_head, 0, &n);
    printf("final: real hold-out region top1 %.1f%% over %d crops\n", 100 * acc, n);
  }
  if (!out.empty()) {
    onx::write_back(t);
    onx::save_onnx(t.g, out);
    printf("wrote %s\n", out.c_str());
  }
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
#endif
  if (argc < 2) {
    printf("jlpr — Japanese license plate pipeline (C++)\n"
           "  jlpr labels     [--dump|--emit-header out.hpp]\n"
           "  jlpr export     --ocr <ref_dir> --out <onnx>\n"
           "  jlpr parity-ocr --ocr <onnx> --ref <ref_dir>\n"
           "  jlpr detect     --img <file> [--det onnx] [--ocr onnx] [--out png] [--corner onnx] [--det-kind v8|yolox] [--fmt xyxy|cxcywh] [--conf f] [--single] [--json]\n"
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
  if (cmd == "gen") return cmd_gen(argc, argv);
  if (cmd == "gen-det") return cmd_gen_det(argc, argv);
  if (cmd == "train") return cmd_train(argc, argv);
  printf("jlpr: '%s' is not implemented yet\n", cmd.c_str());
  return 1;
}
