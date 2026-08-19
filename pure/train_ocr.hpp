// `jlpr train --model ocr` — the C++ half of M5.
//
// Same recipe as tools/train_ocr.py, and deliberately the same *order of random draws*, so
// tools/parity/train.py can put the two on the same batch and compare the loss and the gradients.
// The model being trained is the ONNX file itself (pure/onnx_train.hpp), so nothing here restates
// the architecture.
//
// Data, mirroring the Python side:
//   * synthetic dirs (`jlpr gen`): all 11 heads labelled, but the region head is MASKED OUT unless
//     --synth-region is given — measured: synthetic glyphs transfer 72-92% on digits and only ~28%
//     on region, and letting them train region overwrites what real plates taught the weights.
//   * alpr_jp: real crops, region label from the folder name, every other head masked.
#pragma once
#include "onnx_train.hpp"
#include "crop.hpp"
#include "spec.hpp"
#include "rng.hpp"
// stb_image.h is intentionally not included here (its implementation block sits outside the include
// guard, so a second include in the same TU breaks the build) — the including .cpp pulls it in.
#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

namespace trn {

struct Item {
  std::string path;
  std::string key;             // canonical "<category>/<region>/<file>" — what the sort uses, so
                               // both languages get the same order (and the same hold-out split)
  std::vector<int> heads;      // 11 head indices (unlabelled entries are 0 and masked)
  std::vector<float> mask;     // 11 flags
  bool have_box = false;
  float bx0 = 0, by0 = 0, bx1 = 0, by1 = 0;
};

// Windows needs the wide API here: the dataset directories are Japanese (自家用/品川/...) and the
// ANSI functions read our UTF-8 bytes in the local code page, so FindFirstFileA and fopen simply do
// not find them. Everything below converts UTF-8 <-> UTF-16 explicitly.
#ifdef _WIN32
inline std::wstring to_w(const std::string& s) {
  if (s.empty()) return L"";
  int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
  std::wstring w((size_t)n, L' ');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
  return w;
}
inline std::string from_w(const std::wstring& w) {
  if (w.empty()) return "";
  int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
  std::string s((size_t)n, ' ');
  WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n, nullptr, nullptr);
  return s;
}
#endif

// Read a whole file, UTF-8 path safe. stb_image's stbi_load() goes through fopen(), which cannot
// open a Japanese path given as UTF-8 on Windows — so images are read here and decoded from memory.
inline std::vector<unsigned char> read_file(const std::string& path) {
  std::vector<unsigned char> out;
#ifdef _WIN32
  FILE* f = _wfopen(to_w(path).c_str(), L"rb");
#else
  FILE* f = fopen(path.c_str(), "rb");
#endif
  if (!f) return out;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n > 0) {
    out.resize((size_t)n);
    out.resize(fread(out.data(), 1, (size_t)n, f));
  }
  fclose(f);
  return out;
}

inline std::vector<std::string> list_dir(const std::string& dir, bool want_dirs) {
  std::vector<std::string> out;
#ifdef _WIN32
  WIN32_FIND_DATAW fd;
  HANDLE h = FindFirstFileW(to_w(dir + "/*").c_str(), &fd);
  if (h == INVALID_HANDLE_VALUE) return out;
  do {
    std::string n = from_w(fd.cFileName);
    if (n == "." || n == "..") continue;
    bool isdir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    if (isdir == want_dirs) out.push_back(n);
  } while (FindNextFileW(h, &fd));
  FindClose(h);
#else
  DIR* d = opendir(dir.c_str());
  if (!d) return out;
  while (dirent* e = readdir(d)) {
    std::string n = e->d_name;
    if (n == "." || n == "..") continue;
    bool isdir = e->d_type == DT_DIR;
    if (isdir == want_dirs) out.push_back(n);
  }
  closedir(d);
#endif
  std::sort(out.begin(), out.end());
  return out;
}

// Recursive image listing that survives Japanese paths. std::filesystem on mingw converts char
// paths through the ANSI code page and throws "Illegal byte sequence" on 自家用/ — so the whole
// project lists directories through list_dir() (wide API on Windows) instead.
inline void list_images_recursive(const std::string& dir, std::vector<std::string>& out, int depth = 0) {
  if (depth > 6) return;
  for (const std::string& f : list_dir(dir, false)) {
    std::string low = f;
    for (char& c : low) c = (char)tolower(c);
    if ((low.size() > 4 && (low.rfind(".jpg") == low.size() - 4 || low.rfind(".png") == low.size() - 4)) ||
        (low.size() > 5 && low.rfind(".jpeg") == low.size() - 5))
      out.push_back(dir + "/" + f);
  }
  for (const std::string& d : list_dir(dir, true)) list_images_recursive(dir + "/" + d, out, depth + 1);
}

// A generated directory: labels.txt (11 indices) + corners.txt (the true plate box).
// `uncovered` = region indices the real training split has no example of. For those classes the
// synthetic glyphs are the only signal there is, so the region loss stays enabled for them; for every
// other region it is masked (synthetic fonts transfer at ~28% and overwrite what real plates taught).
// Without this the 2025 additions (十勝/日光/江戸川/安曇野/南信州) can never be predicted at all —
// measured 4e-11 probability on the shipped model.
inline std::vector<Item> read_synth(const std::string& root, bool teach_region,
                                    const std::set<int>& uncovered = {}) {
  std::vector<Item> items;
  std::string R = root;
  if (!R.empty() && R.back() != '/' && R.back() != '\\') R += '/';
  std::map<std::string, std::array<float, 4>> boxes;
  {
    std::ifstream f(R + "corners.txt");
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream ss(line);
      std::string name;
      float v[8];
      if (!(ss >> name)) continue;
      bool ok = true;
      for (int i = 0; i < 8; ++i) if (!(ss >> v[i])) { ok = false; break; }
      if (!ok) continue;
      float x0 = v[0], x1 = v[0], y0 = v[1], y1 = v[1];
      for (int i = 1; i < 4; ++i) {
        x0 = std::min(x0, v[2 * i]); x1 = std::max(x1, v[2 * i]);
        y0 = std::min(y0, v[2 * i + 1]); y1 = std::max(y1, v[2 * i + 1]);
      }
      boxes[name] = {x0, y0, x1, y1};
    }
  }
  std::ifstream f(R + "labels.txt");
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    Item it;
    std::string name;
    if (!(ss >> name)) continue;
    it.heads.resize(11);
    bool ok = true;
    for (int i = 0; i < 11; ++i) if (!(ss >> it.heads[i])) { ok = false; break; }
    if (!ok) continue;
    it.path = R + name;
    it.mask.assign(11, 1.f);
    if (!teach_region && !uncovered.count(it.heads[0])) it.mask[0] = 0.f;
    auto b = boxes.find(name);
    if (b != boxes.end()) {
      it.have_box = true;
      it.bx0 = b->second[0]; it.by0 = b->second[1]; it.bx1 = b->second[2]; it.by1 = b->second[3];
    }
    items.push_back(std::move(it));
  }
  return items;
}

// alpr_jp: <root>/<category>/<地名>/*.png — the folder name is the only label.
inline std::vector<Item> read_alpr(const std::string& root, const spec::Spec& sp) {
  static const char* cats[] = {"\xE8\x87\xAA\xE5\xAE\xB6\xE7\x94\xA8",                         // 自家用
                               "\xE8\x87\xAA\xE5\xAE\xB6\xE7\x94\xA8(\xE8\xBD\xBB)",           // 自家用(軽)
                               "\xE4\xBA\x8B\xE6\xA5\xAD\xE7\x94\xA8",                         // 事業用
                               "\xE4\xBA\x8B\xE6\xA5\xAD\xE7\x94\xA8(\xE8\xBD\xBB)"};          // 事業用(軽)
  std::vector<Item> items;
  std::string R = root;
  if (!R.empty() && R.back() != '/' && R.back() != '\\') R += '/';
  for (const char* c : cats) {
    std::string cd = R + c;
    for (const std::string& region : list_dir(cd, true)) {
      int idx = sp.index_of("region", region);
      if (idx < 0) continue;
      for (const std::string& f : list_dir(cd + "/" + region, false)) {
        std::string low = f;
        for (char& ch : low) ch = (char)tolower(ch);
        if (low.size() < 5) continue;
        if (low.rfind(".png") != low.size() - 4 && low.rfind(".jpg") != low.size() - 4 &&
            low.rfind(".jpeg") != low.size() - 5) continue;
        Item it;
        it.path = cd + "/" + region + "/" + f;
        it.key = std::string(c) + "/" + region + "/" + f;
        it.heads.assign(11, 0);
        it.heads[0] = idx;
        it.mask.assign(11, 0.f);
        it.mask[0] = 1.f;                       // region only
        items.push_back(std::move(it));
      }
    }
  }
  std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.key < b.key; });
  return items;
}

// Deterministic hold-out split, same shuffle as the Python side (splitmix64, seed 11).
inline void split_holdout(size_t n, double hold_out, std::vector<int>& train, std::vector<int>& val) {
  std::vector<int> order((size_t)n);
  for (size_t i = 0; i < n; ++i) order[i] = (int)i;
  Rng rng(11);
  for (size_t i = n; i-- > 1;) {
    size_t j = (size_t)rng.below((uint64_t)i + 1);
    std::swap(order[i], order[j]);
  }
  size_t cut = (size_t)(n * (1.0 - hold_out));
  train.assign(order.begin(), order.begin() + cut);
  val.assign(order.begin() + cut, order.end());
  std::sort(train.begin(), train.end());
  std::sort(val.begin(), val.end());
}

// One 128x128 input from a crop file (same sampling as inference: box + margin -> square), with the
// photometric jitter the Python trainer applies to real crops (brightness, contrast).
inline std::vector<float> load_input(const Item& it, float margin, float bright = 1.f,
                                     float contrast = 1.f) {
  int W = 0, H = 0, C = 0;
  std::vector<unsigned char> blob = read_file(it.path);
  unsigned char* im = blob.empty() ? nullptr
      : stbi_load_from_memory(blob.data(), (int)blob.size(), &W, &H, &C, 3);
  if (!im) return std::vector<float>((size_t)3 * 128 * 128, 0.f);
  float x0 = 0, y0 = 0, x1 = (float)W, y1 = (float)H;
  if (it.have_box) { x0 = it.bx0; y0 = it.by0; x1 = it.bx1; y1 = it.by1; }
  std::vector<float> v = jl::crop_to_input(im, W, H, x0, y0, x1, y1, margin);
  stbi_image_free(im);
  if (bright != 1.f || contrast != 1.f)
    for (float& z : v) z = std::clamp(((z - 0.5f) * contrast + 0.5f) * bright, 0.f, 1.f);
  return v;
}

struct Batch {
  Tensor x;
  std::vector<std::vector<int>> labels;    // [head][batch]
  std::vector<std::vector<float>> mask;    // [head][batch]
};

// The draw order here is the contract with tools/train_ocr.py: per sample, first the set choice
// (unit()), then the index (below(n)), then the crop margin (range).
// `photo[s]` = does set s get the extra brightness/contrast draws (real crops do, synthetic ones
// already carry their own degradation)? The draw counts have to match tools/train_ocr.py exactly.
inline Batch make_batch(const std::vector<const std::vector<Item>*>& sets,
                        const std::vector<const std::vector<int>*>& idx_pools,
                        const std::vector<double>& weights,
                        const std::vector<std::pair<double, double>>& margins,
                        const std::vector<bool>& photo,
                        int batch, Rng& rng) {
  std::vector<float> xb((size_t)batch * 3 * 128 * 128);
  Batch b;
  b.labels.assign(11, std::vector<int>((size_t)batch, 0));
  b.mask.assign(11, std::vector<float>((size_t)batch, 0.f));
  double total = 0;
  for (double w : weights) total += w;
  for (int k = 0; k < batch; ++k) {
    double r = rng.unit() * total, acc = 0;
    size_t pick = 0;
    for (size_t s = 0; s < weights.size(); ++s) {
      acc += weights[s];
      if (r <= acc) { pick = s; break; }
    }
    const std::vector<Item>& ds = *sets[pick];
    size_t n = idx_pools[pick] ? idx_pools[pick]->size() : ds.size();
    if (n == 0) continue;
    size_t j = (size_t)rng.below((uint64_t)n);
    const Item& it = ds[idx_pools[pick] ? (size_t)(*idx_pools[pick])[j] : j];
    float margin = (float)rng.range(margins[pick].first, margins[pick].second);
    float bright = 1.f, contrast = 1.f;
    if (photo[pick]) {
      bright = (float)rng.range(0.75, 1.25);
      contrast = (float)rng.range(0.85, 1.15);
    }
    std::vector<float> v = load_input(it, margin, bright, contrast);
    std::copy(v.begin(), v.end(), xb.begin() + (size_t)k * 3 * 128 * 128);
    for (int h = 0; h < 11; ++h) {
      b.labels[(size_t)h][(size_t)k] = it.heads[(size_t)h];
      b.mask[(size_t)h][(size_t)k] = it.mask[(size_t)h];
    }
  }
  b.x = from_data({batch, 3, 128, 128}, xb);
  return b;
}

// region top-1 on a set of indices (single crop, no TTA, FIXED margin) — the same number the Python
// side prints. The margin is fixed on purpose: running the training augmentation through the metric
// made the baseline look 8 points worse and made every run incomparable.
static constexpr float EVAL_MARGIN = 0.03f;

inline double eval_region(onx::Trainable& t, const std::vector<Item>& ds, const std::vector<int>& idxs,
                          size_t region_head, int limit, int* out_n = nullptr) {
  int n = 0, ok = 0;
  for (size_t k = 0; k < idxs.size() && (limit <= 0 || n < limit); ++k) {
    const Item& it = ds[(size_t)idxs[k]];
    std::vector<float> v = load_input(it, EVAL_MARGIN);
    Tensor x = from_data({1, 3, 128, 128}, v);
    auto vals = onx::forward(t, x);
    const std::vector<float>& p = vals.at(t.heads[region_head])->data;
    int best = 0;
    for (size_t i = 1; i < p.size(); ++i) if (p[i] > p[best]) best = (int)i;
    ok += (best == it.heads[0]) ? 1 : 0;
    ++n;
    free_graph(vals.at(t.heads[region_head]));
  }
  if (out_n) *out_n = n;
  return n ? (double)ok / n : 0.0;
}

// --- checkpoints ------------------------------------------------------------------------------
// The last step is rarely the model you want: with only 576 real crops the region head peaks early and
// then slides, while the region names that only synthetic teaches keep improving to the end. Snapshots
// are cheap here (307k floats), so the trainer keeps candidates and picks at the end, exactly like
// tools/train_ocr.py does.
using Snapshot = std::vector<std::vector<float>>;

inline Snapshot snapshot(const onx::Trainable& t) {
  Snapshot s;
  s.reserve(t.params.size());
  for (const Tensor& p : t.params) s.push_back(p->data);
  return s;
}

inline void restore(onx::Trainable& t, const Snapshot& s) {
  for (size_t i = 0; i < t.params.size() && i < s.size(); ++i) t.params[i]->data = s[i];
}

}  // namespace trn
