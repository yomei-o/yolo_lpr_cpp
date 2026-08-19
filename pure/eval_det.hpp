// `jlpr val --model det` — detector metrics without Ultralytics, without Python.
//
// Two different questions, two different matchings, both reported:
//
//   1. **mAP50 / mAP50-95** — the number everyone quotes. Detections from the whole set are ranked by
//      confidence and matched greedily by IoU (one gt per detection), exactly as Ultralytics'
//      `match_predictions` does, and the precision/recall curve is integrated with the same 101-point
//      interpolation as `compute_ap`. Same algorithm on the Python side (tools/eval_det.py) so the two
//      print the same number; the *inputs* differ from an Ultralytics `val` run, though, because this
//      project feeds the net a plain resize and Ultralytics letterboxes.
//   2. **recall bucketed by plate size** — what this project actually decides on. A single mAP hides
//      the failure this repo exists to fix: the borrowed detectors score 0.83 on a plate at 8% of the
//      frame and nothing at all past 30% (tools/context_test.py). The bucket table is the M7 acceptance
//      test, and its matching is per-gt best-IoU (not confidence-ranked), which is what
//      tools/eval_det.py has always done.
//
// P/R/F1 are reported **at the given confidence** (default 0.25), not at the max-F1 point Ultralytics
// searches for — a fixed threshold is what the CLI and the WASM demo actually run at, and both
// languages can compute it identically without a curve-smoothing convention.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace evd {

struct GtBox { float x1 = 0, y1 = 0, x2 = 0, y2 = 0, share = 0; };   // share = w / image width
struct DetBox { float x1 = 0, y1 = 0, x2 = 0, y2 = 0, score = 0; };

inline float iou_xyxy(const GtBox& a, const DetBox& b) {
  const float iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  const float ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (iw <= 0 || ih <= 0) return 0.f;
  const float inter = iw * ih;
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
}

inline std::vector<float> iou_thresholds() {                 // 0.50, 0.55, ... 0.95
  std::vector<float> t;
  for (int i = 0; i < 10; ++i) t.push_back(0.5f + 0.05f * i);
  return t;
}

// One detection's fate: its confidence and whether it is a true positive at each IoU threshold.
struct Scored {
  float conf = 0;
  std::vector<char> tp;
};

// Ultralytics match_predictions: all (gt, det) pairs above the threshold, sorted by IoU descending,
// then kept only if neither the detection nor the gt has already been used.
inline void match_frame(const std::vector<DetBox>& dets, const std::vector<GtBox>& gts,
                        const std::vector<float>& thr, std::vector<Scored>& out) {
  std::vector<std::array<float, 3>> pairs;                   // iou, gt index, det index
  for (size_t g = 0; g < gts.size(); ++g)
    for (size_t d = 0; d < dets.size(); ++d) {
      const float v = iou_xyxy(gts[g], dets[d]);
      if (v > 0) pairs.push_back({v, (float)g, (float)d});
    }
  std::stable_sort(pairs.begin(), pairs.end(),
                   [](const std::array<float, 3>& a, const std::array<float, 3>& b) { return a[0] > b[0]; });
  const size_t base = out.size();
  for (const DetBox& d : dets) { Scored s; s.conf = d.score; s.tp.assign(thr.size(), 0); out.push_back(s); }
  for (size_t t = 0; t < thr.size(); ++t) {
    std::vector<char> gused(gts.size(), 0), dused(dets.size(), 0);
    for (const std::array<float, 3>& p : pairs) {
      if (p[0] < thr[t]) break;
      const size_t g = (size_t)p[1], d = (size_t)p[2];
      if (gused[g] || dused[d]) continue;
      gused[g] = dused[d] = 1;
      out[base + d].tp[t] = 1;
    }
  }
}

// compute_ap: the precision envelope, integrated at 101 recall points (Ultralytics' "interp" method).
// The sentinels are theirs verbatim — mrec = [0, recall.., recall[-1], 1], mpre = [1, precision.., 0, 0] —
// because they decide what happens past the last recall point, which is most of the curve when the
// detector misses a class entirely.
inline float compute_ap(const std::vector<float>& recall, const std::vector<float>& precision) {
  std::vector<float> mrec, mpre;
  mrec.push_back(0.f);
  mpre.push_back(1.f);
  for (size_t i = 0; i < recall.size(); ++i) { mrec.push_back(recall[i]); mpre.push_back(precision[i]); }
  mrec.push_back(recall.empty() ? 1.f : recall.back());
  mpre.push_back(0.f);
  mrec.push_back(1.f);
  mpre.push_back(0.f);
  for (size_t i = mpre.size() - 1; i-- > 0;) mpre[i] = std::max(mpre[i], mpre[i + 1]);   // envelope
  auto interp = [&](float x) {                                            // np.interp
    if (x <= mrec.front()) return mpre.front();
    if (x >= mrec.back()) return mpre.back();
    const size_t i = (size_t)(std::lower_bound(mrec.begin(), mrec.end(), x) - mrec.begin());
    const float x0 = mrec[i - 1], x1 = mrec[i];
    if (x1 <= x0) return mpre[i];
    return mpre[i - 1] + (x - x0) / (x1 - x0) * (mpre[i] - mpre[i - 1]);
  };
  double ap = 0.0;
  float prev = interp(0.f);
  for (int k = 1; k <= 100; ++k) {                                        // np.trapz over 101 points
    const float y = interp(0.01f * k);
    ap += 0.5 * (y + prev) * 0.01;
    prev = y;
  }
  return (float)ap;
}

struct Report {
  float map50 = 0, map5095 = 0;
  float precision = 0, recall = 0, f1 = 0;                   // at the reporting confidence
  int tp = 0, fp = 0, fn = 0, n_gt = 0, n_det = 0;
};

// `all` = every detection over the whole set (any confidence), `n_gt` = total ground-truth boxes.
inline Report summarize(std::vector<Scored> all, int n_gt, float report_conf) {
  Report r;
  r.n_gt = n_gt;
  r.n_det = (int)all.size();
  std::stable_sort(all.begin(), all.end(), [](const Scored& a, const Scored& b) { return a.conf > b.conf; });
  const size_t T = all.empty() ? 0 : all[0].tp.size();
  std::vector<float> aps(T, 0.f);
  for (size_t t = 0; t < T; ++t) {
    std::vector<float> rec, prec;
    int tpc = 0, fpc = 0;
    for (const Scored& s : all) {
      if (s.tp[t]) ++tpc; else ++fpc;
      rec.push_back((float)tpc / std::max(1e-9f, (float)n_gt));
      prec.push_back((float)tpc / std::max(1, tpc + fpc));
    }
    aps[t] = rec.empty() ? 0.f : compute_ap(rec, prec);
  }
  if (!aps.empty()) {
    r.map50 = aps[0];
    double s = 0;
    for (float v : aps) s += v;
    r.map5095 = (float)(s / aps.size());
  }
  for (const Scored& s : all) {
    if (s.conf < report_conf) continue;
    if (!s.tp.empty() && s.tp[0]) ++r.tp; else ++r.fp;
  }
  r.fn = n_gt - r.tp;
  r.precision = r.tp + r.fp > 0 ? (float)r.tp / (r.tp + r.fp) : 0.f;
  r.recall = n_gt > 0 ? (float)r.tp / n_gt : 0.f;
  r.f1 = (r.precision + r.recall) > 0 ? 2 * r.precision * r.recall / (r.precision + r.recall) : 0.f;
  return r;
}

// The bucket table's matching, kept identical to tools/eval_det.py: for each gt, the best-IoU
// detection that no earlier gt has taken. Confidence order plays no part here.
inline const std::array<std::array<float, 2>, 6>& buckets() {
  static const std::array<std::array<float, 2>, 6> B = {{{0.00f, 0.05f}, {0.05f, 0.10f}, {0.10f, 0.20f},
                                                         {0.20f, 0.35f}, {0.35f, 0.60f}, {0.60f, 1.01f}}};
  return B;
}

inline int bucket_of(float share) {
  const auto& B = buckets();
  for (int i = 0; i < (int)B.size(); ++i) if (share >= B[i][0] && share < B[i][1]) return i;
  return (int)B.size() - 1;
}

inline void bucket_match(const std::vector<DetBox>& dets, const std::vector<GtBox>& gts, float iou_thr,
                         std::vector<int>& tp, std::vector<int>& gt_count, int& unmatched_dets) {
  std::vector<char> used(dets.size(), 0);
  for (const GtBox& g : gts) {
    const int b = bucket_of(g.share);
    ++gt_count[(size_t)b];
    float best = 0.f;
    int bi = -1;
    for (size_t d = 0; d < dets.size(); ++d) {
      if (used[d]) continue;
      const float v = iou_xyxy(g, dets[d]);
      if (v > best) { best = v; bi = (int)d; }
    }
    if (best >= iou_thr && bi >= 0) { ++tp[(size_t)b]; used[(size_t)bi] = 1; }
  }
  for (char u : used) if (!u) ++unmatched_dets;
}

}  // namespace evd
