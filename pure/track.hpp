// Tracking across frames — so a camera can be pointed at a plate and just read it, and so the
// reading gets *better* the longer the plate stays in view.
//
// Two jobs, both cheap:
//
//  1. Identity. Boxes from consecutive frames are matched greedily by IoU, so a plate keeps its id
//     while it moves. Tracks that miss too many frames are dropped.
//
//  2. Temporal voting. Per-head probabilities are SUMMED over the frames of a track, exactly the way
//     the margin TTA sums over crops — but over time, which is free: the frames are arriving anyway.
//     A plate seen for a second is read from a dozen slightly different views (hand shake, exposure,
//     rolling shutter), and the summed winner is the reading the evidence actually supports. The
//     per-head share of the summed probability is reported, so a shaky region head shows as shaky
//     instead of being asserted.
//
// A track is called stable once the decoded string has repeated `stable_needed` frames in a row. That
// is what the UI waits for before it stops scanning.
#pragma once
#include "pipeline.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace jl {

struct Track {
  int id = 0;
  Box box{};                              // last seen box
  int hits = 0, misses = 0, frames = 0;
  std::vector<std::vector<double>> sum;   // [head][class] accumulated probability
  std::vector<int> arg;                   // current winner per head
  std::vector<float> conf;                // winner's share of the summed probability
  spec::Plate plate;                      // decoded from `arg`
  std::string last_text;
  int stable = 0;                         // consecutive frames with an unchanged reading
};

struct TrackerCfg {
  float iou_match = 0.25f;                // low: plates move fast in a hand-held frame
  int max_misses = 8;                     // drop a track after this many frames without a match
  int stable_needed = 3;
};

class Tracker {
 public:
  void reset() { tracks_.clear(); next_id_ = 1; }

  // Fold one frame's detections + per-head probabilities into the tracks.
  // `probs[i]` are the raw per-head probability vectors for boxes[i] (already softmaxed by the graph).
  void update(const std::vector<Box>& boxes, const std::vector<std::vector<std::vector<float>>>& probs,
              const spec::Spec& sp, const TrackerCfg& cfg) {
    std::vector<char> used(tracks_.size(), 0);
    for (size_t i = 0; i < boxes.size(); ++i) {
      int best = -1;
      float best_iou = cfg.iou_match;
      for (size_t t = 0; t < tracks_.size(); ++t) {
        if (used[t]) continue;
        float v = iou(boxes[i], tracks_[t].box);
        if (v >= best_iou) { best_iou = v; best = (int)t; }
      }
      Track* tr;
      if (best < 0) {
        Track nt;
        nt.id = next_id_++;
        tracks_.push_back(std::move(nt));
        tr = &tracks_.back();
        used.push_back(1);
      } else {
        tr = &tracks_[(size_t)best];
        used[(size_t)best] = 1;
      }
      tr->box = boxes[i];
      tr->hits++;
      tr->frames++;
      tr->misses = 0;
      const std::vector<std::vector<float>>& p = probs[i];
      if (tr->sum.size() != p.size()) tr->sum.assign(p.size(), {});
      for (size_t h = 0; h < p.size(); ++h) {
        if (tr->sum[h].size() != p[h].size()) tr->sum[h].assign(p[h].size(), 0.0);
        for (size_t c = 0; c < p[h].size(); ++c) tr->sum[h][c] += p[h][c];
      }
      finalise(*tr, sp, cfg);
    }
    for (size_t t = 0; t < tracks_.size(); ++t) if (!used[t]) tracks_[t].misses++;
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& t) { return t.misses > cfg.max_misses; }),
                  tracks_.end());
  }

  const std::vector<Track>& tracks() const { return tracks_; }

 private:
  static float iou(const Box& a, const Box& b) {
    float iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
    if (iw <= 0 || ih <= 0) return 0.f;
    float inter = iw * ih;
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0 ? inter / ua : 0.f;
  }

  static void finalise(Track& t, const spec::Spec& sp, const TrackerCfg& cfg) {
    t.arg.clear();
    t.conf.clear();
    for (const std::vector<double>& s : t.sum) {
      int best = 0;
      double bv = -1, tot = 0;
      for (size_t c = 0; c < s.size(); ++c) { tot += s[c]; if (s[c] > bv) { bv = s[c]; best = (int)c; } }
      t.arg.push_back(best);
      t.conf.push_back(tot > 0 ? (float)(bv / tot) : 0.f);
    }
    t.plate = spec::decode(sp, t.arg);
    if (t.plate.text == t.last_text) t.stable++;
    else { t.stable = 1; t.last_text = t.plate.text; }
    (void)cfg;
  }

  std::vector<Track> tracks_;
  int next_id_ = 1;
};

// JSON for the UI: one entry per live track, with how long it has been seen and how stable it is.
inline std::string tracks_json(const Tracker& tk, const TrackerCfg& cfg) {
  std::string o = "{\"tracks\":[";
  char b[256];
  const std::vector<Track>& ts = tk.tracks();
  for (size_t i = 0; i < ts.size(); ++i) {
    const Track& t = ts[i];
    snprintf(b, sizeof b,
             "%s{\"id\":%d,\"box\":[%.1f,%.1f,%.1f,%.1f],\"det\":%.3f,\"frames\":%d,\"stable\":%d,"
             "\"settled\":%s,", i ? "," : "", t.id, t.box.x1, t.box.y1, t.box.x2, t.box.y2,
             t.box.score, t.frames, t.stable, t.stable >= cfg.stable_needed ? "true" : "false");
    o += b;
    o += "\"text\":\"" + json_escape(t.plate.text) + "\",";
    o += "\"region\":\"" + json_escape(t.plate.region) + "\",";
    o += "\"kind\":\"" + json_escape(t.plate.kind) + "\",";
    o += "\"conf\":[";
    for (size_t h = 0; h < t.conf.size(); ++h) {
      snprintf(b, sizeof b, "%s%.4f", h ? "," : "", t.conf[h]);
      o += b;
    }
    o += "]}";
  }
  o += "]}";
  return o;
}

}  // namespace jl
