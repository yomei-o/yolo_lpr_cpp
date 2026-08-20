// Training-run plumbing: checkpoint / resume, early stopping, a CSV log.
//
// None of this changes what a step computes — it is what makes a *long* run survivable, which is the
// difference between "the C++ side can train" and "the C++ side can be left training". A 28-epoch
// detector run is 70 minutes on a T4 and hours on a CPU; without resume, any interruption costs the
// whole run, and without a log there is nothing to look at afterwards but scrollback.
//
//   rt::Ckpt ck; ck.step = step; ck.add(name, params, slot1, slot2, ema);   // then ck.save(path)
//   if (!resume.empty() && ck.load(resume)) start = ck.step + 1;            // continue where it died
//   rt::Log log(path, "step,loss,lr");  log.row("%d,%.6f,%.3e", step, loss, lr);
//   rt::Patience pat(n); if (pat.done(metric)) break;                       // n evals with no gain
//
// The checkpoint holds the optimizer's moments and the EMA shadow, not just the weights: resuming
// from weights alone restarts momentum from zero, which shows up as a visible bump in the loss.
#pragma once
#include "rng.hpp"
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace rt {

// ---------------------------------------------------------------- checkpoint

struct Slot {
  std::string name;                 // parameter name, so a resume can refuse a mismatched model
  std::vector<float> data;          // the weights
  std::vector<float> s1, s2, ema;   // SGD buf / Adam m, Adam v, EMA shadow (any may be empty)
};

struct Ckpt {
  static const uint32_t MAGIC = 0x4B434C4Au;   // "JLCK"
  static const uint32_t VERSION = 2;
  int64_t step = 0;                 // the last step that completed
  int64_t total = 0;                // the run length the schedule was computed from. Without this a
                                    // resume silently changes the lr curve: stopping a 6-step run at
                                    // 3 and resuming is not the same as running 3 then 3 (measured —
                                    // the tail losses differ in the third decimal).
  int32_t opt_kind = 0;             // 1 = SGD, 2 = Adam/AdamW
  int32_t opt_t = 0;                // Adam's t, or SGD's "buffer started" flag
  int32_t ema_updates = 0;
  double best = 0;                  // the metric the best-so-far export used
  uint64_t rng = 0;
  std::vector<Slot> slots;

  void add(const std::string& name, const std::vector<float>& d, const std::vector<float>& s1 = {},
           const std::vector<float>& s2 = {}, const std::vector<float>& ema = {}) {
    Slot s;
    s.name = name; s.data = d; s.s1 = s1; s.s2 = s2; s.ema = ema;
    slots.push_back(std::move(s));
  }

  bool save(const std::string& path) const {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    auto u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto i64 = [&](int64_t v) { fwrite(&v, 8, 1, f); };
    auto vec = [&](const std::vector<float>& v) {
      i64((int64_t)v.size());
      if (!v.empty()) fwrite(v.data(), 4, v.size(), f);
    };
    u32(MAGIC); u32(VERSION);
    i64(step); i64(total); u32((uint32_t)opt_kind); u32((uint32_t)opt_t); u32((uint32_t)ema_updates);
    fwrite(&best, 8, 1, f);
    fwrite(&rng, 8, 1, f);
    u32((uint32_t)slots.size());
    for (const Slot& s : slots) {
      u32((uint32_t)s.name.size());
      fwrite(s.name.data(), 1, s.name.size(), f);
      vec(s.data); vec(s.s1); vec(s.s2); vec(s.ema);
    }
    fclose(f);
    return true;
  }

  // Reads into this object. `why` gets a reason on failure, so a bad --resume says what is wrong
  // instead of silently starting over.
  bool load(const std::string& path, std::string* why = nullptr) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { if (why) *why = "cannot open " + path; return false; }
    auto rd = [&](void* p, size_t n) { return fread(p, 1, n, f) == n; };
    uint32_t magic = 0, ver = 0, n = 0;
    bool ok = rd(&magic, 4) && rd(&ver, 4);
    if (!ok || magic != MAGIC) { fclose(f); if (why) *why = path + " is not a jlpr checkpoint"; return false; }
    if (ver != VERSION) { fclose(f); if (why) *why = "checkpoint version mismatch"; return false; }
    uint32_t k = 0, t = 0, eu = 0;
    ok = rd(&step, 8) && rd(&total, 8) && rd(&k, 4) && rd(&t, 4) && rd(&eu, 4) && rd(&best, 8) && rd(&rng, 8) && rd(&n, 4);
    opt_kind = (int32_t)k; opt_t = (int32_t)t; ema_updates = (int32_t)eu;
    if (!ok) { fclose(f); if (why) *why = "truncated header"; return false; }
    slots.clear();
    auto vec = [&](std::vector<float>& v) {
      int64_t m = 0;
      if (!rd(&m, 8) || m < 0) return false;
      v.assign((size_t)m, 0.f);
      return m == 0 || rd(v.data(), (size_t)m * 4);
    };
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t len = 0;
      if (!rd(&len, 4)) { fclose(f); if (why) *why = "truncated slot"; return false; }
      Slot s;
      s.name.assign(len, '\0');
      if (len && !rd(&s.name[0], len)) { fclose(f); if (why) *why = "truncated name"; return false; }
      if (!vec(s.data) || !vec(s.s1) || !vec(s.s2) || !vec(s.ema)) {
        fclose(f); if (why) *why = "truncated tensor in " + s.name; return false;
      }
      slots.push_back(std::move(s));
    }
    fclose(f);
    return true;
  }
};

// ---------------------------------------------------------------- csv log

struct Log {
  FILE* f = nullptr;
  Log() = default;
  Log(const std::string& path, const std::string& header) { open(path, header); }
  void open(const std::string& path, const std::string& header) {
    if (path.empty()) return;
    f = fopen(path.c_str(), "wb");
    if (f) { fputs(header.c_str(), f); fputc('\n', f); fflush(f); }
  }
  // Flushed per row on purpose: a run that dies should still leave the curve up to the last step.
  void row(const char* fmt, ...) {
    if (!f) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fputc('\n', f);
    fflush(f);
  }
  ~Log() { if (f) fclose(f); }
  Log(const Log&) = delete;
  Log& operator=(const Log&) = delete;
};

// ---------------------------------------------------------------- early stop

// "No improvement for `patience` evaluations" — the same rule Ultralytics uses, with the metric's
// direction as a parameter because corner error wants to go down and mAP wants to go up.
struct Patience {
  int patience;                     // 0 disables
  bool higher_better;
  double best = 0;
  int since = 0;
  bool have = false;
  explicit Patience(int p = 0, bool higher_better = true) : patience(p), higher_better(higher_better) {}
  // returns true when the run should stop
  bool done(double metric) {
    bool better = !have || (higher_better ? metric > best : metric < best);
    if (better) { best = metric; since = 0; have = true; }
    else ++since;
    return patience > 0 && since >= patience;
  }
};

}  // namespace rt
