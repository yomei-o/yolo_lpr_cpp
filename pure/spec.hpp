// Label tables read at run time from spec/labels.txt — the single source of truth shared with
// tools/labels.py. Nothing here hardcodes a class list, so C++ and Python cannot drift.
//
// spec format:  [head <name> <n>] / [gen <name> <n>] section headers, then one token per line
//               (line order = class index). '#' starts a comment; a tab before '#' is allowed.
//               Tokens <blank> (renders as nothing) and <unused> (never predicted) are special.
#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include "rng.hpp"

namespace spec {

struct Group { std::string kind, name; int n = 0; std::vector<std::string> tok; };

struct Spec {
  int version = 0;
  std::vector<Group> groups;                       // heads first, then gens (file order)
  const Group* find(const std::string& name) const {
    for (auto& g : groups) if (g.name == name) return &g;
    return nullptr;
  }
  const Group& head(const std::string& name) const {
    const Group* g = find(name);
    if (!g) { printf("spec: no such group '%s'\n", name.c_str()); std::exit(1); }
    return *g;
  }
  std::vector<const Group*> of_kind(const std::string& k) const {
    std::vector<const Group*> v; for (auto& g : groups) if (g.kind == k) v.push_back(&g); return v;
  }
};

inline std::string trim_(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

inline Spec parse(const std::string& text) {
  Spec sp;
  std::istringstream in(text);
  std::string line;
  Group* cur = nullptr;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.rfind("# version ", 0) == 0) sp.version = std::atoi(line.c_str() + 10);
    size_t tab = line.find('\t');                                  // strip trailing tab-comment
    if (tab != std::string::npos && line.find('#', tab) != std::string::npos) line = line.substr(0, tab);
    std::string t = trim_(line);
    if (t.empty() || t[0] == '#') continue;
    if (t[0] == '[') {
      std::string body = t.substr(1, t.find(']') - 1);
      std::istringstream hs(body);
      Group g; hs >> g.kind >> g.name >> g.n;
      sp.groups.push_back(std::move(g));
      cur = &sp.groups.back();
      continue;
    }
    if (!cur) { printf("spec: token outside a section: %s\n", t.c_str()); std::exit(1); }
    cur->tok.push_back(t);
  }
  for (auto& g : sp.groups)
    if ((int)g.tok.size() != g.n) {
      printf("spec: group %s declares %d but has %d tokens\n", g.name.c_str(), g.n, (int)g.tok.size());
      std::exit(1);
    }
  return sp;
}

inline Spec load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { printf("spec: cannot open %s\n", path.c_str()); std::exit(1); }
  std::ostringstream ss; ss << f.rdbuf();
  return parse(ss.str());
}

// ---- decode: head indices -> plate strings -------------------------------------------------
// head order is the file order of the [head ...] sections:
//   region, class_num_01..03, hiragana, plate_num_01..04, plate_kind, legible
struct Plate {
  std::string region, cls, hira, num, disp, kind, legible, text;
};

inline bool special_(const std::string& t) { return t == "<blank>" || t == "<unused>"; }

inline Plate decode(const Spec& sp, const std::vector<int>& idx) {
  auto heads = sp.of_kind("head");
  Plate p;
  auto tok = [&](size_t h) -> std::string {
    if (h >= heads.size() || h >= idx.size()) return "?";
    int i = idx[h];
    if (i < 0 || i >= heads[h]->n) return "?";
    return heads[h]->tok[i];
  };
  p.region = tok(0);
  for (size_t h = 1; h <= 3; ++h) { std::string t = tok(h); if (!special_(t)) p.cls += t; }
  p.hira = tok(4);
  for (size_t h = 5; h <= 8; ++h) {
    std::string t = tok(h);
    if (special_(t)) { p.disp += "\xE3\x83\xBB"; }                 // U+30FB '・'
    else { p.num += t; p.disp += t; }
  }
  p.kind = tok(9);
  p.legible = tok(10);
  p.text = p.region + " " + p.cls + " " + p.hira + " " + p.disp;
  return p;
}

// ---- canonical dump: byte-identical between C++ and Python (the parity test diffs this) ----
inline std::string canonical_dump(const Spec& sp) {
  std::ostringstream o;
  o << "spec version " << sp.version << "\n";
  o << "groups " << sp.groups.size() << "\n";
  for (auto& g : sp.groups) {
    o << g.kind << " " << g.name << " " << g.n << "\n";
    for (int i = 0; i < g.n; ++i) o << "  " << i << "\t" << g.tok[i] << "\n";
  }
  return o.str();
}

// Deterministic decode test vectors: draw index tuples with the shared splitmix64 (rng.hpp) and
// dump the decoded strings. Python must produce the same bytes, which pins parser + decode + RNG.
inline std::string decode_vectors_dump(const Spec& sp, uint64_t seed, int count) {
  auto heads = sp.of_kind("head");
  Rng rng(seed);
  std::ostringstream o;
  o << "decode-vectors " << count << " seed " << seed << "\n";
  for (int k = 0; k < count; ++k) {
    std::vector<int> idx;
    for (auto* h : heads) idx.push_back((int)rng.below((uint64_t)h->n));
    for (size_t i = 0; i < idx.size(); ++i) o << (i ? " " : "  ") << idx[i];
    Plate p = decode(sp, idx);
    o << "\t" << p.region << "|" << p.cls << "|" << p.hira << "|" << p.num << "|" << p.disp
      << "|" << p.kind << "|" << p.legible << "\t" << p.text << "\n";
  }
  return o.str();
}

}  // namespace spec
