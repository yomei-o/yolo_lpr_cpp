// (Ported from the sibling repository yolov8_cpp, same author, BSD-3-Clause. Unchanged except this
// note: it is the piece that lets `jlpr init-det --from-pt yolov8n.pt` exist without Python.)
// Pure-C++ PyTorch .pt reader/writer — no Python, no libtorch. A .pt is a ZIP (STORED)
// holding a pickle (protocol 2) plus one raw storage per tensor under data/<k>.
//   * load_pt(path)        : read a state_dict .pt  (OrderedDict[str->Tensor])
//   * load_pt_module(path) : read a raw checkpoint  ({'model': nn.Module, ...}) by walking
//                            the pickled module tree -> state_dict (handles FP16 storage)
//   * save_pt(tensors,path): write a state_dict .pt (float32) that torch.load / load_state_dict read
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>
#include <algorithm>

namespace pt {

struct Tensor { std::string name; std::vector<int64_t> shape; std::vector<float> data; };

// ----------------------------- CRC32 (zip) -----------------------------
inline uint32_t crc32_of(const uint8_t* p, size_t n) {
  static uint32_t T[256]; static bool init = false;
  if (!init) { for (uint32_t i = 0; i < 256; ++i) { uint32_t c = i; for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1; T[i] = c; } init = true; }
  uint32_t c = 0xFFFFFFFFu; for (size_t i = 0; i < n; ++i) c = T[(c ^ p[i]) & 0xFF] ^ (c >> 8); return c ^ 0xFFFFFFFFu;
}
inline float half_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff, f;
  if (exp == 0) { if (man == 0) f = sign << 31; else { int e = -1; do { ++e; man <<= 1; } while (!(man & 0x400)); man &= 0x3ff; f = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (man << 13); } }
  else if (exp == 0x1f) f = (sign << 31) | (0xffu << 23) | (man << 13);
  else f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
  float r; std::memcpy(&r, &f, 4); return r;
}
inline void pu16(std::string& b, uint16_t v) { b.push_back((char)(v & 0xff)); b.push_back((char)(v >> 8)); }
inline void pu32(std::string& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((char)((v >> (8 * i)) & 0xff)); }

// ============================== WRITER ==============================
inline void pk_bININT(std::string& p, int64_t v) { p.push_back('J'); pu32(p, (uint32_t)(int32_t)v); }
inline void pk_str(std::string& p, const std::string& s) { p.push_back('X'); pu32(p, (uint32_t)s.size()); p += s; }
inline void pk_global(std::string& p, const std::string& mod, const std::string& name) { p.push_back('c'); p += mod; p.push_back('\n'); p += name; p.push_back('\n'); }

inline std::string build_pickle(const std::vector<Tensor>& ts) {
  std::string p;
  p.push_back((char)0x80); p.push_back(2);
  pk_global(p, "collections", "OrderedDict"); p.push_back(')'); p.push_back('R');
  p.push_back('(');
  for (size_t i = 0; i < ts.size(); ++i) {
    int64_t numel = 1; for (int64_t d : ts[i].shape) numel *= d;
    pk_str(p, ts[i].name);
    pk_global(p, "torch._utils", "_rebuild_tensor_v2");
    p.push_back('(');
      p.push_back('(');
        pk_str(p, "storage"); pk_global(p, "torch", "FloatStorage"); pk_str(p, std::to_string(i)); pk_str(p, "cpu"); pk_bININT(p, numel);
      p.push_back('t'); p.push_back('Q');
      pk_bININT(p, 0);
      p.push_back('('); for (int64_t d : ts[i].shape) pk_bININT(p, d); p.push_back('t');
      p.push_back('('); for (size_t k = 0; k < ts[i].shape.size(); ++k) { int64_t s = 1; for (size_t j = k + 1; j < ts[i].shape.size(); ++j) s *= ts[i].shape[j]; pk_bININT(p, s); } p.push_back('t');
      p.push_back((char)0x89);
      pk_global(p, "collections", "OrderedDict"); p.push_back(')'); p.push_back('R');
    p.push_back('t'); p.push_back('R');
  }
  p.push_back('u'); p.push_back('.');
  return p;
}
inline void zip_add(std::string& out, std::string& central, const std::string& name, const std::string& data) {
  uint32_t crc = crc32_of((const uint8_t*)data.data(), data.size());
  uint32_t local_off = (uint32_t)out.size(), hdr = 30 + (uint32_t)name.size();
  uint32_t pad = (64 - (local_off + hdr + 4) % 64) % 64; uint16_t extra = (uint16_t)(4 + pad);
  out += std::string("PK\x03\x04", 4); pu16(out, 20); pu16(out, 0); pu16(out, 0); pu16(out, 0); pu16(out, 0);
  pu32(out, crc); pu32(out, (uint32_t)data.size()); pu32(out, (uint32_t)data.size());
  pu16(out, (uint16_t)name.size()); pu16(out, extra); out += name; pu16(out, 0xCAFE); pu16(out, (uint16_t)pad); out.append(pad, '\0'); out += data;
  central += std::string("PK\x01\x02", 4); pu16(central, 20); pu16(central, 20); pu16(central, 0); pu16(central, 0); pu16(central, 0); pu16(central, 0);
  pu32(central, crc); pu32(central, (uint32_t)data.size()); pu32(central, (uint32_t)data.size());
  pu16(central, (uint16_t)name.size()); pu16(central, 0); pu16(central, 0); pu16(central, 0); pu16(central, 0); pu32(central, 0); pu32(central, local_off); central += name;
}
inline void save_pt(const std::vector<Tensor>& ts, const std::string& path) {
  std::string out, central; const std::string A = "archive/";
  zip_add(out, central, A + "data.pkl", build_pickle(ts));
  zip_add(out, central, A + "byteorder", "little");
  for (size_t i = 0; i < ts.size(); ++i) zip_add(out, central, A + "data/" + std::to_string(i), std::string((const char*)ts[i].data.data(), ts[i].data.size() * 4));
  zip_add(out, central, A + "version", "3\n");
  uint32_t cd_off = (uint32_t)out.size(); out += central;
  out += std::string("PK\x05\x06", 4); pu16(out, 0); pu16(out, 0);
  pu16(out, (uint16_t)ts.size() + 3); pu16(out, (uint16_t)ts.size() + 3);
  pu32(out, (uint32_t)central.size()); pu32(out, cd_off); pu16(out, 0);
  std::ofstream f(path, std::ios::binary); f.write(out.data(), out.size());
}

// ============================== READER ==============================
struct Val {
  enum T { I, F, S, TUP, LIST, GLB, PID, BOOL, DICT, OBJ, TEN, NONE } t = NONE;
  int64_t i = 0; double f = 0; std::string s;                 // I / F / S / GLB
  std::vector<Val> tup;                                       // TUP / LIST / PID
  std::vector<std::pair<Val, Val>> items;                     // DICT / OBJ (state)
  std::string tkey; int tdt = 0; std::vector<int64_t> tshape; // TEN: storage key, dtype(0=f32,1=f16), shape
};

struct Zip {
  std::string buf; const uint8_t* B = nullptr; size_t N = 0;
  std::vector<std::pair<std::string, std::pair<size_t, size_t>>> ent;   // name -> (data offset, size)
  void open(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate); if (!f) { printf("cannot open %s\n", path.c_str()); std::exit(1); }
    N = f.tellg(); f.seekg(0); buf.resize(N); f.read(buf.data(), N); B = (const uint8_t*)buf.data();
    auto u16 = [&](size_t o) { return (uint32_t)B[o] | (B[o + 1] << 8); };
    auto u32 = [&](size_t o) { return (uint32_t)B[o] | (B[o+1]<<8) | (B[o+2]<<16) | ((uint32_t)B[o+3]<<24); };
    size_t eocd = N - 22; while (eocd > 0 && u32(eocd) != 0x06054b50) --eocd;
    uint32_t cd = u32(eocd + 16), cnt = u16(eocd + 10); size_t p = cd;
    for (uint32_t e = 0; e < cnt; ++e) { uint32_t csize = u32(p+20), nlen = u16(p+28), elen = u16(p+30), clen = u16(p+32), lo = u32(p+42);
      std::string nm((const char*)B + p + 46, nlen); uint32_t ln = u16(lo+26), le = u16(lo+28);
      ent.push_back({nm, {lo + 30 + ln + le, csize}}); p += 46 + nlen + elen + clen; }
  }
  std::pair<size_t, size_t> find(const std::string& suf) const {
    for (auto& e : ent) if (e.first.size() >= suf.size() && e.first.compare(e.first.size() - suf.size(), suf.size(), suf) == 0) return e.second;
    return {0, 0};
  }
};

// Run the protocol-2 pickle in `data.pkl` and return the top object as a Val tree.
inline Val run_pickle(const uint8_t* P, size_t PN) {
  size_t ip = 0; std::vector<Val> st; std::vector<size_t> marks; std::vector<Val> memo(65536);
  auto line = [&]() { std::string s; while (P[ip] != '\n') s.push_back((char)P[ip++]); ++ip; return s; };
  auto u32 = [&]() { uint32_t v = P[ip] | (P[ip+1]<<8) | (P[ip+2]<<16) | ((uint32_t)P[ip+3]<<24); ip += 4; return v; };
  auto memoset = [&](uint32_t idx) { if (idx >= memo.size()) memo.resize(idx + 1); memo[idx] = st.back(); };
  while (ip < PN) {
    uint8_t op = P[ip++];
    switch (op) {
      case 0x80: ++ip; break;                                                              // PROTO
      case 'c': { Val v; v.t = Val::GLB; std::string m = line(), nm = line(); v.s = m + " " + nm; st.push_back(v); } break;  // GLOBAL
      case 'q': memoset(P[ip++]); break;                                                    // BINPUT
      case 'r': memoset(u32()); break;                                                      // LONG_BINPUT
      case 'h': st.push_back(memo[P[ip++]]); break;                                          // BINGET
      case 'j': st.push_back(memo[u32()]); break;                                            // LONG_BINGET
      case ')': { st.push_back(Val{Val::TUP}); } break;                                      // EMPTY_TUPLE
      case '(': marks.push_back(st.size()); break;                                           // MARK
      case ']': { st.push_back(Val{Val::LIST}); } break;                                     // EMPTY_LIST
      case '}': { st.push_back(Val{Val::DICT}); } break;                                     // EMPTY_DICT
      case 'X': { uint32_t l = u32(); Val v; v.t = Val::S; v.s.assign((const char*)P + ip, l); ip += l; st.push_back(v); } break;  // BINUNICODE
      case 'K': { Val v; v.t = Val::I; v.i = P[ip++]; st.push_back(v); } break;              // BININT1
      case 'M': { Val v; v.t = Val::I; v.i = P[ip] | (P[ip+1]<<8); ip += 2; st.push_back(v); } break;  // BININT2
      case 'J': { Val v; v.t = Val::I; v.i = (int32_t)u32(); st.push_back(v); } break;       // BININT
      case 0x8a: { uint8_t ln = P[ip++]; int64_t val = 0; for (int k = 0; k < ln; ++k) val |= (int64_t)P[ip++] << (8*k); Val v; v.t = Val::I; v.i = val; st.push_back(v); } break;  // LONG1
      case 'G': { uint64_t u = 0; for (int k = 0; k < 8; ++k) u = (u << 8) | P[ip++]; Val v; v.t = Val::F; std::memcpy(&v.f, &u, 8); st.push_back(v); } break;  // BINFLOAT (big-endian)
      case 0x88: case 0x89: { Val v; v.t = Val::BOOL; v.i = (op == 0x88); st.push_back(v); } break;    // NEWTRUE/FALSE
      case 'N': st.push_back(Val{}); break;                                                  // NONE
      case 't': { size_t m = marks.back(); marks.pop_back(); Val v; v.t = Val::TUP; for (size_t k = m; k < st.size(); ++k) v.tup.push_back(st[k]); st.resize(m); st.push_back(v); } break;  // TUPLE
      case 0x85: case 0x86: case 0x87: { int k = op - 0x84; Val v; v.t = Val::TUP; for (int j = 0; j < k; ++j) { v.tup.insert(v.tup.begin(), st.back()); st.pop_back(); } st.push_back(v); } break;  // TUPLE1/2/3
      case 'Q': { Val v; v.t = Val::PID; v.tup = st.back().tup; st.pop_back(); st.push_back(v); } break;  // BINPERSID
      case 0x81: { Val args = st.back(); st.pop_back(); Val cls = st.back(); st.pop_back(); Val o; o.t = Val::OBJ; o.s = cls.s; st.push_back(o); } break;  // NEWOBJ
      case 'b': { Val state = st.back(); st.pop_back(); Val& o = st.back();                  // BUILD
                  const Val* d = (state.t == Val::DICT) ? &state : (state.t == Val::TUP && state.tup.size() >= 1 && state.tup[0].t == Val::DICT ? &state.tup[0] : nullptr);
                  if (d) for (auto& kv : d->items) o.items.push_back(kv); } break;
      case 'R': { Val args = st.back(); st.pop_back(); Val fn = st.back(); st.pop_back(); Val o;   // REDUCE
                  if (fn.t == Val::GLB && fn.s == "collections OrderedDict") o.t = Val::DICT;
                  else if (fn.t == Val::GLB && fn.s == "torch._utils _rebuild_tensor_v2") {
                    o.t = Val::TEN; const Val& pid = args.tup[0]; o.tkey = pid.tup[2].s;
                    const std::string& sty = pid.tup[1].s;                       // storage class
                    o.tdt = (sty.find("Half") != std::string::npos) ? 1 : (sty.find("Float") != std::string::npos) ? 0 : 2;  // 2 = non-float (e.g. Long) -> skip
                    for (auto& d : args.tup[2].tup) o.tshape.push_back(d.i); }
                  else if (fn.t == Val::GLB && fn.s == "torch._utils _rebuild_parameter") o = args.tup[0];
                  else { o.t = Val::OBJ; o.s = fn.s; }
                  st.push_back(o); } break;
      case 'u': { size_t m = marks.back(); marks.pop_back(); for (size_t k = m; k + 1 < st.size() + 1 && k + 1 <= st.size(); k += 2) if (k + 1 < st.size()) st[m-1].items.push_back({st[k], st[k+1]}); else st[m-1].items.push_back({st[k], st[k]}); st.resize(m); } break;  // SETITEMS
      case 's': { Val v = st.back(); st.pop_back(); Val k = st.back(); st.pop_back(); st.back().items.push_back({k, v}); } break;  // SETITEM
      case 'e': { size_t m = marks.back(); marks.pop_back(); for (size_t k = m; k < st.size(); ++k) st[m-1].tup.push_back(st[k]); st.resize(m); } break;  // APPENDS
      case 'a': { Val v = st.back(); st.pop_back(); st.back().tup.push_back(v); } break;      // APPEND
      case '.': return st.back();                                                            // STOP
      default: printf("ptio: unhandled pickle op 0x%02x at %zu\n", op, ip - 1); std::exit(1);
    }
  }
  return st.empty() ? Val{} : st.back();
}

inline Tensor read_storage(const Zip& z, const Val& ten, const std::string& name) {
  Tensor t; t.name = name; t.shape = ten.tshape;
  int64_t numel = 1; for (int64_t d : ten.tshape) numel *= d;
  auto e = z.find("data/" + ten.tkey); t.data.resize(numel);
  if (ten.tdt == 1) { const uint8_t* p = z.B + e.first; for (int64_t k = 0; k < numel; ++k) { uint16_t h = p[2*k] | (p[2*k+1] << 8); t.data[k] = half_to_float(h); } }
  else std::memcpy(t.data.data(), z.B + e.first, numel * 4);
  return t;
}

// state_dict .pt  (OrderedDict[str -> Tensor])
inline std::vector<Tensor> load_pt(const std::string& path) {
  Zip z; z.open(path); auto pk = z.find("data.pkl"); Val top = run_pickle(z.B + pk.first, pk.second);
  std::vector<Tensor> out;
  for (auto& kv : top.items) if (kv.first.t == Val::S && kv.second.t == Val::TEN && kv.second.tdt != 2) out.push_back(read_storage(z, kv.second, kv.first.s));
  return out;
}

// raw checkpoint .pt  ({'model': nn.Module, ...}) -> state_dict, walking _modules/_parameters/_buffers
inline const Val* get_item(const Val& d, const std::string& key) { for (auto& kv : d.items) if (kv.first.t == Val::S && kv.first.s == key) return &kv.second; return nullptr; }
inline void walk_module(const Zip& z, const Val& mod, const std::string& prefix, std::vector<Tensor>& out) {
  for (const char* sect : {"_parameters", "_buffers"}) { const Val* d = get_item(mod, sect); if (d) for (auto& kv : d->items) if (kv.second.t == Val::TEN && kv.second.tdt != 2) out.push_back(read_storage(z, kv.second, prefix + kv.first.s)); }
  const Val* subs = get_item(mod, "_modules"); if (subs) for (auto& kv : subs->items) if (kv.second.t == Val::OBJ) walk_module(z, kv.second, prefix + kv.first.s + ".", out);
}
inline std::vector<Tensor> load_pt_module(const std::string& path, const std::string& model_key = "model") {
  Zip z; z.open(path); auto pk = z.find("data.pkl"); Val top = run_pickle(z.B + pk.first, pk.second);
  const Val* m = get_item(top, model_key); if (!m || m->t != Val::OBJ) { printf("no '%s' module in %s\n", model_key.c_str(), path.c_str()); std::exit(1); }
  std::vector<Tensor> out; walk_module(z, *m, "", out); return out;
}

}  // namespace pt
