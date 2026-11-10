// Result overlay drawn in C++: a frame around each plate and its 4-digit number above it.
// The CLI (`jlpr detect --out`) and the WASM demo use the same code, so the PNG the CLI writes
// and what the browser shows are the same drawing.
//
// No font file is needed: the label only ever holds digits, '-', '.', '?', so a 5x7 bitmap
// scaled by an integer factor is enough and stays sharp at any size.
//
// Works on RGB (ch=3, blended onto the image) and RGBA (ch=4, written as-is — the WASM page
// draws into a transparent layer laid over the live camera preview).
#pragma once
#include "spec.hpp"
#include <algorithm>
#include <cstdint>
#include <string>

namespace ov {

struct Rgb { unsigned char r, g, b; };
inline constexpr Rgb kRed{255, 60, 60};       // detected / still reading
inline constexpr Rgb kGreen{30, 200, 140};    // tracker settled on this reading

// 5x7 glyphs, one byte per row, bit 4 = leftmost pixel.
inline const uint8_t* glyph(char c) {
  static const uint8_t g[][7] = {
      {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},   // 0
      {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},   // 1
      {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},   // 2
      {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E},   // 3
      {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},   // 4
      {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},   // 5
      {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E},   // 6
      {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},   // 7
      {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},   // 8
      {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C},   // 9
      {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00},   // -
      {0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00, 0x00},   // .  (the plate's '・' padding)
      {0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04},   // ?
  };
  if (c >= '0' && c <= '9') return g[c - '0'];
  if (c == '-') return g[10];
  if (c == '.') return g[11];
  return g[12];
}

struct Canvas {
  unsigned char* px;
  int w, h, ch;   // ch = 3 (RGB, blend) or 4 (RGBA, overwrite)
  void put(int x, int y, Rgb c, unsigned char a = 255) const {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    unsigned char* q = px + ((size_t)y * w + x) * ch;
    if (ch == 4) { q[0] = c.r; q[1] = c.g; q[2] = c.b; q[3] = a; return; }
    q[0] = (unsigned char)((c.r * a + q[0] * (255 - a)) / 255);
    q[1] = (unsigned char)((c.g * a + q[1] * (255 - a)) / 255);
    q[2] = (unsigned char)((c.b * a + q[2] * (255 - a)) / 255);
  }
  void fill(int x0, int y0, int x1, int y1, Rgb c, unsigned char a = 255) const {
    for (int y = std::max(0, y0); y < std::min(h, y1); ++y)
      for (int x = std::max(0, x0); x < std::min(w, x1); ++x) put(x, y, c, a);
  }
};

inline void rect(const Canvas& cv, int x1, int y1, int x2, int y2, Rgb c, int t) {
  cv.fill(x1 - t, y1 - t, x2 + t + 1, y1, c);          // outside the box, so the plate stays visible
  cv.fill(x1 - t, y2 + 1, x2 + t + 1, y2 + t + 1, c);
  cv.fill(x1 - t, y1, x1, y2 + 1, c);
  cv.fill(x2 + 1, y1, x2 + t + 1, y2 + 1, c);
}

// What goes on the label: the 4-digit part of the plate as written on it ("35-91", "..12").
// The '・' padding of a short number becomes '.', an unknown head becomes '?'.
inline std::string number_label(const spec::Plate& p) {
  std::string out;
  const std::string& d = p.disp;
  for (size_t i = 0; i < d.size();) {
    unsigned char c = (unsigned char)d[i];
    if (c >= '0' && c <= '9') { out += (char)c; ++i; }
    else if (d.compare(i, 3, "\xE3\x83\xBB") == 0) { out += '.'; i += 3; }   // U+30FB '・'
    else { out += '?'; i += (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4; }
  }
  // a full 4-digit number is written with a hyphen on the plate: 3591 -> 35-91
  if (out.size() == 4 && std::all_of(out.begin(), out.end(), [](char c) { return c >= '0' && c <= '9'; }))
    out.insert(2, "-");
  return out;
}

// Size of things relative to the frame, so a 640 px and a 1920 px frame look alike.
inline int line_px(int W, int H) { return std::max(2, std::min(W, H) / 160); }
inline int text_scale(int W, int H) { return std::max(2, std::min(W, H) / 120); }

inline void text(const Canvas& cv, int x, int y, const std::string& s, int sc, Rgb c) {
  for (size_t k = 0; k < s.size(); ++k) {
    const uint8_t* g = glyph(s[k]);
    for (int r = 0; r < 7; ++r)
      for (int b = 0; b < 5; ++b)
        if (g[r] & (0x10 >> b)) cv.fill(x + (int)k * 6 * sc + b * sc, y + r * sc,
                                        x + (int)k * 6 * sc + (b + 1) * sc, y + (r + 1) * sc, c);
  }
}

// One plate: the frame, and the number on a dark tag above it (below it when there is no room).
inline void plate(const Canvas& cv, float bx1, float by1, float bx2, float by2,
                  const std::string& label, Rgb c) {
  const int t = line_px(cv.w, cv.h), sc = text_scale(cv.w, cv.h);
  const int x1 = (int)bx1, y1 = (int)by1, x2 = (int)bx2, y2 = (int)by2;
  rect(cv, x1, y1, x2, y2, c, t);
  if (label.empty()) return;
  const int pad = sc * 2;
  const int tw = (int)label.size() * 6 * sc - sc + pad * 2, th = 7 * sc + pad * 2;
  int tx = std::clamp(x1 - t, 0, std::max(0, cv.w - tw));
  int ty = y1 - t - th;
  if (ty < 0) ty = std::min(y2 + t + 1, cv.h - th);
  cv.fill(tx, ty, tx + tw, ty + th, Rgb{0, 0, 0}, 190);
  text(cv, tx + pad, ty + pad, label, sc, c);
}

}  // namespace ov
