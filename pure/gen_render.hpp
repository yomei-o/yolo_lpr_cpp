// Synthetic plate generator — the rendering half (C++ side).
//
// Pipeline for one sample, all at the sample's own resolution so that a small plate really is a
// small plate (upscaling a crisp render would teach the model nothing about 30-pixel plates):
//
//   1. plate texture   : 2 px/mm flat art — rounded body, 地名 / 分類番号 / ひらがな / 一連番号
//                        drawn with stb_truetype, bolts, seal, frame, dirt
//   2. 3D placement    : rotate by yaw/pitch/roll, project with a pinhole so the plate lands
//                        `plate_px` wide, over a car-ish background
//   3. degradation     : brightness / contrast / colour temperature / gaussian blur / motion blur /
//                        gaussian noise
//   4. crop            : bbox of the four projected corners, expanded by `margin` — *the same
//                        framing inference uses* — then resized to out_px and JPEG round-tripped
//
// The 4 corners are recorded in crop pixels, which is exactly what the corner regressor (M6) needs
// and what nobody can label by hand at this volume.
//
// Requires STB_IMAGE_IMPLEMENTATION / STB_IMAGE_WRITE_IMPLEMENTATION / STB_TRUETYPE_IMPLEMENTATION
// in the including .cpp.
#pragma once
#include "gen.hpp"
#include "stb_truetype.h"
// NOTE: stb_image.h / stb_image_write.h are deliberately NOT included here. Their implementation
// blocks sit outside their include guards, so a second #include in the same translation unit
// re-expands them and the build fails on redefinitions. The including .cpp must pull them in first
// (with STB_IMAGE_IMPLEMENTATION / STB_IMAGE_WRITE_IMPLEMENTATION / STB_TRUETYPE_IMPLEMENTATION).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace gen {

static constexpr double PI = 3.14159265358979323846;

// ---- tiny RGB image ---------------------------------------------------------------------------
struct Img {
  int w = 0, h = 0;
  std::vector<unsigned char> d;         // RGB8
  Img() {}
  Img(int W, int H, Rgb fill = {0, 0, 0}) : w(W), h(H), d((size_t)W * H * 3) {
    for (size_t i = 0; i < d.size(); i += 3) { d[i] = (unsigned char)fill.r; d[i + 1] = (unsigned char)fill.g; d[i + 2] = (unsigned char)fill.b; }
  }
  unsigned char* at(int x, int y) { return &d[((size_t)y * w + x) * 3]; }
  const unsigned char* at(int x, int y) const { return &d[((size_t)y * w + x) * 3]; }
  void blend(int x, int y, Rgb c, float a) {
    if (x < 0 || y < 0 || x >= w || y >= h || a <= 0) return;
    unsigned char* p = at(x, y);
    p[0] = (unsigned char)(p[0] + (c.r - p[0]) * a);
    p[1] = (unsigned char)(p[1] + (c.g - p[1]) * a);
    p[2] = (unsigned char)(p[2] + (c.b - p[2]) * a);
  }
  // bilinear sample with edge clamp
  void sample(float x, float y, float out[3]) const {
    int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    float fx = x - x0, fy = y - y0;
    auto px = [&](int xx, int yy, int c) {
      xx = std::clamp(xx, 0, w - 1); yy = std::clamp(yy, 0, h - 1);
      return (float)at(xx, yy)[c];
    };
    for (int c = 0; c < 3; ++c)
      out[c] = px(x0, y0, c) * (1 - fx) * (1 - fy) + px(x0 + 1, y0, c) * fx * (1 - fy)
             + px(x0, y0 + 1, c) * (1 - fx) * fy + px(x0 + 1, y0 + 1, c) * fx * fy;
  }
};

// ---- fonts ------------------------------------------------------------------------------------
struct Font {
  std::vector<unsigned char> blob;
  stbtt_fontinfo info{};
  std::string name;
  bool ok = false;
};

inline bool load_font(const std::string& path, Font& f) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return false;
  std::streamsize n = in.tellg();
  in.seekg(0);
  f.blob.resize((size_t)n);
  in.read((char*)f.blob.data(), n);
  int off = stbtt_GetFontOffsetForIndex(f.blob.data(), 0);
  if (off < 0) return false;
  if (!stbtt_InitFont(&f.info, f.blob.data(), off)) return false;
  f.name = path.substr(path.find_last_of("/\\") + 1);
  f.ok = true;
  return true;
}

// The font list is part of the parity contract: same directory, same sorted order, same index draw.
inline std::vector<std::string> font_files(const std::string& dir, const std::string& only = "") {
  static const char* candidates[] = {"GenSenRounded2-B.ttc", "YuGothB.ttc", "meiryo.ttc", "msgothic.ttc"};
  std::vector<std::string> out;
  for (const char* c : candidates) {
    if (!only.empty() && only != c) continue;      // --font pins one face (for A/B tests)
    std::string p = dir + "/" + c;
    std::ifstream in(p, std::ios::binary);
    if (in) out.push_back(p);
  }
  return out;
}

// UTF-8 -> codepoints
inline std::vector<int> utf8_codepoints(const std::string& s) {
  std::vector<int> cps;
  for (size_t i = 0; i < s.size();) {
    unsigned char c = (unsigned char)s[i];
    int cp = c, len = 1;
    if (c >= 0xF0) { cp = c & 0x07; len = 4; }
    else if (c >= 0xE0) { cp = c & 0x0F; len = 3; }
    else if (c >= 0xC0) { cp = c & 0x1F; len = 2; }
    for (int k = 1; k < len && i + k < s.size(); ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
    cps.push_back(cp);
    i += len;
  }
  return cps;
}

// Draw `text` so that its glyph box is `px_h` tall, centred on (cx, baseline_y is derived).
// Returns the drawn width. `track` widens the letter spacing (plate lettering is airy).
inline float draw_text(Img& im, const Font& f, const std::string& text, float cx, float cy,
                       float px_h, Rgb colour, float track = 0.06f, float max_w = 0.f) {
  if (!f.ok) return 0;
  float scale = stbtt_ScaleForPixelHeight(&f.info, px_h);
  std::vector<int> cps = utf8_codepoints(text);
  // measure
  float total = 0;
  std::vector<float> adv(cps.size(), 0);
  for (size_t i = 0; i < cps.size(); ++i) {
    int a = 0, lsb = 0;
    stbtt_GetCodepointHMetrics(&f.info, cps[i], &a, &lsb);
    adv[i] = a * scale * (1.0f + track);
    total += adv[i];
  }
  if (max_w > 0 && total > max_w) {          // 4-char region names are squeezed on real plates too
    float k2 = max_w / total;
    scale *= k2;
    for (float& v : adv) v *= k2;
    total = max_w;
  }
  float x = cx - total / 2;
  int asc = 0, desc = 0, gap = 0;
  stbtt_GetFontVMetrics(&f.info, &asc, &desc, &gap);
  float baseline = cy + (asc + desc) * 0.5f * scale;      // vertically centre the em box on cy
  // Japanese plates have RAISED characters: light from above leaves a highlight on the upper-left
  // edge and a shadow on the lower-right. Faking that costs two extra blits and is most of what
  // makes a synthetic plate stop looking like a screenshot of a font.
  const int e = std::max(1, (int)std::round(px_h * 0.035f));
  for (size_t i = 0; i < cps.size(); ++i) {
    int bw = 0, bh = 0, xo = 0, yo = 0;
    unsigned char* bmp = stbtt_GetCodepointBitmap(&f.info, scale, scale, cps[i], &bw, &bh, &xo, &yo);
    if (bmp) {
      int gx = (int)std::round(x + (adv[i] - bw) * 0.5f);   // centre the glyph in its advance
      int gy = (int)std::round(baseline + yo);
      for (int yy = 0; yy < bh; ++yy)
        for (int xx = 0; xx < bw; ++xx) {
          float a = bmp[yy * bw + xx] / 255.0f;
          if (a <= 0.02f) continue;
          im.blend(gx + xx - e, gy + yy - e, {255, 255, 255}, a * 0.30f);   // highlight
          im.blend(gx + xx + e, gy + yy + e, {0, 0, 0}, a * 0.30f);         // shadow
        }
      for (int yy = 0; yy < bh; ++yy)
        for (int xx = 0; xx < bw; ++xx)
          im.blend(gx + xx, gy + yy, colour, bmp[yy * bw + xx] / 255.0f);
      stbtt_FreeBitmap(bmp, nullptr);
    }
    x += adv[i];
  }
  return total;
}

inline void fill_rounded(Img& im, float x0, float y0, float x1, float y1, float r, Rgb c, float alpha = 1.f) {
  for (int y = (int)std::floor(y0); y <= (int)std::ceil(y1); ++y)
    for (int x = (int)std::floor(x0); x <= (int)std::ceil(x1); ++x) {
      float dx = std::max({x0 + r - (float)x, 0.f, (float)x - (x1 - r)});
      float dy = std::max({y0 + r - (float)y, 0.f, (float)y - (y1 - r)});
      float d = std::sqrt(dx * dx + dy * dy);
      float a = std::clamp(r - d + 0.5f, 0.f, 1.f);
      if (dx == 0 && dy == 0) a = 1.f;
      im.blend(x, y, c, a * alpha);
    }
}

inline void fill_circle(Img& im, float cx, float cy, float r, Rgb c) {
  for (int y = (int)(cy - r - 1); y <= (int)(cy + r + 1); ++y)
    for (int x = (int)(cx - r - 1); x <= (int)(cx + r + 1); ++x) {
      float d = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
      im.blend(x, y, c, std::clamp(r - d + 0.5f, 0.f, 1.f));
    }
}

// ---- plate texture (flat art, 2 px/mm) -------------------------------------------------------
inline Img plate_texture(const Params& p, const spec::Spec& sp, const std::vector<Font>& fonts,
                         int font_idx, Rng& rng) {
  // Resolution follows how big the plate will actually appear: drawing 660x330 art for a plate that
  // lands 30 px wide is ~20x wasted work, and the downsampling throws the detail away anyway. Two
  // texture pixels per destination pixel is enough for clean bilinear minification.
  const float mm = std::clamp((float)(2.0 * p.plate_px / 330.0), 0.5f, 2.5f);   // px per mm
  const float W_mm = p.large ? 440.f : 330.f, H_mm = p.large ? 220.f : 165.f;
  const int W = (int)(W_mm * mm), H = (int)(H_mm * mm);
  Img im(W, H, {90, 90, 90});
  const float r = (p.large ? 13.f : 10.f) * mm;

  // body
  fill_rounded(im, 2, 2, (float)W - 3, (float)H - 3, r, p.bg);
  if (p.design) {                                          // 図柄入り: 模倣柄（実物の図柄は使わない）
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        float t = (float)y / H;
        Rgb tint{(int)(200 + 40 * t), (int)(225 - 30 * t), (int)(235 - 10 * t)};
        im.blend(x, y, tint, 0.35f * (0.4f + 0.6f * t));
      }
    for (int k = 0; k < 6; ++k) {                          // ぼんやりした図形
      float cx = (float)rng.range(0.1, 0.9) * W, cy = (float)rng.range(0.15, 0.85) * H;
      float rr = (float)rng.range(0.05, 0.16) * W;
      Rgb c{(int)rng.range(120, 220), (int)rng.range(160, 230), (int)rng.range(150, 210)};
      fill_circle(im, cx, cy, rr, c);
    }
    fill_rounded(im, 2, 2, (float)W - 3, (float)H - 3, r, {255, 255, 255}, 0.55f);  // 文字が読める程度に白を重ねる
  }

  const Font& f = fonts[font_idx];
  auto X = [&](float x_mm) { return x_mm * mm * (p.large ? 440.f / 330.f : 1.f); };
  auto Y = [&](float y_mm) { return y_mm * mm * (p.large ? 220.f / 165.f : 1.f); };

  // フレームは文字より先に描く（後から描くと本体を塗り直して文字が消える）
  if (p.frame) {
    fill_rounded(im, 0, 0, (float)W - 1, (float)H - 1, r, {40, 40, 45});
    fill_rounded(im, Y(5), Y(5), (float)W - 1 - Y(5), (float)H - 1 - Y(5), r, p.bg);
  }

  // 上段: 地名（左）と分類番号（右）
  const spec::Group& reg = sp.head("region");
  draw_text(im, f, reg.tok[p.region], X(95), Y(40), Y(40), p.fg, 0.02f, X(150));
  draw_text(im, f, p.cls_text, X(245), Y(40), Y(42), p.fg, 0.08f, X(140));

  // 下段: ひらがな（左）と一連番号（右、2桁ずつハイフン区切り、空き桁は ・）
  const spec::Group& hira = sp.head("hiragana");
  draw_text(im, f, hira.tok[p.hira], X(40), Y(118), Y(58), p.fg, 0.f);
  std::string left, right;
  {
    std::string s4 = p.serial_text;
    while (s4.size() < 4) s4 = "\xE3\x83\xBB" + s4;         // '・' for the blank slots
    // split into two visual halves: the first two glyphs and the last two
    std::vector<int> cps = utf8_codepoints(s4);
    auto put = [&](int cp) {
      std::string o;
      if (cp < 0x80) o += (char)cp;
      else if (cp == 0x30FB) o += "\xE3\x83\xBB";
      return o;
    };
    left = put(cps[0]) + put(cps[1]);
    right = put(cps[2]) + put(cps[3]);
  }
  draw_text(im, f, left, X(140), Y(118), Y(86), p.fg, 0.10f);
  fill_rounded(im, X(196), Y(115), X(214), Y(122), 1.f, p.fg);     // ハイフン
  draw_text(im, f, right, X(270), Y(118), Y(86), p.fg, 0.10f);

  // ボルト・封印・フレーム
  if (p.bolts >= 2) {
    fill_circle(im, X(60), Y(18), Y(7), {160, 160, 165});
    fill_circle(im, X(270), Y(18), Y(7), {160, 160, 165});
  }
  if (p.bolts == 4) {
    fill_circle(im, X(60), Y(150), Y(7), {160, 160, 165});
    fill_circle(im, X(270), Y(150), Y(7), {160, 160, 165});
  }
  if (p.seal) {
    fill_circle(im, X(60), Y(18), Y(9), {70, 110, 90});
    fill_circle(im, X(60), Y(18), Y(5), {200, 210, 200});
  }
  // 光沢: 斜めの明るい帯（実物は塗装の反射がある）
  {
    float gx0 = (float)rng.range(-0.3, 0.6) * W, gw = (float)rng.range(0.25, 0.7) * W;
    float strength = (float)rng.range(0.05, 0.22);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        float t = ((float)x + 0.6f * y - gx0) / gw;
        if (t < 0 || t > 1) continue;
        float a = strength * std::sin(t * PI);
        im.blend(x, y, {255, 255, 255}, a);
      }
  }

  // 汚れ・かすれ
  if (p.dirt > 0.01) {
    int blobs = (int)(p.dirt * 40);
    for (int k = 0; k < blobs; ++k) {
      float cx = (float)rng.range(0, 1) * W, cy = (float)rng.range(0, 1) * H;
      float rr = (float)rng.range(0.01, 0.06) * W;
      float a = (float)(p.dirt * rng.range(0.15, 0.5));
      Rgb c{(int)rng.range(60, 140), (int)rng.range(60, 130), (int)rng.range(50, 120)};
      for (int y = (int)(cy - rr); y <= (int)(cy + rr); ++y)
        for (int x = (int)(cx - rr); x <= (int)(cx + rr); ++x) {
          float d = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
          if (d <= rr) im.blend(x, y, c, a * (1.f - d / rr));
        }
    }
  }
  return im;
}

// ---- 3D placement + projection ---------------------------------------------------------------
// double, not float: these numbers are written into corners.txt and compared with the Python
// generator's, and at 1e-7 a float would round differently in the 5th decimal.
struct Quad { double x[4], y[4]; };     // TL, TR, BR, BL

inline Quad project_plate(const Params& p, double cx, double cy) {
  const float W_mm = p.large ? 440.f : 330.f, H_mm = p.large ? 220.f : 165.f;
  const double ry = p.yaw * PI / 180, rx = p.pitch * PI / 180, rz = p.roll * PI / 180;
  double c3[4][3];
  const double hx = W_mm / 2, hy = H_mm / 2;
  const double src[4][3] = {{-hx, -hy, 0}, {hx, -hy, 0}, {hx, hy, 0}, {-hx, hy, 0}};
  for (int i = 0; i < 4; ++i) {
    double X = src[i][0], Y = src[i][1], Z = src[i][2];
    double x1 = X * std::cos(ry) + Z * std::sin(ry), z1 = -X * std::sin(ry) + Z * std::cos(ry);
    double y2 = Y * std::cos(rx) - z1 * std::sin(rx), z2 = Y * std::sin(rx) + z1 * std::cos(rx);
    double x3 = x1 * std::cos(rz) - y2 * std::sin(rz), y3 = x1 * std::sin(rz) + y2 * std::cos(rz);
    c3[i][0] = x3; c3[i][1] = y3; c3[i][2] = z2;
  }
  const double dist = W_mm * 3.0;                          // camera distance in mm
  double f = 1.0;
  // choose f so the projected width matches plate_px
  double w_proj = 0;
  for (int pass = 0; pass < 2; ++pass) {
    double xs[4], ys[4];
    for (int i = 0; i < 4; ++i) {
      double z = dist + c3[i][2];
      xs[i] = f * c3[i][0] / z;
      ys[i] = f * c3[i][1] / z;
    }
    w_proj = std::max(std::fabs(xs[1] - xs[0]), std::fabs(xs[2] - xs[3]));
    if (pass == 0) f = p.plate_px / std::max(1e-6, w_proj);
  }
  Quad q;
  for (int i = 0; i < 4; ++i) {
    double z = dist + c3[i][2];
    q.x[i] = cx + f * c3[i][0] / z;
    q.y[i] = cy + f * c3[i][1] / z;
  }
  return q;
}

// Homography mapping the destination quad back to texture coords (solve 8x8 by Gauss).
inline bool quad_to_texture_h(const Quad& q, float tw, float th, double H[9]) {
  const double dx[4] = {q.x[0], q.x[1], q.x[2], q.x[3]};
  const double dy[4] = {q.y[0], q.y[1], q.y[2], q.y[3]};
  const double sxp[4] = {0, (double)tw, (double)tw, 0};
  const double syp[4] = {0, 0, (double)th, (double)th};
  double A[8][9] = {};
  for (int i = 0; i < 4; ++i) {
    A[2 * i][0] = dx[i]; A[2 * i][1] = dy[i]; A[2 * i][2] = 1;
    A[2 * i][6] = -dx[i] * sxp[i]; A[2 * i][7] = -dy[i] * sxp[i]; A[2 * i][8] = sxp[i];
    A[2 * i + 1][3] = dx[i]; A[2 * i + 1][4] = dy[i]; A[2 * i + 1][5] = 1;
    A[2 * i + 1][6] = -dx[i] * syp[i]; A[2 * i + 1][7] = -dy[i] * syp[i]; A[2 * i + 1][8] = syp[i];
  }
  for (int c = 0; c < 8; ++c) {                            // Gaussian elimination with pivoting
    int piv = c;
    for (int r2 = c + 1; r2 < 8; ++r2) if (std::fabs(A[r2][c]) > std::fabs(A[piv][c])) piv = r2;
    if (std::fabs(A[piv][c]) < 1e-12) return false;
    if (piv != c) for (int k = 0; k < 9; ++k) std::swap(A[c][k], A[piv][k]);
    for (int r2 = 0; r2 < 8; ++r2) {
      if (r2 == c) continue;
      double m = A[r2][c] / A[c][c];
      for (int k = c; k < 9; ++k) A[r2][k] -= m * A[c][k];
    }
  }
  for (int i = 0; i < 8; ++i) H[i] = A[i][8] / A[i][i];
  H[8] = 1.0;
  return true;
}

inline void gaussian_blur(Img& im, float sigma);      // defined below; used by synth_background

// ---- reusable pieces for compositing (used by both the crop generator and gen-det) ----------
// Paste `tex` into `canvas` through the quad, sampling the texture per destination pixel.
inline void paste_textured_quad(Img& canvas, const Img& tex, const Quad& q) {
  double H[9];
  if (!quad_to_texture_h(q, (float)tex.w, (float)tex.h, H)) return;
  double minx = q.x[0], maxx = q.x[0], miny = q.y[0], maxy = q.y[0];
  for (int i = 1; i < 4; ++i) {
    minx = std::min(minx, q.x[i]); maxx = std::max(maxx, q.x[i]);
    miny = std::min(miny, q.y[i]); maxy = std::max(maxy, q.y[i]);
  }
  for (int y = std::max(0, (int)std::floor(miny)); y <= std::min(canvas.h - 1, (int)std::ceil(maxy)); ++y)
    for (int x = std::max(0, (int)std::floor(minx)); x <= std::min(canvas.w - 1, (int)std::ceil(maxx)); ++x) {
      double w = H[6] * x + H[7] * y + H[8];
      if (std::fabs(w) < 1e-9) continue;
      double u = (H[0] * x + H[1] * y + H[2]) / w;
      double v = (H[3] * x + H[4] * y + H[5]) / w;
      if (u < 0 || v < 0 || u > tex.w - 1 || v > tex.h - 1) continue;
      float c[3];
      tex.sample((float)u, (float)v, c);
      unsigned char* d = canvas.at(x, y);
      for (int k = 0; k < 3; ++k) d[k] = (unsigned char)std::clamp(c[k], 0.f, 255.f);
    }
}

// A car-ish background: hue/darkness base, soft blobs, then blur.
inline void synth_background(Img& im, double bg_hue, double bg_dark, Rng& rng) {
  double hue = bg_hue * 6.0;
  int i = (int)hue;
  double f = hue - i, v = bg_dark, s = 0.35;
  double pv = v * (1 - s), qv = v * (1 - s * f), tv = v * (1 - s * (1 - f));
  double rr, gg, bb;
  switch (i % 6) {
    case 0: rr = v; gg = tv; bb = pv; break;
    case 1: rr = qv; gg = v; bb = pv; break;
    case 2: rr = pv; gg = v; bb = tv; break;
    case 3: rr = pv; gg = qv; bb = v; break;
    case 4: rr = tv; gg = pv; bb = v; break;
    default: rr = v; gg = pv; bb = qv; break;
  }
  for (int y = 0; y < im.h; ++y) {
    float shade = 0.75f + 0.5f * y / im.h;
    for (int x = 0; x < im.w; ++x) {
      unsigned char* d = im.at(x, y);
      d[0] = (unsigned char)std::clamp(rr * 255 * shade, 0.0, 255.0);
      d[1] = (unsigned char)std::clamp(gg * 255 * shade, 0.0, 255.0);
      d[2] = (unsigned char)std::clamp(bb * 255 * shade, 0.0, 255.0);
    }
  }
  int blobs = 14;
  for (int k = 0; k < blobs; ++k) {
    float bx = (float)rng.range(-0.1, 1.1) * im.w, by = (float)rng.range(-0.1, 1.1) * im.h;
    float br = (float)rng.range(0.08, 0.45) * im.w;
    float a = (float)rng.range(0.05, 0.25);
    Rgb c{(int)rng.range(0, 255), (int)rng.range(0, 255), (int)rng.range(0, 255)};
    c.r = (int)(0.6 * c.r + 0.4 * rr * 255); c.g = (int)(0.6 * c.g + 0.4 * gg * 255);
    c.b = (int)(0.6 * c.b + 0.4 * bb * 255);
    for (int y = std::max(0, (int)(by - br)); y <= std::min(im.h - 1, (int)(by + br)); ++y)
      for (int x = std::max(0, (int)(bx - br)); x <= std::min(im.w - 1, (int)(bx + br)); ++x) {
        float dd = std::sqrt((x - bx) * (x - bx) + (y - by) * (y - by));
        if (dd <= br) im.blend(x, y, c, a * (1.f - dd / br) * (1.f - dd / br));
      }
  }
  gaussian_blur(im, std::max(1.f, im.w * 0.02f));
}

// Render one plate at `plate_px` wide, centred on (cx,cy) of `canvas`; returns its 4 corners.
inline Quad paste_plate(Img& canvas, const Params& p_in, const spec::Spec& sp,
                        const std::vector<Font>& fonts, int font_idx, double cx, double cy,
                        double plate_px, Rng& rng) {
  Params p = p_in;
  p.plate_px = plate_px;
  Img tex = plate_texture(p, sp, fonts, font_idx, rng);
  Quad q = project_plate(p, cx, cy);
  paste_textured_quad(canvas, tex, q);
  return q;
}

// ---- degradation --------------------------------------------------------------------------
inline void gaussian_blur(Img& im, float sigma) {
  if (sigma < 0.05f) return;
  int r = std::max(1, (int)std::ceil(sigma * 2.5f));
  std::vector<float> k(2 * r + 1);
  float s = 0;
  for (int i = -r; i <= r; ++i) { k[i + r] = std::exp(-(i * i) / (2 * sigma * sigma)); s += k[i + r]; }
  for (float& v : k) v /= s;
  Img tmp = im;
  for (int y = 0; y < im.h; ++y)                            // horizontal
    for (int x = 0; x < im.w; ++x) {
      float acc[3] = {0, 0, 0};
      for (int i = -r; i <= r; ++i) {
        const unsigned char* p = tmp.at(std::clamp(x + i, 0, im.w - 1), y);
        for (int c = 0; c < 3; ++c) acc[c] += k[i + r] * p[c];
      }
      for (int c = 0; c < 3; ++c) im.at(x, y)[c] = (unsigned char)std::clamp(acc[c], 0.f, 255.f);
    }
  tmp = im;
  for (int y = 0; y < im.h; ++y)                            // vertical
    for (int x = 0; x < im.w; ++x) {
      float acc[3] = {0, 0, 0};
      for (int i = -r; i <= r; ++i) {
        const unsigned char* p = tmp.at(x, std::clamp(y + i, 0, im.h - 1));
        for (int c = 0; c < 3; ++c) acc[c] += k[i + r] * p[c];
      }
      for (int c = 0; c < 3; ++c) im.at(x, y)[c] = (unsigned char)std::clamp(acc[c], 0.f, 255.f);
    }
}

inline void motion_blur(Img& im, float len) {
  int n = (int)std::round(len);
  if (n < 1) return;
  Img tmp = im;
  for (int y = 0; y < im.h; ++y)
    for (int x = 0; x < im.w; ++x) {
      float acc[3] = {0, 0, 0};
      for (int i = 0; i <= n; ++i) {
        const unsigned char* p = tmp.at(std::clamp(x - n / 2 + i, 0, im.w - 1), y);
        for (int c = 0; c < 3; ++c) acc[c] += p[c];
      }
      for (int c = 0; c < 3; ++c) im.at(x, y)[c] = (unsigned char)std::clamp(acc[c] / (n + 1), 0.f, 255.f);
    }
}

inline void photometric(Img& im, const Params& p, Rng& rng) {
  for (int y = 0; y < im.h; ++y)
    for (int x = 0; x < im.w; ++x) {
      unsigned char* q = im.at(x, y);
      for (int c = 0; c < 3; ++c) {
        float v = q[c] / 255.f;
        v = (v - 0.5f) * (float)p.contrast + 0.5f;
        v *= (float)p.brightness;
        if (c == 0) v *= (float)(1.0 + p.warm);
        if (c == 2) v *= (float)(1.0 - p.warm);
        q[c] = (unsigned char)std::clamp(v * 255.f, 0.f, 255.f);
      }
    }
  if (p.noise > 0.1) {
    for (size_t i = 0; i < im.d.size(); ++i) {
      double u1 = std::max(1e-9, rng.unit()), u2 = rng.unit();
      double g = std::sqrt(-2 * std::log(u1)) * std::cos(2 * PI * u2);    // Box-Muller
      im.d[i] = (unsigned char)std::clamp((double)im.d[i] + g * p.noise, 0.0, 255.0);
    }
  }
}

// ---- one sample -------------------------------------------------------------------------------
struct Rendered {
  Img crop;
  float corners[8] = {};                 // TL,TR,BR,BL in crop pixels
  std::string font;
};

inline Rendered render(const Params& p, const spec::Spec& sp, const std::vector<Font>& fonts,
                      int font_idx, int out_px, Rng& rng) {
  Rendered R;
  R.font = fonts[font_idx].name;
  Img tex = plate_texture(p, sp, fonts, font_idx, rng);

  const int canvas = std::max(64, (int)std::ceil(p.plate_px * 2.4));
  const double cx = canvas * 0.5, cy = canvas * 0.5;
  Quad q = project_plate(p, cx, cy);

  Img im(canvas, canvas, {0, 0, 0});
  synth_background(im, p.bg_hue, p.bg_dark, rng);

  paste_textured_quad(im, tex, q);

  photometric(im, p, rng);
  gaussian_blur(im, (float)p.blur);
  motion_blur(im, (float)p.motion);

  // crop: bbox of the corners expanded by margin, offset a little, then resize to out_px
  double minx = q.x[0], maxx = q.x[0], miny = q.y[0], maxy = q.y[0];
  for (int i = 1; i < 4; ++i) {
    minx = std::min(minx, q.x[i]); maxx = std::max(maxx, q.x[i]);
    miny = std::min(miny, q.y[i]); maxy = std::max(maxy, q.y[i]);
  }
  double bw = maxx - minx, bh = maxy - miny;
  double x0 = minx - bw * p.margin + bw * p.off_x;
  double y0 = miny - bh * p.margin + bh * p.off_y;
  double x1 = maxx + bw * p.margin + bw * p.off_x;
  double y1 = maxy + bh * p.margin + bh * p.off_y;

  R.crop = Img(out_px, out_px, {0, 0, 0});
  for (int y = 0; y < out_px; ++y)
    for (int x = 0; x < out_px; ++x) {
      float sx = x0 + (x + 0.5f) * (x1 - x0) / out_px - 0.5f;
      float sy = y0 + (y + 0.5f) * (y1 - y0) / out_px - 0.5f;
      float c[3];
      im.sample(sx, sy, c);
      unsigned char* d = R.crop.at(x, y);
      for (int k = 0; k < 3; ++k) d[k] = (unsigned char)std::clamp(c[k], 0.f, 255.f);
    }
  for (int i = 0; i < 4; ++i) {
    R.corners[2 * i] = (float)((q.x[i] - x0) * out_px / (x1 - x0));
    R.corners[2 * i + 1] = (float)((q.y[i] - y0) * out_px / (y1 - y0));
  }

  // JPEG round trip, so the model sees the artefacts a phone/camera adds
  if (p.jpeg_q < 100) {
    std::vector<unsigned char> buf;
    auto wr = [](void* ctx, void* data, int size) {
      auto* v = (std::vector<unsigned char>*)ctx;
      v->insert(v->end(), (unsigned char*)data, (unsigned char*)data + size);
    };
    if (stbi_write_jpg_to_func(wr, &buf, R.crop.w, R.crop.h, 3, R.crop.d.data(), p.jpeg_q) && !buf.empty()) {
      int w2 = 0, h2 = 0, c2 = 0;
      unsigned char* dec = stbi_load_from_memory(buf.data(), (int)buf.size(), &w2, &h2, &c2, 3);
      if (dec && w2 == R.crop.w && h2 == R.crop.h) std::memcpy(R.crop.d.data(), dec, R.crop.d.size());
      if (dec) stbi_image_free(dec);
    }
  }
  return R;
}

}  // namespace gen
