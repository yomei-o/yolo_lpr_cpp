// splitmix64 — the ONE random generator this project uses, so that C++ and Python can be
// asked for the same stream and produce the same synthetic data / test vectors.
// Spec (both implementations must match bit for bit):
//   state += 0x9E3779B97F4A7C15
//   z = state; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) * 0x94D049BB133111EB
//   return z ^ (z >> 31)
// uniform integer in [0,n)  = next() % n      (modulo bias is accepted and specified)
// uniform float   in [0,1)  = (next() >> 11) * 2^-53
#pragma once
#include <cstdint>

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed = 0) : s(seed) {}
  uint64_t next() {
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  uint64_t below(uint64_t n) { return next() % n; }            // [0,n)
  double unit() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }  // [0,1)
  double range(double a, double b) { return a + (b - a) * unit(); }
};
