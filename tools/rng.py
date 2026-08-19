"""splitmix64 — the ONE random generator this project uses. Mirrors pure/rng.hpp bit for bit,
so C++ and Python can be asked for the same stream (synthetic data, test vectors)."""

M64 = (1 << 64) - 1


class Rng:
    def __init__(self, seed=0):
        self.s = seed & M64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def below(self, n):
        return self.next() % n

    def unit(self):
        return (self.next() >> 11) * (1.0 / 9007199254740992.0)

    def range(self, a, b):
        return a + (b - a) * self.unit()
