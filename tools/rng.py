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

    def units(self, n):
        """n consecutive unit() values at once, advancing the state by exactly n draws.
        splitmix64's state is seed + k*GOLDEN, so the whole block can be mixed vectorised — this is
        the identical stream, just not one call at a time (the generator needs ~150k draws of
        gaussian noise per image and a Python loop for that is unusable)."""
        import numpy as np
        g = np.uint64(0x9E3779B97F4A7C15)
        k = np.arange(1, n + 1, dtype=np.uint64)
        s = np.uint64(self.s) + k * g
        z = s.copy()
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        self.s = int(s[-1]) if n else self.s
        return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
