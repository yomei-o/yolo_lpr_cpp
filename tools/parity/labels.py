"""Parity test M1: the C++ and Python label tables must be the same thing, not two things that
look alike. Compares, byte for byte:

  1. `jlpr labels --dump`            vs  `python tools/jlpr.py labels --dump`
     (spec parse + class order + decode + shared splitmix64 RNG)
  2. `jlpr labels --emit-header`     vs  `python tools/jlpr.py labels --emit-header`
     (the embedded-spec header used for WASM builds)

usage: python tools/parity/labels.py [--jlpr <path to jlpr exe>]
"""
import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEC = os.path.join(ROOT, "spec", "labels.txt")


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        print("FAILED to run: %s\n%s" % (" ".join(cmd), r.stderr.decode("utf-8", "replace")))
        sys.exit(2)
    return r.stdout


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            lo = max(0, i - 40)
            return i, a[lo:i + 40], b[lo:i + 40]
    return n, a[n:n + 80], b[n:n + 80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jlpr", default=os.path.join(ROOT, "jlpr.exe"))
    ap.add_argument("--vectors", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    if not os.path.exists(a.jlpr):
        print("no jlpr binary at %s — build it first (build/gcc.sh or build/cc.sh)" % a.jlpr)
        return 2

    py = [sys.executable, os.path.join(ROOT, "tools", "jlpr.py")]
    args = ["labels", "--spec", SPEC, "--vectors", str(a.vectors), "--seed", str(a.seed)]

    c_dump = run([a.jlpr] + args + ["--dump"])
    p_dump = run(py + args + ["--dump"])
    ok = c_dump == p_dump
    print("dump:         %s  (%d bytes C++, %d bytes Python)" %
          ("MATCH" if ok else "DIFFER", len(c_dump), len(p_dump)))
    if not ok:
        i, ca, pa = first_diff(c_dump, p_dump)
        print("  first difference at byte %d\n  C++    : %r\n  Python : %r" % (i, ca, pa))

    with tempfile.TemporaryDirectory() as td:
        ch = os.path.join(td, "c.hpp")
        ph = os.path.join(td, "p.hpp")
        run([a.jlpr, "labels", "--spec", SPEC, "--emit-header", ch])
        run(py + ["labels", "--spec", SPEC, "--emit-header", ph])
        cb, pb = open(ch, "rb").read(), open(ph, "rb").read()
        ok2 = cb == pb
        print("emit-header:  %s  (%d bytes C++, %d bytes Python)" %
              ("MATCH" if ok2 else "DIFFER", len(cb), len(pb)))
        if not ok2:
            i, ca, pa = first_diff(cb, pb)
            print("  first difference at byte %d\n  C++    : %r\n  Python : %r" % (i, ca, pa))

    print("M1 parity: %s" % ("PASS" if (ok and ok2) else "FAIL"))
    return 0 if (ok and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
