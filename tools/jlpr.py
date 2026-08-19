"""jlpr — the one CLI for this project (Python side). Same subcommands and same flag names as the
C++ `jlpr` (pure/jlpr.cpp); whatever one can do, the other must be able to do too.

  python tools/jlpr.py labels [--spec spec/labels.txt] [--dump | --emit-header out.hpp]
                              [--vectors N] [--seed S]
  python tools/jlpr.py export --ocr <ref_dir> --out models/plate_ocr.onnx
  python tools/jlpr.py detect --img <file> [--det <onnx>] [--ocr <onnx>] [--out out.png]
  python tools/jlpr.py gen | train | val        (later milestones)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labels as L  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cmd_labels(a):
    sp = L.load(a.spec)
    if a.emit_header:
        n = L.emit_header(a.spec, a.emit_header)
        print("wrote %s (%d bytes)" % (a.emit_header, n))
        return 0
    out = L.canonical_dump(sp) + L.decode_vectors_dump(sp, a.seed, a.vectors)
    sys.stdout.buffer.write(out.encode("utf-8"))   # bytes, so the dump matches C++ exactly
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="jlpr.py", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("labels")
    q.add_argument("--spec", default=os.path.join(ROOT, "spec", "labels.txt"))
    q.add_argument("--dump", action="store_true")
    q.add_argument("--emit-header", dest="emit_header", default="")
    q.add_argument("--vectors", type=int, default=8)
    q.add_argument("--seed", type=int, default=12345)
    q.set_defaults(fn=cmd_labels)

    # detect is handled by tools/infer.py, which takes the same flags as `jlpr detect`
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "detect":
        import infer
        return infer.main(argv[1:])

    for name in ("export", "gen", "train", "val"):
        s = sub.add_parser(name)
        s.set_defaults(fn=lambda a, name=name: (print("jlpr.py: '%s' is not implemented yet" % name), 1)[1])
        s.add_argument("rest", nargs="*")

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
