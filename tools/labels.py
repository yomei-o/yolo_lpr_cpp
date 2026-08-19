"""Label tables read at run time from spec/labels.txt — the single source of truth shared with
pure/spec.hpp. Nothing here hardcodes a class list, so C++ and Python cannot drift.

The canonical_dump() / decode_vectors_dump() output must be byte-identical to the C++ side
(`jlpr labels --dump`); tools/parity/labels.py asserts exactly that.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rng import Rng  # noqa: E402

SPEC_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "spec", "labels.txt")


class Group:
    def __init__(self, kind, name, n):
        self.kind, self.name, self.n = kind, name, n
        self.tok = []


class Spec:
    def __init__(self):
        self.version = 0
        self.groups = []

    def find(self, name):
        for g in self.groups:
            if g.name == name:
                return g
        return None

    def head(self, name):
        g = self.find(name)
        if g is None:
            raise KeyError("spec: no such group %r" % name)
        return g

    def index_of(self, group, token):
        g = self.find(group)
        if g is None:
            return -1
        return g.tok.index(token) if token in g.tok else -1

    def of_kind(self, kind):
        return [g for g in self.groups if g.kind == kind]


def parse(text):
    sp = Spec()
    cur = None
    for line in text.split("\n"):
        if line.endswith("\r"):
            line = line[:-1]
        if line.startswith("# version "):
            sp.version = int(line[len("# version "):].split()[0])
        tab = line.find("\t")
        if tab >= 0 and "#" in line[tab:]:
            line = line[:tab]
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        if t.startswith("["):
            kind, name, n = t[1:t.index("]")].split()
            cur = Group(kind, name, int(n))
            sp.groups.append(cur)
            continue
        if cur is None:
            raise ValueError("spec: token outside a section: %r" % t)
        cur.tok.append(t)
    for g in sp.groups:
        if len(g.tok) != g.n:
            raise ValueError("spec: group %s declares %d but has %d tokens" % (g.name, g.n, len(g.tok)))
    return sp


def load(path=SPEC_DEFAULT):
    with io.open(path, encoding="utf-8") as f:
        return parse(f.read())


# ---- decode: head indices -> plate strings -------------------------------------------------
# head order is the file order of the [head ...] sections:
#   region, class_num_01..03, hiragana, plate_num_01..04, plate_kind, legible
SPECIAL = ("<blank>", "<unused>")


class Plate:
    __slots__ = ("region", "cls", "hira", "num", "disp", "kind", "legible", "text")


def decode(sp, idx):
    heads = sp.of_kind("head")

    def tok(h):
        if h >= len(heads) or h >= len(idx):
            return "?"
        i = idx[h]
        if i < 0 or i >= heads[h].n:
            return "?"
        return heads[h].tok[i]

    p = Plate()
    p.region = tok(0)
    p.cls = "".join(t for t in (tok(h) for h in (1, 2, 3)) if t not in SPECIAL)
    p.hira = tok(4)
    p.num, p.disp = "", ""
    for h in (5, 6, 7, 8):
        t = tok(h)
        if t in SPECIAL:
            p.disp += "・"
        else:
            p.num += t
            p.disp += t
    p.kind = tok(9)
    p.legible = tok(10)
    p.text = "%s %s %s %s" % (p.region, p.cls, p.hira, p.disp)
    return p


# ---- canonical dump: byte-identical between C++ and Python ---------------------------------
def canonical_dump(sp):
    o = ["spec version %d\n" % sp.version, "groups %d\n" % len(sp.groups)]
    for g in sp.groups:
        o.append("%s %s %d\n" % (g.kind, g.name, g.n))
        for i in range(g.n):
            o.append("  %d\t%s\n" % (i, g.tok[i]))
    return "".join(o)


def decode_vectors_dump(sp, seed, count):
    heads = sp.of_kind("head")
    rng = Rng(seed)
    o = ["decode-vectors %d seed %d\n" % (count, seed)]
    for _ in range(count):
        idx = [rng.below(h.n) for h in heads]
        p = decode(sp, idx)
        o.append("  %s\t%s|%s|%s|%s|%s|%s|%s\t%s\n" % (
            " ".join(str(i) for i in idx),
            p.region, p.cls, p.hira, p.num, p.disp, p.kind, p.legible, p.text))
    return "".join(o)


def emit_header(spec_path, out_path):
    """Embedded copy of the spec for builds without a filesystem (WASM).
    Byte-identical to `jlpr labels --emit-header`."""
    with io.open(spec_path, encoding="utf-8", newline="") as f:
        text = f.read()
    bs, dq = chr(0x5C), chr(0x22)
    o = ["// GENERATED from %s -- do not edit."
         " (jlpr labels --emit-header | python tools/jlpr.py labels --emit-header)\n" % spec_path,
         "#pragma once\n", "inline const char* SPEC_LABELS_TXT =\n"]
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()               # C++ side stops at the final newline; no empty tail line
    for line in lines:
        if line.endswith("\r"):
            line = line[:-1]
        esc = []
        for ch in line:
            if ch in (dq, bs):
                esc.append(bs + ch)
            elif ord(ch) < 0x20:
                esc.append("%sx%02x" % (bs, ord(ch)))
            else:
                esc.append(ch)
        o.append("  %s%s%sn%s\n" % (dq, "".join(esc), bs, dq))
    o.append("  ;\n")
    s = "".join(o)
    with io.open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(s)
    return len(s.encode("utf-8"))
