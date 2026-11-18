#!/usr/bin/env python3
"""Correctness tests for the cube model (no rendering)."""

import numpy as np

from rubik_render import Cube, DIRS, FACE_NAMES, parse_scramble, parse_token


def fresh():
    return Cube(3)


def run(cube, text):
    cube.scramble(text)


def invert_token(tok: str) -> str:
    if "2" in tok:
        return tok
    return tok + "'" if "'" not in tok else tok.replace("'", "")


def inverse(text: str) -> str:
    toks = [t for t in text.replace("(", " ").replace(")", " ").split() if t]
    return " ".join(invert_token(t) for t in reversed(toks))


def test_order_4():
    moves = list("RLUDFBMES") + list("rludfb")
    for tok in moves:
        c = fresh()
        c.scramble(" ".join([tok] * 4))
        assert c.is_solved(), f"{tok} x4 not solved"
    for tok in "xyz":
        c = fresh()
        c.scramble(" ".join([tok] * 4))
        assert c.is_solved(), f"{tok} x4 not solved"


def test_sexy_move():
    c = fresh()
    c.scramble(" ".join(["R U R' U'"] * 6))
    assert c.is_solved(), "sexy move x6 not solved"


def test_r_then_rinverse():
    c = fresh()
    run(c, "R")
    run(c, "R'")
    assert c.is_solved()


def test_inverse_roundtrip():
    scrambles = [
        "R U F' U2 B' L D R2 U' L2 D2 F2 B2",
        "F R U' R' U' R U R' F'",
        "U2 D2 R2 L2 F2 B2 M2 E2 S2",
        "r u f' l' d b Rw2 M E' S",
        "R M' U' S2 L' E F' B2",
    ]
    for s in scrambles:
        c = fresh()
        run(c, s)
        run(c, inverse(s))
        assert c.is_solved(), f"inverse roundtrip failed for {s!r}"
        assert sum(sum(f.values()) for f in c.face_colours().values()) == 54


def test_colour_counts_after_r():
    c = fresh()
    run(c, "R")
    fc = c.face_colours()
    assert fc["U"] == {"U": 6, "F": 3}, fc["U"]
    assert fc["F"] == {"F": 6, "D": 3}, fc["F"]
    assert fc["R"] == {"R": 9}, fc["R"]
    assert fc["B"] == {"B": 6, "U": 3}, fc["B"]
    assert fc["D"] == {"D": 6, "B": 3}, fc["D"]
    assert fc["L"] == {"L": 9}, fc["L"]


def test_colour_counts_after_f():
    c = fresh()
    run(c, "F")
    fc = c.face_colours()
    assert fc["U"] == {"U": 6, "L": 3}, fc["U"]
    assert fc["R"] == {"R": 6, "U": 3}, fc["R"]
    assert fc["D"] == {"D": 6, "R": 3}, fc["D"]
    assert fc["L"] == {"L": 6, "D": 3}, fc["L"]


def test_colour_counts_after_rr():
    c = fresh()
    run(c, "R U")
    fc = c.face_colours()
    total = sum(sum(f.values()) for f in fc.values())
    assert total == 54
    for face in FACE_NAMES:
        assert sum(fc[face].values()) == 9


def test_parser():
    cases = {
        "R": (0, -90, {1}),
        "R'": (0, 90, {1}),
        "R2": (0, -180, {1}),
        "R2'": (0, 180, {1}),
        "L": (0, 90, {-1}),
        "U": (1, -90, {1}),
        "D": (1, 90, {-1}),
        "F": (2, -90, {1}),
        "B": (2, 90, {-1}),
        "M": (0, -90, {0}),
        "E": (1, -90, {0}),
        "S": (2, -90, {0}),
        "r": (0, -90, {1, 0}),
        "l": (0, 90, {-1, 0}),
        "Rw": (0, -90, {1, 0}),
        "Rw'": (0, 90, {1, 0}),
        "x": (0, -90, {-1, 0, 1}),
        "y'": (1, 90, {-1, 0, 1}),
        "z2": (2, -180, {-1, 0, 1}),
    }
    for tok, (axis, angle, layers) in cases.items():
        m = parse_token(tok)
        assert (m.axis, m.angle % 360, set(m.layers)) == (axis, angle % 360, layers), tok
    seq = parse_scramble("R U F' U2 B'  L  D R2 U'")
    assert len(seq) == 9


def test_sticker_presence():
    # every cubie carries exactly 3 stickers if corner, 2 if edge, 1 if centre
    c = fresh()
    counts = sorted(sum(s is not None for s in cu.stickers) for cu in c.cubies)
    assert counts.count(1) == 6 and counts.count(2) == 12 and counts.count(3) == 8


def test_scramble_no_int_overflow():
    c = fresh()
    run(c, " ".join(["R'", "U2", "F", "D'"] * 50))
    for cu in c.cubies:
        assert cu.pos.dtype == np.int64
        assert all(abs(int(v)) <= 1 for v in cu.pos)


if __name__ == "__main__":
    import sys
    import time

    start = time.time()
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in funcs:
        f()
        print(f"ok  {f.__name__}")
    print(f"\n{len(funcs)} tests passed in {time.time() - start:.2f}s")
