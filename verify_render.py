#!/usr/bin/env python3
"""Verify rendered pixels against the cube model.

Samples the centre of every visible sticker cell and compares it with the
colour the model says should be there (same camera + same shading maths).
"""

import numpy as np
from PIL import Image

from rubik_render import Camera, Cube, DIRS, LIGHT, PALETTE, render_cube, shade


def expected(cube, size, yaw, pitch, dist, fit_frac):
    cam = Camera(yaw, pitch, dist, cube.half)
    cam.fit(size, size, fit_frac)
    out = []
    for c in cube.cubies:
        for i, d in enumerate(DIRS):
            n = c.orient @ d
            if c.stickers[i] is None or int(c.pos @ n) != 1:
                continue
            centre = c.pos + c.orient @ ((0.5 + 0.004) * d)
            if (cam.pos - centre) @ n <= 1e-9:
                continue
            uv, _ = cam.project(np.array([centre], float))
            x, y = cam.to_pixels(uv, size, size)
            k = 0.68 + 0.32 * max(0.0, float(n @ LIGHT))
            out.append((int(x[0]), int(y[0]), shade(PALETTE[c.stickers[i]], k), n))
    return out


def check(scramble, size=512, yaw=38.0, pitch=22.0, dist=6.4, fit_frac=0.86):
    cube = Cube(3)
    cube.scramble(scramble)
    img = render_cube(cube, size=size, ss=3, yaw=yaw, pitch=pitch, dist=dist,
                      fit_frac=fit_frac, bg=(20, 23, 30), label=None)
    px = np.asarray(img)
    bad = 0
    for x, y, rgb, n in expected(cube, size, yaw, pitch, dist, fit_frac):
        got = px[y, x]
        if not np.all(np.abs(got.astype(int) - np.array(rgb)) <= 3):
            bad += 1
            print(f"  mismatch at ({x},{y}) normal={n} expected={rgb} got={tuple(got)}")
    return bad, len(expected(cube, size, yaw, pitch, dist, fit_frac))


if __name__ == "__main__":
    total = bad = 0
    for scramble in [
        "",
        "R",
        "R U",
        "R U F' U2 B' L D R2 U' L2 F2 D2 B2 R' U'",
        "F R U' R' U' R U R' F'",
        "M E S r u f l d b x y z",
    ]:
        b, n = check(scramble)
        total += n
        bad += b
        print(f"{'OK ' if b == 0 else 'FAIL'} {scramble or '(solved)'}: {n} stickers checked, {b} mismatches")
    print(f"\n{total} sticker samples, {bad} mismatches")
    raise SystemExit(1 if bad else 0)
