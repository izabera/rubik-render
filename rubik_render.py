#!/usr/bin/env python3
"""rubik-render — render a Rubik's cube scramble to a PNG.

Cube model: 26 cubies, each with an integer position in {-1,0,1}^3 and an
integer 3x3 orientation matrix. A move is a 90-degree rotation of a layer;
it updates both position and orientation, so stickers carry their colour
with them.

Rendering: perspective camera, painter's-free (the exterior surface of a
convex cube has no overlapping front faces), supersampled polygons,
rounded-corner stickers inset into dark plastic cells.

Usage:
    ./rubik_render.py "R U F' U2 B' L D R2 U'" -o scramble.png
    ./rubik_render.py "F U2 R' B" --size 1600 --yaw 32 --pitch 26
    ./rubik_render.py "R U R' U'" --steps -o steps.png
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Face geometry.  x = right, y = up, z = front (towards the viewer).
# --------------------------------------------------------------------------

DIRS = np.array(
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    dtype=np.int64,
)
FACE_NAMES = ["R", "L", "U", "D", "F", "B"]

# Clockwise face turn = -90 degrees about the face's outward normal
# (right-hand rule).  Verified against the standard U/F/R cycles.
FACE_AXIS_SIGN = {
    "R": (0, 1), "L": (0, -1),
    "U": (1, 1), "D": (1, -1),
    "F": (2, 1), "B": (2, -1),
}
SLICE_AXIS_SIGN = {"M": (0, 1), "E": (1, 1), "S": (2, 1)}
ROT_AXIS_SIGN = {"x": (0, 1), "y": (1, 1), "z": (2, 1)}

PALETTE = {
    "U": (244, 244, 244),
    "D": (255, 212, 8),
    "F": (32, 176, 74),
    "B": (38, 104, 216),
    "R": (222, 46, 46),
    "L": (255, 132, 18),
}

PLASTIC = (46, 50, 58)
LIGHT = np.array([0.30, 0.80, 0.55], float)
LIGHT /= np.linalg.norm(LIGHT)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

_SIN_COS = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}


def rot_matrix(axis: int, deg: int) -> np.ndarray:
    """Exact integer rotation matrix about a coordinate axis (degrees)."""
    deg %= 360
    if deg not in _SIN_COS:
        raise ValueError(f"rotation must be a multiple of 90, got {deg}")
    c, s = _SIN_COS[deg]
    if axis == 0:
        m = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == 1:
        m = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        m = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return np.array(m, dtype=np.int64)


# --------------------------------------------------------------------------
# Cube model
# --------------------------------------------------------------------------


class Cubie:
    __slots__ = ("pos", "orient", "stickers")

    def __init__(self, pos):
        self.pos = np.array(pos, dtype=np.int64)
        self.orient = np.eye(3, dtype=np.int64)  # local axis -> world axis
        self.stickers = [None] * 6                # indexed like DIRS
        for i, d in enumerate(DIRS):
            if int(self.pos @ d) == 1:            # face lies on the outside
                self.stickers[i] = FACE_NAMES[i]


@dataclass
class Move:
    axis: int
    angle: int                 # signed degrees, multiple of 90
    layers: frozenset          # cubie coordinates on this axis
    token: str


class Cube:
    def __init__(self, n: int = 3):
        if n % 2 != 1:
            raise ValueError("only odd cube sizes are supported")
        self.n = n
        lo = -(n // 2)
        coords = range(lo, lo + n)
        centre = lo + n // 2
        self.cubies = [
            Cubie((x, y, z))
            for x in coords for y in coords for z in coords
            if (x, y, z) != (centre,) * 3
        ]
        self.half = n / 2.0

    def apply(self, move: Move) -> None:
        m = rot_matrix(move.axis, move.angle)
        for c in self.cubies:
            if int(c.pos[move.axis]) in move.layers:
                c.pos = m @ c.pos
                c.orient = m @ c.orient

    def scramble(self, tokens) -> None:
        for move in parse_scramble(tokens):
            self.apply(move)

    def is_solved(self) -> bool:
        for c in self.cubies:
            if not np.array_equal(c.orient, np.eye(3, dtype=np.int64)):
                return False
            # the cubie is "home" iff its stickers match its position
            for i, d in enumerate(DIRS):
                if (c.stickers[i] is not None) != (int(c.pos @ d) == 1):
                    return False
                if c.stickers[i] is not None and c.stickers[i] != FACE_NAMES[i]:
                    return False
        return True

    def face_colours(self) -> dict:
        """Visible sticker colour counts per world face (sanity check)."""
        out = {name: {} for name in FACE_NAMES}
        for c in self.cubies:
            for i, d in enumerate(DIRS):
                colour = c.stickers[i]
                if colour is None:
                    continue
                n = c.orient @ d
                if int(c.pos @ n) != 1:
                    continue
                for j, d2 in enumerate(DIRS):
                    if np.array_equal(n, d2):
                        face = FACE_NAMES[j]
                        break
                out[face][colour] = out[face].get(colour, 0) + 1
        return out


# --------------------------------------------------------------------------
# Scramble parsing
# --------------------------------------------------------------------------


def parse_token(tok: str) -> Move | None:
    tok = tok.strip()
    if not tok:
        return None
    letter, rest = tok[0], tok[1:]
    sign = 0
    layers = {0}

    if letter in "RLUDFB":
        axis, sign = FACE_AXIS_SIGN[letter]
        wide = rest[:1] in ("w", "W")
        if wide:
            rest = rest[1:]
            layers = {sign, 0}
        else:
            layers = {sign}
    elif letter in "rludfb":
        axis, sign = FACE_AXIS_SIGN[letter.upper()]
        layers = {sign, 0}                      # wide two-layer turn
    elif letter in "MES":
        axis, sign = SLICE_AXIS_SIGN[letter]
        layers = {0}                            # middle slice
    elif letter in "xyz":
        axis, sign = ROT_AXIS_SIGN[letter]
        layers = {-1, 0, 1}                     # whole-cube rotation
    else:
        raise ValueError(f"unrecognised move token: {tok!r}")

    double = prime = False
    for ch in rest:
        if ch == "2":
            double = True
        elif ch in "'\u2019\u2032\uff02":
            prime = True
        else:
            raise ValueError(f"unrecognised move token: {tok!r}")

    angle = -90 * sign * (2 if double else 1)
    if prime:
        angle = -angle
    return Move(axis=axis, angle=int(angle), layers=frozenset(layers), token=tok)


def parse_scramble(text: str) -> list[Move]:
    cleaned = text.replace("(", " ").replace(")", " ").replace(",", " ")
    moves = []
    for tok in cleaned.split():
        # tolerate stray full stops / semicolons / brackets in copied scrambles
        tok = tok.strip("./;[]")
        if tok:
            moves.append(parse_token(tok))
    return moves


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------


def rounded_rect(hw: float, hh: float, r: float, seg: int = 6):
    """Outline of a rounded rectangle centred on the origin."""
    if r <= 0:
        return [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    pts = []
    corners = [
        (hw - r, hh - r, 0.0),
        (-(hw - r), hh - r, 90.0),
        (-(hw - r), -(hh - r), 180.0),
        (hw - r, -(hh - r), 270.0),
    ]
    for cx, cy, a0 in corners:
        for k in range(seg + 1):
            a = math.radians(a0 + 90.0 * k / seg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def shade(rgb, k: float):
    return tuple(max(0, min(255, int(round(v * k)))) for v in rgb)


def make_background(w: int, h: int, flat=None) -> Image.Image:
    if flat is not None:
        return Image.new("RGB", (w, h), flat)
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx - (w - 1) / 2.0) / (w * 0.62)
    ny = (yy - (h - 1) * 0.44) / (h * 0.80)
    d = np.clip(np.sqrt(nx * nx + ny * ny), 0.0, 1.0)[..., None]
    inner = np.array([40, 46, 62], float)
    outer = np.array([10, 11, 15], float)
    arr = inner * (1.0 - d) + outer * d
    return Image.fromarray(arr.astype(np.uint8))


class Camera:
    def __init__(self, yaw: float, pitch: float, dist: float, half: float):
        yr, pr = math.radians(yaw), math.radians(pitch)
        direction = np.array(
            [math.cos(pr) * math.sin(yr), math.sin(pr), math.cos(pr) * math.cos(yr)]
        )
        self.pos = direction * dist
        self.fwd = -self.pos / np.linalg.norm(self.pos)
        up_ref = np.array([0.0, 1.0, 0.0])
        self.right = np.cross(self.fwd, up_ref)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.fwd)
        self.up /= np.linalg.norm(self.up)
        self.half = half
        self.uv, _ = self.project(self._corners())
        self.scale = 0.0
        self.centre = ((self.uv[:, 0].min() + self.uv[:, 0].max()) / 2.0,
                       (self.uv[:, 1].min() + self.uv[:, 1].max()) / 2.0)

    def _corners(self):
        h = self.half
        return np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], float)

    def project(self, points: np.ndarray):
        v = np.atleast_2d(points) - self.pos
        z = v @ self.fwd
        uv = np.stack([(v @ self.right) / z, -(v @ self.up) / z], axis=-1)
        return uv, z

    def fit(self, width: int, height: int, fraction: float):
        bbw = self.uv[:, 0].max() - self.uv[:, 0].min()
        bbh = self.uv[:, 1].max() - self.uv[:, 1].min()
        self.scale = min(width * fraction / bbw, height * fraction / bbh)

    def to_pixels(self, uv, width: int, height: int):
        x = (uv[:, 0] - self.centre[0]) * self.scale + width / 2.0
        y = (uv[:, 1] - self.centre[1]) * self.scale + height / 2.0
        return x, y


# --------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------


def render_cube(
    cube: Cube,
    *,
    size: int = 1024,
    ss: int = 3,
    yaw: float = 38.0,
    pitch: float = 22.0,
    dist: float = 6.4,
    fit_frac: float = 0.86,
    bg=None,
    label: str | None = None,
) -> Image.Image:
    rw, rh = size * ss, size * ss
    cam = Camera(yaw, pitch, dist, cube.half)
    cam.fit(rw, rh, fit_frac)

    img = make_background(rw, rh, bg)
    draw = ImageDraw.Draw(img)
    off = 0.004  # stickers float a hair above the plastic cell

    sticker_hw = 0.44
    sticker_r = 0.12

    faces = []  # (world_normal, cell_centre, plastic_pts, sticker_pts, colour)
    for c in cube.cubies:
        for i, d in enumerate(DIRS):
            n = c.orient @ d
            if int(c.pos @ n) != 1:          # sticker is not on the surface
                continue
            centre = c.pos + c.orient @ (0.5 * d)
            if (cam.pos - centre) @ n <= 1e-9:  # back-facing
                continue

            axis = int(np.argmax(np.abs(d)))
            tangents = [k for k in range(3) if k != axis]
            t1 = np.zeros(3); t1[tangents[0]] = 1.0
            t2 = np.zeros(3); t2[tangents[1]] = 1.0

            def poly(hw, hh, r, extra):
                pts = rounded_rect(hw, hh, r)
                return np.array(
                    [c.pos + c.orient @ ((0.5 + extra) * d + u * t1 + v * t2)
                     for u, v in pts],
                    float,
                )

            plastic = poly(0.5, 0.5, 0.0, 0.0)
            sticker = poly(sticker_hw, sticker_hw, sticker_r, off)
            faces.append((n, plastic, sticker, c.stickers[i]))

    # Every face here is exterior to a convex solid, so front faces tile the
    # silhouette without overlapping; draw plastic everywhere, stickers on top.
    for n, plastic, sticker, colour in faces:
        t = max(0.0, float(n @ LIGHT))
        k = 0.68 + 0.32 * t
        uv, depth = cam.project(plastic)
        x, y = cam.to_pixels(uv, rw, rh)
        pts = list(zip(x, y))
        draw.polygon(pts, fill=shade(PLASTIC, k * 0.72))
        if sticker is not None:
            uv, depth = cam.project(sticker)
            x, y = cam.to_pixels(uv, rw, rh)
            pts = list(zip(x, y))
            draw.polygon(pts, fill=shade(PALETTE[colour], k))

    img = img.resize((size, size), Image.LANCZOS)

    if label:
        img = draw_label(img, label)
    return img


def draw_label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    size = img.width
    fs = max(13, int(size * 0.042))
    while fs > 9:
        font = ImageFont.truetype(FONT_PATH, fs)
        if draw.textlength(text, font=font) <= size * 0.88:
            break
        fs -= 1
    font = ImageFont.truetype(FONT_PATH, fs)
    w = draw.textlength(text, font=font)
    y = size - int(size * 0.055) - font.size
    x = (size - w) / 2.0
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(168, 176, 192))
    return img


def render_steps(
    tokens: list[str],
    *,
    size: int = 1024,
    ss: int = 3,
    yaw: float = 38.0,
    pitch: float = 22.0,
    dist: float = 6.4,
) -> Image.Image:
    """Contact sheet: the cube after each successive move."""
    moves = parse_scramble(" ".join(tokens))
    if not moves:
        raise SystemExit("no moves to render")

    cols = min(5, len(moves))
    rows = (len(moves) + cols - 1) // cols
    cell = max(150, size // 4)
    gap = max(8, cell // 22)
    pad = max(10, cell // 16)
    font_px = max(11, int(cell * 0.16))
    label_h = font_px + max(4, cell // 22)
    W = cols * cell + (cols - 1) * gap + 2 * pad
    H = rows * (cell + label_h) + (rows - 1) * gap + 2 * pad

    sheet = Image.new("RGB", (W, H), (15, 17, 22))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT_PATH, font_px)

    cube = Cube(3)
    y = pad
    for r in range(rows):
        x = pad
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(moves):
                break
            cube.apply(moves[idx])
            frame = render_cube(
                cube, size=cell, ss=ss, yaw=yaw, pitch=pitch, dist=dist,
                fit_frac=0.80, bg=(24, 27, 34), label=None,
            )
            sheet.paste(frame, (x, y))
            tok = moves[idx].token
            tw = draw.textlength(tok, font=font)
            ty = y + cell + max(2, label_h // 4)
            draw.text((x + (cell - tw) / 2 + 1, ty + 1), tok, font=font, fill=(0, 0, 0))
            draw.text((x + (cell - tw) / 2, ty), tok, font=font, fill=(176, 184, 200))
            x += cell + gap
        y += cell + label_h + gap
    return sheet


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="rubik-render",
        description="Render a Rubik's cube scramble to a PNG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Moves: R L U D F B (outer) · r l u d f b or Rw (wide) · "
            "M E S (slice) · x y z (rotations)\n"
            "Modifiers: ' (counter-clockwise), 2 (half turn)\n\n"
            "Examples:\n"
            '  rubik-render.py "R U F\' U2 B\' L D R2 U\'" -o scramble.png\n'
            '  rubik-render.py "R U R\' U\'" --steps -o steps.png\n'
        ),
    )
    ap.add_argument("scramble", help='scramble string, e.g. "R U F\' U2 B\'" (use - for stdin)')
    ap.add_argument("-o", "--out", help="output PNG path (default: cube.png)")
    ap.add_argument("--size", type=int, default=1024, help="image size in pixels (default 1024)")
    ap.add_argument("--ss", type=int, default=3, help="supersampling factor (default 3)")
    ap.add_argument("--yaw", type=float, default=38.0, help="camera yaw, degrees (default 38)")
    ap.add_argument("--pitch", type=float, default=22.0, help="camera pitch, degrees (default 22)")
    ap.add_argument("--dist", type=float, default=6.4, help="camera distance (default 6.4)")
    ap.add_argument("--flat-bg", metavar="R,G,B", help="flat background colour instead of gradient")
    ap.add_argument("--no-label", action="store_true", help="do not print the scramble")
    ap.add_argument("--steps", action="store_true", help="contact sheet of the cube after each move")
    args = ap.parse_args(argv)

    text = sys.stdin.read() if args.scramble == "-" else args.scramble
    tokens = [t for t in text.split() if t]

    bg = None
    if args.flat_bg:
        bg = tuple(int(v) for v in args.flat_bg.split(","))
        if len(bg) != 3:
            ap.error("--flat-bg expects R,G,B")

    out = args.out or ("steps.png" if args.steps else "cube.png")

    if args.steps:
        img = render_steps(tokens, size=args.size, ss=args.ss, yaw=args.yaw,
                           pitch=args.pitch, dist=args.dist)
    else:
        cube = Cube(3)
        cube.scramble(text)
        img = render_cube(cube, size=args.size, ss=args.ss, yaw=args.yaw,
                          pitch=args.pitch, dist=args.dist, bg=bg,
                          label=None if args.no_label else text.strip())

    img.save(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
