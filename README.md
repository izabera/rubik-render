# rubik-render

Render any Rubik's cube scramble (3x3) to a PNG — pure Python, numpy + Pillow.

```
./rubik_render.py "R U F' U2 B' L D R2 U'"              # -> cube.png (1024px, labelled)
./rubik_render.py "R U F' U2 B' L D R2 U'" --size 1600 -o big.png
./rubik_render.py "R U R' U'" --steps                   # contact sheet, cube after each move
echo "F R U' R' U' R U R' F'" | ./rubik_render.py - -o out.png
```

Options: `--yaw` / `--pitch` (camera angle), `--dist` (perspective), `--flat-bg R,G,B`,
`--ss` (supersampling), `--no-label`.

Notation supported: outer turns `R L U D F B`, wide turns `r l u d f b` or `Rw Uw`,
middle slices `M E S`, whole-cube rotations `x y z`; modifiers `'` and `2`.

Model: 26 cubies, integer position + integer 3x3 orientation matrix; a move rotates
a layer. Rendering: perspective camera, rounded-corner stickers inset into dark
plastic cells, supersampled polygons, radial-gradient background.

Tests: `python3 test_cube.py` (model + notation), `python3 verify_render.py`
(samples every visible sticker centre in the PNG and compares with the model).
