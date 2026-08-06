"""Generate the PolyFut app icon (ICO + PNG) for the installer and website.

Run from repo root:
    python packaging/make_icon.py

The mark is a football drawn as an open net: the app's own background - thin
green strands with dots where they meet - wrapped onto a sphere.

The geometry is real, not drawn by hand. An icosahedron truncated one third
along every edge turns its 12 vertices into pentagons and its 20 faces into
hexagons, which is exactly how a ball is stitched. Rendering only the edges,
with the far side of the net showing faintly through the near side, gives the
strands the uneven foreshortening a hand-drawn mesh never gets right.

A filled version came first and was wrong for two reasons. Its dark panels
(#07522d) sat at hue 150, and a green that dark reads as teal once it is small -
it looked like blue patches in the taskbar. And a solid ball had nothing to do
with the rest of the app. A wireframe has no dark fills to misread and is the
same drawing as the background.

Colour is --lt-accent #0b7a42, taken from the stylesheet so the icon cannot
drift from the UI.
"""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "icons"
WEB_ASSETS = ROOT / "website" / "assets"

# Straight from the stylesheet, so the icon cannot drift from the UI.
GREEN = (11, 122, 66)          # --lt-accent, exactly
# Nothing darker than this. The filled version used #07522d for the pentagons,
# and at hue 150 a green that dark reads as teal once it is small - which is
# what looked like blue patches in the taskbar. A wireframe has no dark fills
# at all, so the problem goes away rather than being tuned around.
GREEN_BACK = (150, 196, 172)   # far side of the net, seen through the front
TILE = (246, 248, 246)         # --lt-page

PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _icosahedron():
    """12 vertices and 20 triangular faces of a unit icosahedron."""
    verts = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            verts += [(0, s1 * 1.0, s2 * PHI),
                      (s1 * 1.0, s2 * PHI, 0),
                      (s1 * PHI, 0, s2 * 1.0)]
    # de-duplicate and normalise
    uniq = []
    for v in verts:
        if not any(all(abs(a - b) < 1e-9 for a, b in zip(v, u)) for u in uniq):
            uniq.append(v)
    n = [tuple(c / math.sqrt(sum(x * x for x in v)) for c in v) for v in uniq]

    # Edge length of the icosahedron on the unit sphere; faces are vertex
    # triples all mutually that far apart.
    d2 = sorted({round(sum((a - b) ** 2 for a, b in zip(p, q)), 6)
                 for i, p in enumerate(n) for q in n[i + 1:]})
    edge2 = d2[0]
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(sum((a - b) ** 2 for a, b in zip(n[i], n[j])) - edge2) > 1e-6:
                continue
            for k in range(j + 1, 12):
                if (abs(sum((a - b) ** 2 for a, b in zip(n[i], n[k])) - edge2) < 1e-6
                        and abs(sum((a - b) ** 2 for a, b in zip(n[j], n[k])) - edge2) < 1e-6):
                    faces.append((i, j, k))
    return n, faces, edge2


def _truncated_icosahedron():
    """Pentagon and hexagon faces of a football, as unit-sphere polygons.

    Truncating an icosahedron one third along each edge turns every vertex into
    a pentagon and every triangular face into a hexagon - 12 and 20, which is
    exactly a football.
    """
    verts, faces, edge2 = _icosahedron()

    def norm(p):
        m = math.sqrt(sum(c * c for c in p))
        return tuple(c / m for c in p)

    def cut(a, b, t):
        return norm(tuple(verts[a][i] + (verts[b][i] - verts[a][i]) * t
                          for i in range(3)))

    adj = {i: set() for i in range(12)}
    for f in faces:
        for a in f:
            for b in f:
                if a != b:
                    adj[a].add(b)

    def order_ring(points, axis):
        """Sort a face's points into a ring around its own normal."""
        cx = tuple(sum(p[i] for p in points) / len(points) for i in range(3))
        up = (0, 0, 1) if abs(axis[2]) < 0.9 else (1, 0, 0)
        u = norm(tuple(up[i] - axis[i] * sum(up[j] * axis[j] for j in range(3))
                       for i in range(3)))
        v = (axis[1] * u[2] - axis[2] * u[1],
             axis[2] * u[0] - axis[0] * u[2],
             axis[0] * u[1] - axis[1] * u[0])
        return sorted(points, key=lambda p: math.atan2(
            sum((p[i] - cx[i]) * v[i] for i in range(3)),
            sum((p[i] - cx[i]) * u[i] for i in range(3))))

    polys = []
    # A pentagon around every original vertex.
    for i in range(12):
        pts = [cut(i, j, 1 / 3) for j in adj[i]]
        polys.append((order_ring(pts, verts[i]), "pent"))
    # A hexagon on every original face.
    for f in faces:
        pts = []
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            pts += [cut(a, b, 1 / 3), cut(a, b, 2 / 3)]
        c = norm(tuple(sum(verts[k][i] for k in f) / 3 for i in range(3)))
        polys.append((order_ring(pts, c), "hex"))
    return polys


def _draw_icon(size: int):
    from PIL import Image, ImageDraw

    # Supersample: the seams are sub-pixel at 16px and alias badly otherwise.
    ss = 8 if size <= 64 else 4
    S = size * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = max(2, S // 16)
    draw.rounded_rectangle((pad // 2, pad // 2, S - pad // 2 - 1, S - pad // 2 - 1),
                           radius=S // 5, fill=TILE)

    cx = cy = S / 2
    r = S * 0.36
    # Tilted so a pentagon sits near the centre rather than edge-on, which is
    # what makes it read as a football instead of a wireframe sphere.
    ax, ay = math.radians(-24), math.radians(16)

    def rot(p):
        x, y, z = p
        y, z = y * math.cos(ax) - z * math.sin(ax), y * math.sin(ax) + z * math.cos(ax)
        x, z = x * math.cos(ay) + z * math.sin(ay), -x * math.sin(ay) + z * math.cos(ay)
        return x, y, z

    # An open net rather than a filled ball - the app's own background is thin
    # green lines with dots where they meet, and this is the same drawing
    # wrapped onto a sphere. No panel fills, so nothing is dark enough to read
    # as teal once it shrinks, which is what the old pentagons did.
    faces = [([rot(q) for q in pts], kind)
             for pts, kind in _truncated_icosahedron()]

    # Each edge once, tagged near/far by midpoint depth and by whether it rings
    # a pentagon. Every pentagon edge is shared with a hexagon, so the pentagon
    # rings alone are 60 of the 90 strands - a third fewer lines that still
    # reads as a football, which is what the small sizes need.
    edges = {}
    for rp, kind in faces:
        for i in range(len(rp)):
            a, b = rp[i], rp[(i + 1) % len(rp)]
            key = tuple(sorted((tuple(round(c, 5) for c in a),
                                tuple(round(c, 5) for c in b))))
            depth, was_pent = edges.get(key, (None, False))
            edges[key] = ((a[2] + b[2]) / 2, was_pent or kind == "pent")

    # At 32px and below, 90 strands collapse into a green blob with white
    # speckle. Drop to the pentagon rings only: fewer lines, same silhouette,
    # still obviously a ball.
    if size <= 32:
        edges = {k: v for k, v in edges.items() if v[1]}
    if size <= 16:
        # 16px is ~11px of ball. Even the pentagon rings are too many strands
        # there, so keep only the ones squarely facing the viewer and let the
        # rim carry the rest. It reads as a ball with a couple of panels, which
        # is all that fits.
        edges = {k: v for k, v in edges.items() if v[0] > 0.55}

    def flat(q):
        return (cx + q[0] * r, cy - q[1] * r)

    # Weights in FINAL pixels so they survive the downscale at every size.
    lw_front = max(1, round(ss * max(1.15, 3.2 * size / 256.0)))
    lw_back = max(1, round(lw_front * 0.62))

    # The far side of the net, seen through the near side. Dropped below 48px,
    # where it is only mud behind the front strands.
    if size >= 48:
        for (a, b), (depth, _p) in edges.items():
            if depth < 0:
                draw.line([flat(a), flat(b)], fill=GREEN_BACK, width=lw_back)

    for (a, b), (depth, _p) in edges.items():
        if depth >= 0:
            draw.line([flat(a), flat(b)], fill=GREEN, width=lw_front)

    # Dots where strands meet - the motif from the net background, and what
    # keeps the mesh reading as deliberate rather than as scribble.
    if size >= 32:
        dot = max(1, round(ss * max(1.5, 4.0 * size / 256.0)))
        seen = set()
        for (a, b), (depth, _p) in edges.items():
            if depth < 0:
                continue
            for q in (a, b):
                if q[2] < 0.15:
                    continue
                k = tuple(round(c, 4) for c in q)
                if k in seen:
                    continue
                seen.add(k)
                x, y = flat(q)
                draw.ellipse((x - dot, y - dot, x + dot, y + dot), fill=GREEN)

    # The rim closes the net into a ball. Without it the mesh has no silhouette
    # and dissolves at small sizes.
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=GREEN,
                 width=max(1, round(ss * max(1.3, 3.4 * size / 256.0))))

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    from PIL import Image

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WEB_ASSETS.mkdir(parents=True, exist_ok=True)

    sizes = [16, 32, 48, 64, 128, 256]
    # Render each size separately rather than downscaling one master: at 16px a
    # downscaled 256px ball turns to mush, while a purpose-rendered one keeps
    # its seams.
    frames = [_draw_icon(s) for s in sizes]

    ico_path = OUT_DIR / "polyfut.ico"
    frames[-1].save(ico_path, format="ICO", sizes=[(s, s) for s in sizes],
                    append_images=frames[:-1])

    png_path = OUT_DIR / "polyfut-256.png"
    frames[-1].save(png_path, format="PNG")
    frames[-1].save(WEB_ASSETS / "polyfut-icon.png", format="PNG")
    print(f"Wrote {ico_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
