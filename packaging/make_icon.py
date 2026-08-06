"""Generate the PolyFut app icon (ICO + PNG) for the installer and website.

Run from repo root:
    python packaging/make_icon.py

The mark is a faceted football, built from real truncated-icosahedron geometry
rather than drawn by hand: 12 pentagons and 20 hexagons, orthographically
projected, back faces culled. That is the same shape a real ball is stitched
from, and it gives the panel seams the slightly irregular foreshortening a
hand-drawn version never gets right.

It matches the app: the ball is --lt-accent #0b7a42, and the seams and vertex
dots echo the net background, which is the same green drawn as lines with dots
at its vertices.
"""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "icons"
WEB_ASSETS = ROOT / "website" / "assets"

# Straight from the stylesheet, so the icon cannot drift from the UI.
GREEN = (11, 122, 66)          # --lt-accent, exactly
# The depth shading runs symmetrically about GREEN rather than upward from it,
# so the ball AVERAGES to the app's green instead of reading a shade lighter
# than every green in the UI.
GREEN_SHADE = (8, 96, 52)      # facets turning away
GREEN_LIT = (14, 148, 80)      # facets facing the viewer
GREEN_DARK = (7, 82, 45)       # pentagons
TILE = (246, 248, 246)         # --lt-page
SEAM = (246, 248, 246)
EDGE = (6, 74, 40)

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

    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=GREEN)

    # Below ~64px a 32-facet ball cannot resolve: thin seams blur to grey haze,
    # thick ones eat the green until the icon is mostly white speckle. Neither
    # is a seam-width problem, so small sizes drop the light seams entirely and
    # keep only the dark pentagons on a solid green ball. That still reads as a
    # football instantly - it is how a ball looks at a distance - and it keeps a
    # solid green silhouette, which is what a taskbar actually needs.
    detailed = size >= 64
    seam_w = max(1, round(ss * max(1.0, 2.75 * size / 256.0)))
    front = []
    for pts, kind in _truncated_icosahedron():
        rp = [rot(p) for p in pts]
        # Back-face cull: the face normal must point at the viewer (+z).
        c = tuple(sum(p[i] for p in rp) / len(rp) for i in range(3))
        if c[2] <= 0.06:
            continue
        front.append((c[2], rp, kind))

    for depth, rp, kind in sorted(front):          # far to near
        flat = [(cx + p[0] * r, cy - p[1] * r) for p in rp]
        if not detailed:
            # Simplified mark: dark pentagons only, no seams.
            if kind == "pent":
                draw.polygon(flat, fill=GREEN_DARK)
            continue
        if kind == "pent":
            fill = GREEN_DARK
        else:
            # Light the hexagons by depth so the sphere reads as round, running
            # shade -> lit through GREEN at the midpoint.
            t = max(0.0, min(1.0, (depth - 0.1) / 0.9))
            fill = tuple(int(GREEN_SHADE[i] + (GREEN_LIT[i] - GREEN_SHADE[i]) * t)
                         for i in range(3))
        draw.polygon(flat, fill=fill, outline=SEAM, width=seam_w)

    # (No vertex dots. The net's dots sit on thin lines against open space; here
    # the seams are already the light element and dots in the same colour drew
    # invisibly on top of them. The faceting carries the motif on its own.)

    # Crisp rim last, so the silhouette survives down to 16px.
    draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                 outline=EDGE, width=max(1, int(S * 0.012)))

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
