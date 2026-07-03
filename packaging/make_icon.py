"""Generate PolyFut app icon (ICO + PNG) for installer and website.

Run from repo root:
    python packaging/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "icons"
WEB_ASSETS = ROOT / "website" / "assets"

BG = (18, 52, 36)
FIELD = (34, 120, 72)
LINE = (235, 245, 238)
ACCENT = (255, 210, 60)


def _draw_icon(size: int):
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), BG)
    draw = ImageDraw.Draw(img)
    pad = max(2, size // 16)
    draw.rounded_rectangle(
        (pad, pad, size - pad - 1, size - pad - 1),
        radius=size // 6,
        fill=FIELD,
        outline=LINE,
        width=max(1, size // 48),
    )
    cx, cy = size // 2, size // 2
    r = size // 5
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=LINE, width=max(1, size // 40))
    draw.line((pad * 2, cy, size - pad * 2, cy), fill=LINE, width=max(1, size // 48))
    # center spot
    spot = max(2, size // 28)
    draw.ellipse((cx - spot, cy - spot, cx + spot, cy + spot), fill=ACCENT)
    # corner arcs
    arc_r = size // 3
    for ox, oy in ((cx, pad * 2), (cx, size - pad * 2)):
        draw.arc(
            (cx - arc_r, oy - arc_r // 2, cx + arc_r, oy + arc_r // 2),
            200,
            340,
            fill=LINE,
            width=max(1, size // 48),
        )
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WEB_ASSETS.mkdir(parents=True, exist_ok=True)

    ico_path = OUT_DIR / "polyfut.ico"
    sizes = [16, 32, 48, 64, 128, 256]
    base = _draw_icon(256)
    base.save(
        ico_path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
    )

    png_path = OUT_DIR / "polyfut-256.png"
    base.save(png_path, format="PNG")
    base.save(WEB_ASSETS / "polyfut-icon.png", format="PNG")
    print(f"Wrote {ico_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
