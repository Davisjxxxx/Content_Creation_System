from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = 512
OUT = Path("build")
OUT.mkdir(parents=True, exist_ok=True)

base = Image.new("RGBA", (SIZE, SIZE), (5, 5, 7, 255))

def heart_points(cx: float, cy: float, scale: float) -> list[tuple[float, float]]:
    # Parametric heart curve, flipped vertically into image coordinates.
    pts = []
    for i in range(720):
        t = i * 3.141592653589793 * 2 / 719
        x = 16 * (__import__("math").sin(t) ** 3)
        y = 13 * __import__("math").cos(t) - 5 * __import__("math").cos(2*t) - 2 * __import__("math").cos(3*t) - __import__("math").cos(4*t)
        pts.append((cx + x * scale, cy - y * scale))
    return pts

# Layer several blurred pink/red hearts to create the app's neon glow.
for radius, alpha, scale in [(42, 90, 11.1), (24, 120, 10.8), (12, 140, 10.5)]:
    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    draw.polygon(heart_points(256, 258, scale), fill=(255, 45, 125, alpha))
    glow = glow.filter(ImageFilter.GaussianBlur(radius))
    base.alpha_composite(glow)

heart = Image.new("RGBA", base.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(heart)
points = heart_points(256, 252, 10.25)
draw.polygon(points, fill=(255, 45, 125, 255))
# Red lower-right highlight.
overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
od = ImageDraw.Draw(overlay)
od.ellipse((238, 230, 430, 438), fill=(255, 59, 48, 165))
overlay = overlay.filter(ImageFilter.GaussianBlur(55))
heart = Image.alpha_composite(heart, overlay)
# Soft inner highlight.
draw = ImageDraw.Draw(heart)
draw.line(points[75:245], fill=(255, 190, 218, 120), width=3)
base.alpha_composite(heart)

png = OUT / "avatar_v2_icon.png"
ico = OUT / "avatar_v2.ico"
base.save(png)
base.save(ico, sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])
print(f"created {png} and {ico}")
