"""Generate synthetic base/highlighted PDFs to test extract_space_masks.py
before real client PDFs are available. Also rasterizes the ground-truth
polygons at the same DPI used by the pipeline so IoU can be checked.
"""
import json
from pathlib import Path

import cv2
import fitz
import numpy as np

OUT = Path(__file__).parent
PAGE_W, PAGE_H = 800, 500  # points
DPI = 200

BURGUNDY = (152 / 255, 0, 0)
LIGHT_GREY = (0.92, 0.92, 0.92)
DARK = (0.15, 0.15, 0.15)

# ground-truth shapes, in PDF point coordinates
RECEPTION_POLY = [(120, 120), (260, 90), (300, 180), (230, 240), (140, 210)]
# board room: two disjoint blobs under one space, to test multi-part mask handling
BOARDROOM_POLY_A = [(450, 300), (560, 280), (590, 360), (500, 390)]
BOARDROOM_POLY_B = [(600, 300), (650, 290), (660, 340), (610, 350)]


def draw_base_plan(page):
    page.draw_rect(fitz.Rect(20, 20, PAGE_W - 20, PAGE_H - 20), color=DARK, width=2)
    # a few plain "rooms" so diffing has something non-trivial to ignore
    page.draw_rect(fitz.Rect(40, 40, 380, 260), color=DARK, fill=LIGHT_GREY, width=1)
    page.draw_rect(fitz.Rect(420, 40, 760, 260), color=DARK, fill=LIGHT_GREY, width=1)
    page.draw_rect(fitz.Rect(40, 300, 760, 460), color=DARK, fill=LIGHT_GREY, width=1)


def make_pdf(path, extra_polys=None):
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    draw_base_plan(page)
    if extra_polys:
        for poly in extra_polys:
            page.draw_polyline(poly + [poly[0]], color=BURGUNDY, fill=BURGUNDY, width=0)
    doc.save(path)
    doc.close()


def rasterize_ground_truth(polys, dpi=DPI):
    """Rasterize the known PDF-point polygons at the same zoom as the pipeline,
    for IoU comparison against the extracted mask."""
    zoom = dpi / 72.0
    w, h = int(PAGE_W * zoom), int(PAGE_H * zoom)
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        pts = np.array([[x * zoom, y * zoom] for x, y in poly], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def main():
    make_pdf(OUT / "base.pdf")
    make_pdf(OUT / "Reception.pdf", extra_polys=[RECEPTION_POLY])
    make_pdf(OUT / "Board Room.pdf", extra_polys=[BOARDROOM_POLY_A, BOARDROOM_POLY_B])

    gt = {
        "reception": rasterize_ground_truth([RECEPTION_POLY]),
        "board-room": rasterize_ground_truth([BOARDROOM_POLY_A, BOARDROOM_POLY_B]),
    }
    for slug, mask in gt.items():
        cv2.imwrite(str(OUT / f"_gt_{slug}.png"), mask)

    print("wrote base.pdf, Reception.pdf, Board Room.pdf + ground-truth masks into", OUT)


if __name__ == "__main__":
    main()
