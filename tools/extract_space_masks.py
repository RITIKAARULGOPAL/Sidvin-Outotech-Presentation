"""
Extract per-space hover hotspots from a folder of highlighted axo PDFs.

Each PDF in --in (other than --base) is assumed to be visually identical to the
base axo PDF except one space is filled/highlighted in red/burgundy. The PDF's
filename (minus extension) is used as that space's display name.

For each such PDF we:
  1. Rasterize it (and the base) at an identical resolution via PyMuPDF.
  2. Diff it against the base + threshold on the red/burgundy hue band to
     isolate exactly the highlighted pixels (rejects anti-aliasing/compression
     noise elsewhere in the page).
  3. Clean the mask (morphology + drop tiny connected-component blobs).
  4. Emit ONE composited PNG per space: RGB = that PDF's own rendered pixels
     (cropped to the mask's bounding box), alpha = the mask. No color is
     synthesized — the on-hover look is literally the source PDF's pixels.
  5. Emit a full-canvas binary hit-mask PNG per space, used client-side to
     build an id-map canvas for mouse hit-testing.
  6. Write manifest.json describing every space (id/name/bbox/asset paths).

Usage:
    python extract_space_masks.py --in <folder_of_pdfs> --base base-axo.pdf \
        --out ../assets/pegman/spaces --dpi 200

Re-run any time new space PDFs are dropped into --in; the whole output is
regenerated from the directory listing, nothing is hand-coded per space.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image

# --- tunables -----------------------------------------------------------
DIFF_THRESHOLD = 18        # min per-pixel abs-diff (0-255) to count as "changed"
MIN_BLOB_AREA_FRAC = 0.0005  # drop connected components smaller than this fraction of canvas area
BBOX_PADDING_PX = 12
MORPH_KERNEL = 5


def rasterize(pdf_path, dpi):
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    doc.close()
    return img  # RGB uint8


def isolate_highlight_mask(base_rgb, variant_rgb):
    if base_rgb.shape != variant_rgb.shape:
        raise ValueError(
            f"base/variant raster size mismatch {base_rgb.shape} vs {variant_rgb.shape} "
            "- were they rendered from PDFs with different page sizes?"
        )

    diff = cv2.absdiff(base_rgb, variant_rgb)
    diff_mask = (diff.max(axis=2) > DIFF_THRESHOLD)

    # the base plan is neutral gray/white; a highlight is a *colored* tint on top
    # of it - but the tint color isn't consistent across files (some are red,
    # some blue), so key on "saturated, not neutral gray" rather than a hue band
    hsv = cv2.cvtColor(variant_rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    saturation_mask = (s > 35) & (v > 30)

    mask = (diff_mask & saturation_mask).astype(np.uint8) * 255

    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = MIN_BLOB_AREA_FRAC * mask.shape[0] * mask.shape[1]
    clean = np.zeros_like(mask)
    for label_id in range(1, num_labels):  # 0 = background
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label_id] = 255

    return clean


def slugify(name):
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "space"


def display_name(stem):
    # "Board_Room-v2" / "board room" -> "Board Room"; leaves already-clean names alone.
    # Short all-caps words (<=3 chars, e.g. "MD", "HR") are treated as acronyms and
    # kept upper rather than title-cased down to "Md"/"Hr".
    cleaned = re.sub(r"[_\-]+", " ", stem).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not (cleaned.islower() or cleaned.isupper()):
        return cleaned
    words = cleaned.split(" ")
    return " ".join(w if (w.isupper() and len(w) <= 3) else w.capitalize() for w in words)


def rasterize_image_base(image_path, dpi, reference_pdf):
    """Load a raster base image (not a PDF) and resize it (if needed) to match
    the exact pixel dimensions the PDF variants get rasterized at - rasterizes
    the reference PDF itself rather than computing from page-rect math, since
    fitz's own rounding doesn't always match a naive width*dpi/72 calculation."""
    target_h, target_w = rasterize(reference_pdf, dpi).shape[:2]

    img = Image.open(image_path).convert("RGB")
    if img.size != (target_w, target_h):
        print(f"upscaling base image {img.size} -> ({target_w}, {target_h}) to match PDF raster resolution")
        img = img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(img)


def synthesize_base(pdf_files, dpi):
    """Reconstruct the unhighlighted base by taking the per-pixel median across
    every highlighted PDF. Each PDF highlights a different, non-overlapping
    space, so at any given pixel at most one PDF is "wrong" - the median across
    all of them cancels that highlight out and recovers the plain base."""
    rasters = [rasterize(p, dpi) for p in pdf_files]
    stack = np.stack(rasters, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def process(in_dir, base_arg, out_dir, dpi):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pdfs = sorted(in_dir.glob("*.pdf"))
    if not all_pdfs:
        print(f"no PDFs found in {in_dir}", file=sys.stderr)
        return

    if base_arg == "auto":
        print(f"no explicit --base given - synthesizing base from the median of all {len(all_pdfs)} PDFs")
        base_rgb = synthesize_base(all_pdfs, dpi)
        pdf_files = all_pdfs
    else:
        base_path = Path(base_arg)
        if base_path.suffix.lower() == ".pdf":
            base_rgb = rasterize(base_path, dpi)
        else:
            base_rgb = rasterize_image_base(base_path, dpi, all_pdfs[0])
        pdf_files = sorted(p for p in all_pdfs if p.resolve() != base_path.resolve())

    canvas_h, canvas_w = base_rgb.shape[:2]

    axo_png_path = out_dir.parent / "axo.png"
    Image.fromarray(base_rgb).save(axo_png_path)
    print(f"wrote base axo -> {axo_png_path} ({canvas_w}x{canvas_h})")

    if not pdf_files:
        print(f"no space PDFs found in {in_dir} (besides the base)", file=sys.stderr)

    manifest = {
        "canvas": {"width": canvas_w, "height": canvas_h},
        "spaces": [],
    }

    for pdf_path in pdf_files:
        stem = pdf_path.stem
        slug = slugify(stem)
        name = display_name(stem)

        variant_rgb = rasterize(pdf_path, dpi)
        mask = isolate_highlight_mask(base_rgb, variant_rgb)

        if not mask.any():
            print(f"WARNING: no highlighted pixels found in {pdf_path.name} - skipping", file=sys.stderr)
            continue

        ys, xs = np.where(mask > 0)
        x0, x1 = max(xs.min() - BBOX_PADDING_PX, 0), min(xs.max() + BBOX_PADDING_PX, canvas_w - 1)
        y0, y1 = max(ys.min() - BBOX_PADDING_PX, 0), min(ys.max() + BBOX_PADDING_PX, canvas_h - 1)

        cropped_rgb = variant_rgb[y0:y1 + 1, x0:x1 + 1]
        cropped_alpha = mask[y0:y1 + 1, x0:x1 + 1]
        overlay_rgba = np.dstack([cropped_rgb, cropped_alpha])

        overlay_name = f"{slug}.png"
        hitmask_name = f"{slug}.hit-mask.png"
        Image.fromarray(overlay_rgba).save(out_dir / overlay_name)
        # hit-mask needs a real alpha channel (not flat grayscale) so the browser
        # can isolate this space's shape via canvas compositing (source-in + fill)
        white = np.full_like(mask, 255)
        hitmask_rgba = np.dstack([white, white, white, mask])
        Image.fromarray(hitmask_rgba).save(out_dir / hitmask_name)

        manifest["spaces"].append({
            "id": slug,
            "name": name,
            "bbox": {
                "xPct": round(x0 / canvas_w * 100, 3),
                "yPct": round(y0 / canvas_h * 100, 3),
                "wPct": round((x1 - x0 + 1) / canvas_w * 100, 3),
                "hPct": round((y1 - y0 + 1) / canvas_h * 100, 3),
            },
            "overlay": overlay_name,
            "hitMask": hitmask_name,
        })
        print(f"{pdf_path.name} -> {name} ({slug}), bbox px ({x0},{y0})-({x1},{y1})")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest -> {manifest_path} ({len(manifest['spaces'])} spaces)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", required=True, help="folder containing the base + highlighted PDFs")
    ap.add_argument("--base", default="auto",
                     help="path to the base (unhighlighted) axo - a PDF or a raster image "
                          "(png/jpg, upscaled to match the PDF raster resolution if needed) - "
                          "or 'auto' (default) to reconstruct it as the per-pixel median of "
                          "every PDF in --in")
    ap.add_argument("--out", required=True, help="output folder for manifest.json + per-space PNGs")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    process(args.in_dir, args.base, args.out, args.dpi)


if __name__ == "__main__":
    main()
