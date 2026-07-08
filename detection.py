"""
Stage 1/3 — DETECT the ID card in every document of a folder.

Accepts single images (jpg/png/...), multi-page PDFs and multi-page TIFFs.
For multi-page documents, every page is scanned and the page containing the
card is selected automatically. Works on both color and black & white scans.

For every document, each page is scored and the best-scoring page (the one
whose card detection is strongest) is kept. The detected document is then
cropped, perspective-corrected, rotated for reading and resized.

Outputs (into --output):
  <name>.png     the final normalized card/passport-details crop
  summary.csv    one table: per document, the page the card was found on
                 and the detection score
  debug_report.html plus annotated page images when --debug is used

Stage 2 (02_rotate.py) consumes the JSONs unchanged.

Requires: opencv-python, numpy, and pymupdf for PDF input
  pip install pymupdf

Usage:
  python 01_detect.py scans/ -o stage1_detected/
  python 01_detect.py scans/ -o stage1_detected/ --debug
"""

import argparse
import csv
import html
import sys
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import fitz  # PyMuPDF


CARD_W_MM = 85.6
CARD_H_MM = 53.98
ASPECT = CARD_W_MM / CARD_H_MM  # ~1.5857
MIN_CARD_SCORE = 0.40
STANDARD_CARD_W = 1016
STANDARD_CARD_H = 640

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TIFF_EXTS = {".tif", ".tiff"}
PDF_EXTS = {".pdf"}
ALL_EXTS = IMG_EXTS | TIFF_EXTS | PDF_EXTS


# ---------------------------------------------------------------- loading

def load_pages(path: Path, pdf_dpi: int = 300):
    """Yield (page_number_1_based, BGR image) for any supported input."""
    ext = path.suffix.lower()

    if ext in PDF_EXTS:
       
        doc = fitz.open(path)
        zoom = pdf_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.height, pix.width, pix.n)
            if pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            yield i, img
        doc.close()

    elif ext in TIFF_EXTS:
        ok, pages = cv2.imreadmulti(str(path), flags=cv2.IMREAD_COLOR)
        if not ok:
            single = cv2.imread(str(path))
            pages = [single] if single is not None else []
        for i, img in enumerate(pages, start=1):
            if img is not None:
                yield i, img

    else:
        img = cv2.imread(str(path))
        if img is not None:
            yield 1, img


def is_grayscale(img: np.ndarray) -> bool:
    """True for B&W scans (all channels equal / near-zero saturation)."""
    small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) < 8.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def band_score(value: float, low: float, good_low: float,
               good_high: float, high: float) -> float:
    """Soft score: 1 inside the good band, tapered toward low/high."""
    value = float(value)
    if value <= low or value >= high:
        return 0.0
    if good_low <= value <= good_high:
        return 1.0
    if value < good_low:
        return clamp01((value - low) / max(good_low - low, 1e-6))
    return clamp01((high - value) / max(high - good_high, 1e-6))


def warp_candidate(img: np.ndarray, quad: np.ndarray,
                   max_width: int | None = 720) -> np.ndarray | None:
    """Perspective-warp a quad and rotate it so the long side is horizontal."""
    side_w = (np.linalg.norm(quad[1] - quad[0]) +
              np.linalg.norm(quad[2] - quad[3])) / 2
    side_h = (np.linalg.norm(quad[3] - quad[0]) +
              np.linalg.norm(quad[2] - quad[1])) / 2
    out_w = max(24, int(round(side_w)))
    out_h = max(24, int(round(side_h)))
    if out_w < 24 or out_h < 24:
        return None

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1],
                    [0, out_h - 1]], dtype=np.float32)
    mat = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    roi = cv2.warpPerspective(img, mat, (out_w, out_h))

    if roi.shape[0] > roi.shape[1]:
        roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
    if max_width is not None and roi.shape[1] > max_width:
        scale = max_width / roi.shape[1]
        roi = cv2.resize(roi, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    return roi


def inner_gray(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    pad_x = max(1, int(w * 0.05))
    pad_y = max(1, int(h * 0.05))
    return gray[pad_y:h - pad_y, pad_x:w - pad_x]


def find_photo_blob(gray: np.ndarray) -> tuple[float, tuple[int, int, int, int] | None]:
    """Find a face/photo-like dark blob inside an ID/passport candidate."""
    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return 0.0, None

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cutoff = min(150.0, max(75.0, float(np.percentile(blur, 28))))
    mask = (blur < cutoff).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best_score = 0.0
    best_box = None
    area_total = h * w
    for cnt in contours:
        area = cv2.contourArea(cnt)
        frac = area / area_total
        if frac < 0.006 or frac > 0.22:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 12 or bh < 12:
            continue
        cx = x + bw / 2
        cy = y + bh / 2
        cx_norm = cx / max(w, 1)
        if cx_norm > 0.68:
            continue
        ratio = bh / max(bw, 1)
        fill = area / max(bw * bh, 1)
        if fill < 0.16:
            continue
        ratio_score = band_score(ratio, 0.45, 0.75, 2.3, 3.4)
        area_score = band_score(frac, 0.006, 0.018, 0.11, 0.22)
        fill_score = band_score(fill, 0.16, 0.28, 0.82, 1.0)
        x_position_score = band_score(cx_norm, -0.05, 0.08, 0.42, 0.68)
        y_position_score = 1.0 if cy > h * 0.15 else 0.65
        position_score = x_position_score * y_position_score
        score = ratio_score * area_score * fill_score * position_score
        if score > best_score:
            best_score = score
            best_box = (x, y, bw, bh)
    return best_score, best_box


def left_portrait_mass_score(gray: np.ndarray) -> float:
    """Estimate whether the strongest portrait-like dark mass is on the left."""
    h, w = gray.shape[:2]
    if h < 40 or w < 40:
        return 0.5

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cutoff = min(150.0, max(55.0, float(np.percentile(blur, 18))))
    mask = (blur < cutoff).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))

    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, 8)
    left_score = 0.0
    right_score = 0.0
    total = h * w
    for label in range(1, n_labels):
        x, y, bw, bh, area = stats[label]
        frac = area / max(total, 1)
        if frac < 0.004 or frac > 0.28:
            continue
        if bw < w * 0.035 or bh < h * 0.10:
            continue
        ratio = bh / max(bw, 1)
        ratio_score = band_score(ratio, 0.45, 0.75, 2.6, 4.0)
        area_score = band_score(frac, 0.004, 0.015, 0.16, 0.30)
        mass = ratio_score * area_score
        cx = (x + bw / 2) / max(w, 1)
        if cx < 0.50:
            left_score += mass
        else:
            right_score += mass

    return clamp01((left_score - right_score + 0.35) / 0.70)


def photo_blob_score(gray: np.ndarray) -> float:
    score, _ = find_photo_blob(gray)
    return score


def color_score(roi: np.ndarray) -> float:
    """Reward colored cards without penalizing B&W cards."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    sat_mean = float(sat.mean())
    sat_density = float((sat > 35).mean())
    return clamp01(0.45 * ((sat_mean - 12) / 50) +
                   0.55 * ((sat_density - 0.08) / 0.28))


def interior_detail_score(gray: np.ndarray) -> tuple[float, float, float]:
    edges = cv2.Canny(gray, 45, 140)
    edge_density = float((edges > 0).mean())
    ink_density = float((gray < 245).mean())
    contrast = float(gray.std())

    edge_score = band_score(edge_density, 0.010, 0.035, 0.20, 0.38)
    ink_score = band_score(ink_density, 0.035, 0.16, 0.78, 0.96)
    contrast_score = band_score(contrast, 8.0, 24.0, 95.0, 145.0)
    detail = 0.34 * edge_score + 0.33 * ink_score + 0.33 * contrast_score
    return detail, edge_density, ink_density


def form_grid_penalty(gray: np.ndarray, ink_density: float) -> float:
    """Penalize blank form boxes with several long internal ruling lines."""
    h, w = gray.shape[:2]
    if h < 40 or w < 40:
        return 1.0

    edges = cv2.Canny(gray, 45, 140)
    min_len = max(18, int(min(w, h) * 0.55))
    threshold = max(18, int(min(w, h) * 0.18))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold,
                            minLineLength=min_len,
                            maxLineGap=max(4, int(min(w, h) * 0.03)))
    if lines is None:
        return 1.0

    internal_long = 0
    for line in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = map(int, line)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        near_horizontal = dy <= max(3, int(0.03 * h))
        near_vertical = dx <= max(3, int(0.03 * w))
        if not (near_horizontal or near_vertical):
            continue

        if near_horizontal:
            y_mid = (y1 + y2) / 2
            if y_mid < h * 0.08 or y_mid > h * 0.92:
                continue
        if near_vertical:
            x_mid = (x1 + x2) / 2
            if x_mid < w * 0.08 or x_mid > w * 0.92:
                continue
        internal_long += 1

    penalty = 1.0 - 0.45 * clamp01((internal_long - 1) / 6)
    if ink_density < 0.18 and internal_long >= 2:
        penalty *= 0.45
    return clamp01(penalty)


def surrounding_score(page_gray: np.ndarray, quad: np.ndarray) -> float:
    """Prefer cards isolated on a page over rectangles embedded in forms."""
    h, w = page_gray.shape[:2]
    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    pad = max(12, int(max(bw, bh) * 0.18))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    crop = page_gray[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.5

    mask = np.ones(crop.shape[:2], dtype=np.uint8)
    ix0 = max(0, x - x0)
    iy0 = max(0, y - y0)
    ix1 = min(crop.shape[1], ix0 + bw)
    iy1 = min(crop.shape[0], iy0 + bh)
    mask[iy0:iy1, ix0:ix1] = 0
    ring_area = int(mask.sum())
    if ring_area < 50:
        return 0.7

    edges = cv2.Canny(crop, 45, 140)
    ring = mask.astype(bool)
    edge_density = float((edges[ring] > 0).mean())
    dark_density = float((crop[ring] < 230).mean())
    clean_edges = clamp01((0.075 - edge_density) / 0.075)
    clean_ink = clamp01((0.32 - dark_density) / 0.32)
    return 0.65 * clean_edges + 0.35 * clean_ink


def score_card_candidate(img: np.ndarray, page_gray: np.ndarray,
                         quad: np.ndarray, ratio: float, area_frac: float,
                         solidity: float, extent: float) -> float:
    """Confidence that one quadrilateral is an identity document."""
    aspect_score = band_score(ratio, 1.18, 1.40, 1.82, 2.12)
    area_score = band_score(area_frac, 0.008, 0.045, 0.55, 0.86)
    shape_score = (0.42 * aspect_score +
                   0.22 * clamp01(solidity) +
                   0.20 * clamp01(extent) +
                   0.16 * area_score)

    roi = warp_candidate(img, quad)
    if roi is None:
        return 0.0

    gray_roi = inner_gray(roi)
    detail, _edge_density, ink_density = interior_detail_score(gray_roi)
    photo = photo_blob_score(gray_roi)
    color = color_score(roi)
    mrz = mrz_like_score(roi)
    context = surrounding_score(page_gray, quad)
    grid_penalty = form_grid_penalty(gray_roi, ink_density)
    visual_identity = max(photo, color)
    mrz_identity = mrz if visual_identity >= 0.32 else 0.0
    # An MRZ is strong passport evidence only when the same crop also carries
    # visual identity cues. Otherwise ordinary form rows can masquerade as MRZ.
    identity_evidence = max(visual_identity, mrz_identity)

    # a warp that grabs scanner background shows up as large near-black
    # margins; that is not a real card no matter how card-like its aspect is
    black_frac = float((gray_roi < 25).mean())
    background_penalty = 1.0 - 0.7 * clamp01((black_frac - 0.06) / 0.30)

    surface = band_score(ink_density, 0.18, 0.32, 0.88, 0.98)
    content_score = (0.30 * detail + 0.30 * photo +
                     0.14 * color + 0.08 * mrz_identity +
                     0.18 * surface)
    score = (0.26 * shape_score + 0.40 * content_score +
             0.24 * context + 0.10 * area_score)
    if content_score < 0.22:
        score *= 0.65
    if context < 0.20:
        score *= 0.65
    if area_frac < 0.025 and identity_evidence < 0.35:
        score *= 0.55
    if surface < 0.22:
        score *= 0.45
    if identity_evidence < 0.12:
        score *= 0.32
    elif identity_evidence < 0.24:
        score *= 0.62
    elif identity_evidence > 0.45:
        score += 0.06 * identity_evidence

    # a clear photo / MRZ / colour print means this is a real document, not a
    # blank ruled form; don't let the form-grid heuristic punish its structure
    grid_penalty = grid_penalty + (1.0 - grid_penalty) * clamp01(identity_evidence)
    return clamp01(score * grid_penalty * background_penalty)


def rotate_to_landscape(img: np.ndarray) -> np.ndarray:
    if img.shape[0] > img.shape[1]:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def crop_bounds(img: np.ndarray, x0: int, y0: int,
                x1: int, y1: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(x0 + 1, min(w, int(x1)))
    y1 = max(y0 + 1, min(h, int(y1)))
    return img[y0:y1, x0:x1]


def trim_document_margins(img: np.ndarray,
                          pad_frac: float = 0.018) -> np.ndarray:
    """Trim scanner/page background around a detected card/passport crop."""
    h, w = img.shape[:2]
    if h < 80 or w < 120:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    background = float(np.percentile(blur, 94))
    cutoff = int(max(165, min(238, background - 16)))
    dark = (blur < cutoff).astype(np.uint8) * 255
    edges = cv2.Canny(blur, 45, 140)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    edge_support = (blur < min(250, background - 5)).astype(np.uint8) * 255
    edges = cv2.bitwise_and(edges, edge_support)
    mask = cv2.bitwise_or(dark, edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((17, 17), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    area_total = h * w
    useful = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area / area_total < 0.03:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < w * 0.25 or bh < h * 0.25:
            continue
        useful.append((area, x, y, bw, bh))
    if not useful:
        return img

    _area, x, y, bw, bh = max(useful, key=lambda item: item[0])
    if bw * bh > area_total * 0.94:
        return img

    pad = max(4, int(max(bw, bh) * pad_frac))
    return crop_bounds(img, x - pad, y - pad, x + bw + pad, y + bh + pad)


def text_band_density(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    dark = float((gray < 180).mean())
    edges = cv2.Canny(gray, 45, 140)
    edge_density = float((edges > 0).mean())
    return 0.55 * dark + 0.45 * edge_density


def text_character_mask(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Return text-like ink after removing table/border ruling lines."""
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8), 0.0

    block = max(15, min(51, (min(gray.shape[:2]) // 8) * 2 + 1))
    binary = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, 12)
    h, w = binary.shape[:2]
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(18, int(w * 0.22)), 1))
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(12, int(h * 0.28))))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    rules = cv2.bitwise_or(horizontal, vertical)
    rule_density = float((rules > 0).mean())
    chars = cv2.bitwise_and(binary, cv2.bitwise_not(rules))
    chars = cv2.morphologyEx(chars, cv2.MORPH_OPEN,
                             np.ones((2, 2), np.uint8))
    return chars, rule_density



def text_line_score(gray: np.ndarray) -> tuple[float, float, float]:
    """Score dense OCR/MRZ-like character rows, not form-table lines."""
    if gray.size == 0:
        return 0.0, 0.0, 1.0

    chars, rule_density = text_character_mask(gray)
    h, w = chars.shape[:2]
    if h < 16 or w < 40:
        return 0.0, 0.0, rule_density

    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        chars, 8)
    char_count = 0
    char_pixels = 0
    coverage = np.zeros(w, dtype=np.uint8)
    max_component_frac = 0.0
    for label in range(1, n_labels):
        x, y, bw, bh, area = stats[label]
        if area < 3:
            continue
        max_component_frac = max(max_component_frac, area / max(h * w, 1))
        if bw > w * 0.18 or bh > h * 0.45:
            continue
        if bh < 2 or bw < 1:
            continue
        char_count += 1
        char_pixels += int(area)
        coverage[x:x + bw] = 1

    char_density = char_pixels / max(h * w, 1)
    coverage_frac = float(coverage.mean())
    row_density = (chars > 0).mean(axis=1)
    kernel_len = max(3, h // 28)
    kernel = np.ones(kernel_len, dtype=np.float32) / kernel_len
    smooth = np.convolve(row_density, kernel, mode="same")
    row_threshold = max(0.018, float(smooth.mean() + 0.65 * smooth.std()))
    text_rows = float((smooth > row_threshold).mean())

    count_score = band_score(char_count, 6, 24, 180, 320)
    density_score = band_score(char_density, 0.006, 0.020, 0.18, 0.36)
    coverage_score = band_score(coverage_frac, 0.12, 0.35, 0.95, 1.0)
    rows_score = band_score(text_rows, 0.015, 0.05, 0.38, 0.70)
    large_blob_penalty = 1.0 - 0.65 * clamp01(
        (max_component_frac - 0.05) / 0.25)
    rule_penalty = 1.0 - 0.70 * clamp01((rule_density - 0.025) / 0.16)

    score = (0.30 * count_score + 0.26 * density_score +
             0.26 * coverage_score + 0.18 * rows_score)
    return clamp01(score * large_blob_penalty * rule_penalty), char_density, rule_density


def chevron_template(size: int, left: bool) -> np.ndarray:
    tmpl = np.zeros((size, size), dtype=np.uint8)
    thickness = max(1, size // 7)
    pad = max(1, size // 5)
    mid_y = size // 2
    if left:
        vertex = (pad, mid_y)
        cv2.line(tmpl, (size - pad - 1, pad), vertex, 255, thickness)
        cv2.line(tmpl, vertex, (size - pad - 1, size - pad - 1),
                 255, thickness)
    else:
        vertex = (size - pad - 1, mid_y)
        cv2.line(tmpl, (pad, pad), vertex, 255, thickness)
        cv2.line(tmpl, vertex, (pad, size - pad - 1), 255, thickness)
    return tmpl


def mrz_chevron_score(gray: np.ndarray) -> float:
    """Prefer upright MRZ rows, where filler characters look like '<'."""
    if gray.size == 0:
        return 0.0

    chars, _rule_density = text_character_mask(gray)
    if chars.shape[0] < 14 or chars.shape[1] < 60:
        return 0.0
    if chars.shape[1] > 900:
        scale = 900 / chars.shape[1]
        chars = cv2.resize(chars, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)

    left_hits = 0
    right_hits = 0
    for size in (9, 13, 17, 21):
        if chars.shape[0] <= size or chars.shape[1] <= size:
            continue
        left_tmpl = chevron_template(size, left=True)
        right_tmpl = chevron_template(size, left=False)
        left_match = cv2.matchTemplate(chars, left_tmpl,
                                       cv2.TM_CCOEFF_NORMED)
        right_match = cv2.matchTemplate(chars, right_tmpl,
                                        cv2.TM_CCOEFF_NORMED)
        left_hits += int((left_match > 0.45).sum())
        right_hits += int((right_match > 0.45).sum())

    total = left_hits + right_hits
    if total == 0:
        return 0.0
    presence = clamp01((left_hits - 2) / 30)
    balance = clamp01((left_hits - right_hits) / max(total, 1))
    return clamp01(0.65 * presence + 0.35 * balance)


def mrz_like_score(img: np.ndarray, force_landscape: bool = True) -> float:
    """Score a crop for passport-style dense MRZ/text near the bottom."""
    if force_landscape:
        img = rotate_to_landscape(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 60 or w < 120:
        return 0.0

    x0 = int(w * 0.04)
    x1 = int(w * 0.96)
    bottom = gray[int(h * 0.62):int(h * 0.96), x0:x1]
    top = gray[int(h * 0.06):int(h * 0.40), x0:x1]
    if bottom.size == 0:
        return 0.0

    bottom_text, bottom_density, bottom_rules = text_line_score(bottom)
    top_text, top_density, _top_rules = text_line_score(top)

    density_score = band_score(bottom_density, 0.006, 0.020, 0.17, 0.34)
    bottom_bias = clamp01((bottom_text - top_text + 0.10) / 0.42)
    rules_penalty = 1.0 - 0.55 * clamp01((bottom_rules - 0.035) / 0.18)
    return clamp01((0.58 * bottom_text +
                    0.24 * density_score +
                    0.18 * bottom_bias) * rules_penalty)


def passport_details_score(img: np.ndarray) -> tuple[float, float]:
    crop = rotate_to_landscape(img)
    gray = inner_gray(crop)
    detail, _edge_density, _ink_density = interior_detail_score(gray)
    photo = photo_blob_score(gray)
    mrz = mrz_like_score(crop)
    return 0.55 * mrz + 0.30 * photo + 0.15 * detail, mrz


def passport_details_score_any_orientation(
        img: np.ndarray) -> tuple[np.ndarray, float, float]:
    crop = rotate_to_landscape(img)
    candidates = [crop, cv2.rotate(crop, cv2.ROTATE_180)]
    best_crop = crop
    best_score = 0.0
    best_mrz = 0.0
    for candidate in candidates:
        score, mrz = passport_details_score(candidate)
        if score > best_score:
            best_crop = candidate
            best_score = score
            best_mrz = mrz
    return best_crop, best_score, best_mrz


def horizontal_text_score(img: np.ndarray) -> float:
    """Prefer rotations where text-like strokes form horizontal rows."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 40 or w < 40:
        return 0.5

    scale = 420 / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    binary = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
    y0 = int(binary.shape[0] * 0.05)
    y1 = int(binary.shape[0] * 0.95)
    x0 = int(binary.shape[1] * 0.05)
    x1 = int(binary.shape[1] * 0.95)
    binary = binary[y0:y1, x0:x1]
    if binary.size == 0:
        return 0.5

    row_var = float((binary > 0).mean(axis=1).std())
    col_var = float((binary > 0).mean(axis=0).std())
    return clamp01(row_var / max(row_var + col_var, 1e-6))


def id_header_score(img: np.ndarray) -> float:
    """Score an ID/passport crop for a title/security header near the top."""
    gray = inner_gray(rotate_to_landscape(img))
    h, w = gray.shape[:2]
    if h < 60 or w < 120:
        return 0.0

    band = gray[int(h * 0.02):int(h * 0.28), int(w * 0.05):int(w * 0.78)]
    if band.size == 0:
        return 0.0

    text_score, _density, rule_density = text_line_score(band)
    cutoff = min(110, max(45, int(np.percentile(band, 18))))
    dark = (band < cutoff).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    square_score = 0.0
    bh_total, bw_total = band.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < bw_total * 0.025 or bh < bh_total * 0.12:
            continue
        frac = area / max(bw_total * bh_total, 1)
        ratio = bw / max(bh, 1)
        component_score = (
            band_score(frac, 0.0015, 0.004, 0.08, 0.18) *
            band_score(ratio, 0.45, 0.70, 1.65, 2.20)
        )
        square_score = max(square_score, component_score)

    rule_penalty = 1.0 - 0.55 * clamp01((rule_density - 0.04) / 0.18)
    return clamp01((0.55 * square_score + 0.45 * text_score) * rule_penalty)


def crop_passport_details_if_needed(roi: np.ndarray) -> tuple[np.ndarray, str]:
    """Crop an open passport to the details/MRZ half when detectable."""
    roi = rotate_to_landscape(roi)
    h, w = roi.shape[:2]
    if h < 80 or w < 120:
        return roi, "id_card"

    spread_ratio = w / max(h, 1)

    # Only a clearly two-page-wide spread can be an open passport worth
    # splitting. A single ID card (~1.585) or single passport data page
    # (~1.42) must be kept whole -- splitting one would cut off half the
    # personal details, so we never do it below this ratio.
    if spread_ratio > 1.30:
        overlap_x = max(4, int(w * 0.035))
        left = crop_bounds(roi, 0, 0, w // 2 + overlap_x, h)
        right = crop_bounds(roi, w // 2 - overlap_x, 0, w, h)
        left_crop, left_score, left_mrz = (
            passport_details_score_any_orientation(left))
        right_crop, right_score, right_mrz = (
            passport_details_score_any_orientation(right))
        if left_score >= right_score:
            best_crop, best_score, best_mrz, other_score = (
                left_crop, left_score, left_mrz, right_score)
        else:
            best_crop, best_score, best_mrz, other_score = (
                right_crop, right_score, right_mrz, left_score)
        # keep the data-page half only if it carries a real MRZ and the facing
        # page is clearly weaker (the visa/emblem page has no MRZ or photo)
        required_gap = 0.08 if spread_ratio > 1.9 else 0.25
        if best_mrz >= 0.25 and best_score >= other_score + required_gap:
            return rotate_to_landscape(best_crop), "passport_details"

    # A single detected ID card is kept whole. A passport data page is also
    # kept whole; open passport spreads are labelled separately when the data
    # page can be split from the facing page.
    return roi, "id_card"


def reading_orientation_score(img: np.ndarray) -> float:
    gray = inner_gray(img)
    photo_score, box = find_photo_blob(gray)
    h, w = gray.shape[:2]

    face_left = 0.35
    if box is not None:
        x, _y, bw, _bh = box
        cx = (x + bw / 2) / max(w, 1)
        face_left = clamp01((0.72 - cx) / 0.52)

    top = gray[int(h * 0.05):int(h * 0.35), :]
    bottom = gray[int(h * 0.65):int(h * 0.95), :]
    top_density = text_band_density(top)
    bottom_density = text_band_density(bottom)
    bottom_bias = clamp01((bottom_density - top_density + 0.04) / 0.18)
    mrz = mrz_like_score(img, force_landscape=False)
    horizontal = horizontal_text_score(img)
    portrait_left = left_portrait_mass_score(gray)
    header = id_header_score(img)

    return (0.26 * horizontal +
            0.12 * photo_score * face_left +
            0.20 * mrz +
            0.32 * header +
            0.06 * bottom_bias +
            0.04 * portrait_left)


def orient_reading_side(img: np.ndarray,
                        allow_quarter_turns: bool = True) -> np.ndarray:
    candidates = [img, cv2.rotate(img, cv2.ROTATE_180)]
    if allow_quarter_turns:
        candidates.extend([
            cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ])
    original_score = reading_orientation_score(img)
    best = max(candidates, key=reading_orientation_score)
    if reading_orientation_score(best) > original_score + 0.10:
        return best
    return img


def passport_orientation_score(img: np.ndarray) -> float:
    img = rotate_to_landscape(img)
    gray = inner_gray(img)
    h, w = gray.shape[:2]
    if h < 40 or w < 40:
        return 0.0

    top = gray[int(h * 0.05):int(h * 0.35), :]
    bottom = gray[int(h * 0.64):int(h * 0.96), :]
    top_text, _top_density, _top_rules = text_line_score(top)
    bottom_text, _bottom_density, _bottom_rules = text_line_score(bottom)
    bottom_bias = clamp01((bottom_text - top_text + 0.06) / 0.40)
    bottom_chevrons = mrz_chevron_score(bottom)

    photo_score, box = find_photo_blob(gray)
    face_left = 0.35
    if box is not None:
        x, _y, bw, _bh = box
        cx = (x + bw / 2) / max(w, 1)
        face_left = clamp01((0.65 - cx) / 0.50)

    return (0.34 * bottom_chevrons +
            0.28 * mrz_like_score(img, force_landscape=False) +
            0.20 * bottom_bias +
            0.12 * photo_score * face_left +
            0.06 * horizontal_text_score(img))


def orient_passport_details(img: np.ndarray) -> np.ndarray:
    base = rotate_to_landscape(img)
    candidates = [base, cv2.rotate(base, cv2.ROTATE_180)]
    def combined_score(candidate: np.ndarray) -> float:
        return (0.55 * passport_orientation_score(candidate) +
                0.45 * reading_orientation_score(candidate))

    return max(candidates, key=combined_score)


def build_final_card_image(img: np.ndarray,
                           quad: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Return the normalized readable card/passport-details image."""
    crop = warp_candidate(img, quad, max_width=None)
    if crop is None:
        return None, ""
    crop, crop_type = crop_passport_details_if_needed(crop)
    trim_pad = 0.055 if crop_type == "passport_details" else 0.018
    crop = trim_document_margins(crop, pad_frac=trim_pad)
    if crop_type == "passport_details":
        crop = orient_passport_details(crop)
    else:
        crop = orient_reading_side(crop, allow_quarter_turns=True)
    crop = trim_document_margins(crop, pad_frac=trim_pad)
    final = cv2.resize(crop, (STANDARD_CARD_W, STANDARD_CARD_H),
                       interpolation=cv2.INTER_AREA)
    return final, crop_type


# ----------------- detection--------------

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def quad_dimensions(quad: np.ndarray) -> tuple[float, float]:
    side_w = (np.linalg.norm(quad[1] - quad[0]) +
              np.linalg.norm(quad[2] - quad[3])) / 2
    side_h = (np.linalg.norm(quad[3] - quad[0]) +
              np.linalg.norm(quad[2] - quad[1])) / 2
    return float(side_w), float(side_h)


def expanded_quad(quad: np.ndarray, left: float, right: float,
                  top: float, bottom: float,
                  page_w: int, page_h: int) -> np.ndarray:
    """Grow a quadrilateral in its own local axes and keep it on-page."""
    tl, tr, br, bl = quad.astype(np.float32)
    top_vec = tr - tl
    bottom_vec = br - bl
    left_vec = bl - tl
    right_vec = br - tr
    grown = np.array([
        tl - left * top_vec - top * left_vec,
        tr + right * top_vec - top * right_vec,
        br + right * bottom_vec + bottom * right_vec,
        bl - left * bottom_vec + bottom * left_vec,
    ], dtype=np.float32)
    grown[:, 0] = np.clip(grown[:, 0], 0, page_w - 1)
    grown[:, 1] = np.clip(grown[:, 1], 0, page_h - 1)
    return order_corners(grown)


def unique_quads(quads: list[np.ndarray]) -> list[np.ndarray]:
    unique = []
    for quad in quads:
        if not any(np.allclose(quad, existing, atol=2.0)
                   for existing in unique):
            unique.append(quad)
    return unique


def candidate_identity_strength(img: np.ndarray,
                                quad: np.ndarray) -> tuple[float, float]:
    roi = warp_candidate(img, quad)
    if roi is None:
        return 0.0, 0.0
    gray_roi = inner_gray(roi)
    detail, _edge_density, _ink_density = interior_detail_score(gray_roi)
    photo = photo_blob_score(gray_roi)
    color = color_score(roi)
    mrz = mrz_like_score(roi)
    passport_score, passport_mrz = passport_details_score(roi)
    visual_identity = max(photo, color)
    mrz_identity = max(mrz, passport_mrz) if visual_identity >= 0.32 else 0.0
    identity = max(visual_identity, mrz_identity)
    combined = identity
    if visual_identity >= 0.20:
        combined = max(identity, 0.50 * passport_score + 0.20 * detail)
    return clamp01(identity), clamp01(combined)


def expansion_variants_for_candidate(img: np.ndarray, quad: np.ndarray,
                                     page_w: int, page_h: int
                                     ) -> list[np.ndarray]:
    identity, combined = candidate_identity_strength(img, quad)
    variants = [quad]
    if identity < 0.18 and combined < 0.34:
        return variants

    # Small margin variants help copied/scanned cards whose contour falls
    # inside the printed border. The larger directional variants are for
    # passport data pages where the first contour often grabs only the text
    # block and misses the MRZ or the far edge.
    expansions = [
        (0.03, 0.03, 0.03, 0.03),
        (0.06, 0.06, 0.06, 0.18),
        (0.06, 0.06, 0.18, 0.06),
        (0.04, 0.16, 0.06, 0.38),
        (0.04, 0.16, 0.38, 0.06),
        (0.16, 0.04, 0.06, 0.38),
        (0.16, 0.04, 0.38, 0.06),
        (0.10, 0.10, 0.22, 0.22),
    ]
    for left, right, top, bottom in expansions:
        variants.append(expanded_quad(quad, left, right, top, bottom,
                                      page_w, page_h))
    return unique_quads(variants)


def find_card_quad(img: np.ndarray):
    """
    Best card-like quadrilateral in one page.
    Returns (quad 4x2 float32, score) or (None, 0.0).
    Score is comparable across pages of the same document.
    """
    h, w = img.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    candidates = []

    # A: Canny edges — works in color and B&W
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    candidates.append(edges)

    # B: adaptive threshold — uneven lighting, works in B&W
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 51, 10)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    candidates.append(thr)

    # C: Otsu on gray — B&W scans where the card is a darker block on paper
    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    candidates.append(otsu)

    # D: saturation channel — only meaningful on color scans
    if not is_grayscale(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, sat = cv2.threshold(hsv[:, :, 1], 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE,
                               np.ones((9, 9), np.uint8))
        candidates.append(sat)

    best_quad, best_score = None, 0.0

    for binary in candidates:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # too small to be a card, or large enough to be the sheet itself
            if area / img_area < 0.006:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                quad = order_corners(approx)
            else:
                rect = cv2.minAreaRect(cnt)
                quad = order_corners(cv2.boxPoints(rect))

            side_w, side_h = quad_dimensions(quad)
            if min(side_w, side_h) < 20:
                continue

            quad_area = abs(cv2.contourArea(quad.astype(np.float32)))
            frac = quad_area / img_area
            if frac < 0.008 or frac > 0.86:
                continue

            # ID-1 cards are ~1.586, but scans may include passport pages,
            # copied margins, or mild perspective distortion.
            ratio = max(side_w, side_h) / min(side_w, side_h)
            if ratio < 1.18 or ratio > 2.12:
                continue

            # a genuine document lies (almost) entirely within the page; a
            # skewed shadow / sheet-edge quad has corners that fall off-page.
            # rejecting those removes a common false detection.
            margin_x, margin_y = 0.02 * w, 0.02 * h
            if (quad[:, 0].min() < -margin_x or
                    quad[:, 0].max() > w + margin_x or
                    quad[:, 1].min() < -margin_y or
                    quad[:, 1].max() > h + margin_y):
                continue

            # classic contour shape descriptors (OpenCV "Contour Properties"):
            #   solidity = contour area / convex-hull area
            #   extent   = contour area / bounding-box area (rectangularity)
            # both are in [0, 1]; a clean, filled card rectangle scores near 1.
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area / hull_area if hull_area > 0 else 0.0
            variants = expansion_variants_for_candidate(img, quad, w, h)
            for test_quad in variants:
                test_w, test_h = quad_dimensions(test_quad)
                if min(test_w, test_h) < 20:
                    continue

                test_area = abs(cv2.contourArea(
                    test_quad.astype(np.float32)))
                test_frac = test_area / img_area
                if test_frac < 0.008 or test_frac > 0.86:
                    continue

                test_ratio = max(test_w, test_h) / min(test_w, test_h)
                if test_ratio < 1.12 or test_ratio > 2.20:
                    continue

                test_extent = min(area, test_area) / max(test_area, 1.0)
                score = score_card_candidate(img, gray, test_quad,
                                             test_ratio, test_frac,
                                             solidity, test_extent)

                if score > best_score:
                    best_score, best_quad = score, test_quad

    if best_score < MIN_CARD_SCORE:
        return None, best_score
    return best_quad, best_score


# ---------------------------------------------------------------- driver

def detect_page(img: np.ndarray):
    """Detect on a bounded copy, return full-res (quad, score)."""
    scale = 1.0
    det = img
    max_side = 1600
    if max(img.shape[:2]) > max_side:
        scale = max_side / max(img.shape[:2])
        det = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    quad, score = find_card_quad(det)
    if quad is not None:
        quad = quad / scale
    return quad, score


def save_page_debug(debug_dir: Path, stem: str, page_no: int,
                    img: np.ndarray, quad, score: float) -> Path:
    """Write an annotated copy of ONE page so per-page detection is visible."""
    dbg = img.copy()
    if quad is not None:
        cv2.polylines(dbg, [quad.astype(int)], True, (0, 255, 0), 6)
        for (x, y) in quad.astype(int):
            cv2.circle(dbg, (x, y), 12, (0, 0, 255), -1)
    label = f"page {page_no}  score={score:.3f}" + \
            ("" if quad is not None else "  (no ID/passport)")
    cv2.putText(dbg, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (0, 0, 255), 4)
    # shrink so the debug folder stays small
    if max(dbg.shape[:2]) > 1400:
        s = 1400 / max(dbg.shape[:2])
        dbg = cv2.resize(dbg, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    debug_dir.mkdir(parents=True, exist_ok=True)
    out = debug_dir / f"{stem}_p{page_no:02d}_score{score:.3f}.jpg"
    cv2.imwrite(str(out), dbg)
    return out


def html_attr(value) -> str:
    return html.escape(str(value), quote=True)


def html_url(path_text: str) -> str:
    return quote(path_text.replace("\\", "/"), safe="/._-()")


def display_status_label(status: str) -> str:
    if status == "no_card":
        return "ID or passport is missing"
    if status == "ok":
        return "ID/passport detected"
    return status.replace("_", " ")


def write_debug_report(out_dir: Path, reports: list[dict]) -> Path:
    """Write an interactive HTML report for per-page detection QA."""
    report_path = out_dir / "debug_report.html"
    total_docs = len(reports)
    ok_docs = sum(1 for doc in reports if doc["status"] == "ok")
    total_pages = sum(len(doc["pages"]) for doc in reports)
    threshold = f"{MIN_CARD_SCORE:.2f}"

    parts = ["""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ID detection debug report</title>
<style>
:root {
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #17202a;
  --muted: #5d6b7a;
  --line: #d7dde5;
  --accent: #1f7a4d;
  --warn: #9a5b00;
  --bad: #9b2d2d;
  --shadow: 0 10px 30px rgba(28, 39, 54, 0.12);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
  color: var(--ink);
  background: var(--bg);
}
header {
  position: sticky;
  top: 0;
  z-index: 5;
  background: var(--panel);
  border-bottom: 1px solid var(--line);
  box-shadow: 0 2px 10px rgba(28, 39, 54, 0.06);
}
.topbar {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 12px;
  align-items: center;
  padding: 14px 18px;
}
h1 {
  margin: 0;
  font-size: 20px;
  letter-spacing: 0;
}
.summary {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  color: var(--muted);
  font-size: 13px;
}
.controls {
  display: flex;
  gap: 8px;
  align-items: center;
}
input, select {
  height: 36px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0 10px;
  background: #fff;
  color: var(--ink);
  font-size: 14px;
}
.layout {
  display: grid;
  grid-template-columns: 300px minmax(0, 1fr);
  min-height: calc(100vh - 65px);
}
nav {
  border-right: 1px solid var(--line);
  background: #eef2f6;
  padding: 12px;
  overflow: auto;
}
.doc-button {
  width: 100%;
  display: grid;
  gap: 6px;
  text-align: left;
  border: 1px solid transparent;
  border-radius: 8px;
  padding: 10px;
  background: transparent;
  color: var(--ink);
  cursor: pointer;
  margin-bottom: 8px;
}
.doc-button:hover, .doc-button.active {
  background: #fff;
  border-color: var(--line);
}
.doc-button.active {
  box-shadow: var(--shadow);
}
.doc-title {
  font-weight: 700;
  font-size: 13px;
  line-height: 1.25;
  overflow-wrap: anywhere;
}
.doc-meta {
  color: var(--muted);
  font-size: 12px;
}
main {
  padding: 18px;
  overflow: auto;
}
.doc-section {
  display: none;
}
.doc-section.active {
  display: block;
}
.doc-header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  margin-bottom: 16px;
}
.doc-header h2 {
  margin: 0 0 6px;
  font-size: 20px;
}
.badge {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 3px 8px;
  border-radius: 6px;
  font-weight: 700;
  font-size: 12px;
  border: 1px solid var(--line);
  background: #fff;
}
.badge.ok { color: var(--accent); border-color: #9fcfb6; }
.badge.no_card, .badge.unreadable { color: var(--bad); border-color: #e1a8a8; }
.missing-banner {
  margin: 0 0 16px;
  padding: 12px 14px;
  border: 1px solid #e1a8a8;
  border-radius: 8px;
  background: #fff4f4;
  color: var(--bad);
  font-weight: 700;
}
.final-output {
  display: grid;
  grid-template-columns: minmax(180px, 360px) minmax(0, 1fr);
  gap: 14px;
  align-items: center;
  margin: 0 0 16px;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
}
.final-output img {
  display: block;
  width: 100%;
  height: auto;
  border: 1px solid var(--line);
}
.final-output strong {
  display: block;
  margin-bottom: 6px;
}
.pages {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
  align-items: start;
}
.page-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 3px 12px rgba(28, 39, 54, 0.06);
}
.page-card.best {
  border-color: #4faf7d;
  box-shadow: 0 0 0 2px rgba(79, 175, 125, 0.2), var(--shadow);
}
.page-head {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: center;
  padding: 10px;
  border-bottom: 1px solid var(--line);
}
.score {
  font-variant-numeric: tabular-nums;
  font-weight: 700;
}
.thumb {
  display: block;
  width: 100%;
  padding: 0;
  border: 0;
  background: #fff;
  cursor: zoom-in;
}
.thumb img {
  display: block;
  width: 100%;
  height: auto;
}
.page-foot {
  padding: 10px;
  color: var(--muted);
  font-size: 12px;
}
.bar {
  height: 8px;
  background: #e7ebf0;
  border-radius: 999px;
  overflow: hidden;
  margin-top: 8px;
}
.bar span {
  display: block;
  height: 100%;
  background: var(--accent);
}
.page-card.low .bar span {
  background: var(--warn);
}
.page-card.no-detection .bar span {
  background: var(--bad);
}
.empty {
  padding: 28px;
  color: var(--muted);
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
}
.modal {
  position: fixed;
  inset: 0;
  display: none;
  z-index: 20;
  background: rgba(11, 18, 28, 0.82);
  padding: 20px;
}
.modal.open {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  gap: 12px;
}
.modal-top {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: #fff;
}
.modal-title {
  font-weight: 700;
}
.modal button.close {
  border: 1px solid rgba(255,255,255,0.5);
  border-radius: 6px;
  background: rgba(255,255,255,0.12);
  color: #fff;
  height: 34px;
  padding: 0 12px;
  cursor: pointer;
}
.modal img {
  max-width: 100%;
  max-height: 100%;
  margin: auto;
  display: block;
  box-shadow: var(--shadow);
}
@media (max-width: 800px) {
  .topbar, .layout, .doc-header {
    display: block;
  }
  .controls {
    margin-top: 10px;
    flex-wrap: wrap;
  }
  .final-output {
    grid-template-columns: 1fr;
  }
  nav {
    border-right: 0;
    border-bottom: 1px solid var(--line);
    max-height: 260px;
  }
}
</style>
</head>
<body>
<header>
  <div class="topbar">
    <div>
      <h1>ID detection debug report</h1>
      <div class="summary">"""]

    parts.append(
        f"<span>{total_docs} documents</span>"
        f"<span>{ok_docs} detected</span>"
        f"<span>{total_pages} pages</span>"
        f"<span>threshold {threshold}</span>"
    )
    parts.append("""</div>
    </div>
    <div class="controls">
      <input id="search" type="search" placeholder="Search documents">
      <select id="statusFilter" aria-label="Status filter">
        <option value="all">All statuses</option>
        <option value="ok">ID/passport detected</option>
        <option value="no_card">ID/passport missing</option>
        <option value="unreadable">Unreadable</option>
      </select>
    </div>
  </div>
</header>
<div class="layout">
<nav id="docNav">""")

    for i, doc in enumerate(reports):
        active = " active" if i == 0 else ""
        title = html_attr(doc["document"])
        status = html_attr(doc["status"])
        status_label = display_status_label(doc["status"])
        best = doc.get("best_page")
        best_text = f"best page {best}" if best else status_label
        score = doc.get("detection_score")
        score_text = f"score {score:.3f}" if isinstance(score, float) else ""
        crop_type = doc.get("crop_type") or ""
        meta = html_attr(
            f"{doc['total_pages']} pages, {best_text}"
            + (f", {score_text}" if score_text else "")
            + (f", {crop_type}" if crop_type else "")
        )
        parts.append(
            f'<button class="doc-button{active}" data-index="{i}" '
            f'data-status="{status}" data-name="{title}">'
            f'<span class="doc-title">{title}</span>'
            f'<span class="doc-meta">{meta}</span>'
            f'</button>'
        )

    parts.append("""</nav>
<main id="docMain">""")

    for i, doc in enumerate(reports):
        active = " active" if i == 0 else ""
        title = html_attr(doc["document"])
        status = html_attr(doc["status"])
        status_label = display_status_label(doc["status"])
        best = doc.get("best_page")
        score = doc.get("detection_score")
        score_text = f"{score:.3f}" if isinstance(score, float) else "-"
        crop_type = doc.get("crop_type") or ""
        status_class = html_attr(status)
        parts.append(
            f'<section class="doc-section{active}" data-index="{i}" '
            f'data-status="{status_class}" data-name="{title}">'
            '<div class="doc-header">'
            '<div>'
            f'<h2>{title}</h2>'
            f'<div class="doc-meta">{doc["total_pages"]} pages'
            + (f' | selected page {best}' if best else
               f' | {html_attr(status_label)}')
            + f' | best score {score_text}'
            + (f' | output {html_attr(crop_type)}' if crop_type else '')
            + '</div>'
            '</div>'
            f'<span class="badge {status_class}">'
            f'{html_attr(status_label)}</span>'
            '</div>'
        )

        if doc["status"] == "no_card":
            parts.append(
                '<div class="missing-banner">'
                'ID or passport is missing from this document.'
                '</div>'
            )
        elif doc.get("output_image"):
            output_src = html_url(doc["output_image"])
            output_title = html_attr(
                f"{doc['document']} - final normalized {crop_type}"
            )
            parts.append(
                '<div class="final-output">'
                f'<button class="thumb" data-src="{output_src}" '
                f'data-title="{output_title}" type="button">'
                f'<img src="{output_src}" alt="{output_title}">'
                '</button>'
                '<div>'
                '<strong>Final saved image</strong>'
                f'<div class="doc-meta">type {html_attr(crop_type)} | '
                f'{STANDARD_CARD_W} x {STANDARD_CARD_H}px</div>'
                '</div>'
                '</div>'
            )

        if not doc["pages"]:
            parts.append('<div class="empty">No readable pages.</div>')
        else:
            parts.append('<div class="pages">')
            for page in doc["pages"]:
                page_no = page["page_no"]
                page_score = float(page["score"])
                score_pct = int(round(clamp01(page_score) * 100))
                is_best = page.get("is_best", False)
                has_card = page.get("has_card", False)
                classes = ["page-card"]
                if is_best:
                    classes.append("best")
                if not has_card:
                    classes.append("no-detection")
                elif page_score < MIN_CARD_SCORE:
                    classes.append("low")
                label = "selected" if is_best else (
                    "detected" if has_card else "No ID/passport detected"
                )
                src = html_url(page["debug_image"])
                modal_title = html_attr(
                    f"{doc['document']} - page {page_no} - score "
                    f"{page_score:.3f}"
                )
                parts.append(
                    f'<article class="{" ".join(classes)}">'
                    '<div class="page-head">'
                    f'<strong>Page {page_no}</strong>'
                    f'<span class="score">{page_score:.3f}</span>'
                    '</div>'
                    f'<button class="thumb" data-src="{src}" '
                    f'data-title="{modal_title}" type="button">'
                    f'<img src="{src}" alt="{modal_title}">'
                    '</button>'
                    '<div class="page-foot">'
                    f'<div>{html_attr(label)}</div>'
                    '<div class="bar">'
                    f'<span style="width:{score_pct}%"></span>'
                    '</div>'
                    '</div>'
                    '</article>'
                )
            parts.append('</div>')
        parts.append('</section>')

    parts.append("""</main>
</div>
<div id="modal" class="modal" aria-hidden="true">
  <div class="modal-top">
    <div id="modalTitle" class="modal-title"></div>
    <button class="close" type="button">Close</button>
  </div>
  <img id="modalImage" alt="">
</div>
<script>
const buttons = Array.from(document.querySelectorAll('.doc-button'));
const sections = Array.from(document.querySelectorAll('.doc-section'));
const search = document.getElementById('search');
const statusFilter = document.getElementById('statusFilter');

function selectDoc(index) {
  buttons.forEach(btn => btn.classList.toggle('active', btn.dataset.index === String(index)));
  sections.forEach(sec => sec.classList.toggle('active', sec.dataset.index === String(index)));
}

function applyFilters() {
  const term = search.value.trim().toLowerCase();
  const status = statusFilter.value;
  let firstVisible = null;
  buttons.forEach(btn => {
    const nameMatch = btn.dataset.name.toLowerCase().includes(term);
    const statusMatch = status === 'all' || btn.dataset.status === status;
    const visible = nameMatch && statusMatch;
    btn.hidden = !visible;
    if (visible && firstVisible === null) firstVisible = btn.dataset.index;
  });
  const active = buttons.find(btn => btn.classList.contains('active') && !btn.hidden);
  if (!active && firstVisible !== null) selectDoc(firstVisible);
}

buttons.forEach(btn => {
  btn.addEventListener('click', () => selectDoc(btn.dataset.index));
});
search.addEventListener('input', applyFilters);
statusFilter.addEventListener('change', applyFilters);

const modal = document.getElementById('modal');
const modalImage = document.getElementById('modalImage');
const modalTitle = document.getElementById('modalTitle');
document.querySelectorAll('.thumb').forEach(btn => {
  btn.addEventListener('click', () => {
    modalImage.src = btn.dataset.src;
    modalImage.alt = btn.dataset.title;
    modalTitle.textContent = btn.dataset.title;
    modal.classList.add('open');
    modal.setAttribute('aria-hidden', 'false');
  });
});
function closeModal() {
  modal.classList.remove('open');
  modal.setAttribute('aria-hidden', 'true');
  modalImage.src = '';
}
modal.querySelector('.close').addEventListener('click', closeModal);
modal.addEventListener('click', event => {
  if (event.target === modal) closeModal();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && modal.classList.contains('open')) closeModal();
});
</script>
</body>
</html>
""")

    report_path.write_text("".join(parts), encoding="utf-8")
    return report_path


