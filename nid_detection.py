import os
# paddlepaddle 3.x + oneDNN hits a PIR runtime bug on some CPUs
# (ConvertPirAttribute2RuntimeAttribute ... onednn_instruction.cc). Disabling
# oneDNN routes to plain CPU kernels and avoids it. Must be set before paddle
# is imported.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
import re
from paddleocr import PaddleOCR
import argparse
import csv
import fnmatch
import gc
import html
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
NAF = HERE / "NAFNet"
if not NAF.exists():
    NAF = HERE / "NAFnet"
EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff")


_SRC_COLOR = {
    "contour": (0, 200, 0),
    "gdino": (255, 120, 0),
    "learned": (255, 200, 0),
    "sam2": (200, 0, 200),
}

def iter_images(in_dir: Path):
    return sorted(p for ext in EXTS for p in in_dir.glob(ext))


def copy_images(paths: list[Path], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, out_dir / path.name)
    return out_dir


# Albanian ID / passport personal number: one letter + 8 digits + one letter
# (format example only: A12345678Z). The "card no." field is 9 plain
# digits, so requiring the leading/trailing letter avoids confusing the two.
PERSONAL_RE = re.compile(r'[A-Z]\d{8}[A-Z]')
# Dates as DD-MM-YYYY (Albanian order) or YYYY-MM-DD, with . / - separators.
DATE_RE = re.compile(r'\b(\d{2})[./-](\d{2})[./-](\d{4})\b'
                     r'|\b(\d{4})[./-](\d{2})[./-](\d{2})\b')
EXPIRY_KEYWORDS = ("skadimit", "expiry", "date of expiry", "vlefshme")


def _parse_date(text: str):
    """Return (date_obj, matched_string) for the first date in text, or None."""
    m = DATE_RE.search(text)
    if not m:
        return None
    if m.group(1):                       # DD-MM-YYYY
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:                                # YYYY-MM-DD
        y, mo, d = int(m.group(4)), int(m.group(5)), int(m.group(6))
    try:
        return date(y, mo, d), m.group(0)
    except ValueError:
        return None


def make_ocr():
    """Construct a PaddleOCR reader across the 2.x and 3.x APIs.

    oneDNN is disabled (enable_mkldnn=False / FLAGS_use_mkldnn=0 above) to dodge
    the paddlepaddle 3.x PIR+oneDNN runtime bug on this CPU.
    """
    for kwargs in (
        dict(use_textline_orientation=True, lang='en', enable_mkldnn=False),
        dict(use_textline_orientation=True, lang='en'),          # 3.x, no flag
        dict(use_angle_cls=True, lang='en', show_log=False),     # 2.x
    ):
        try:
            return PaddleOCR(**kwargs)
        except TypeError:
            continue
    raise RuntimeError("could not construct PaddleOCR with any known signature")


def _ocr_lines(ocr, image_path: str):
    """Return [(text, conf, box 4x2 float32)] for either PaddleOCR API.

    3.x: ocr.predict() -> [dict(rec_texts, rec_scores, rec_polys/dt_polys, ...)]
    2.x: ocr.ocr(cls=True) -> [[[box, (text, conf)], ...]]
    """
    lines = []
    if hasattr(ocr, "predict"):           # PaddleOCR 3.x -- authoritative path
        results = ocr.predict(image_path)
        if results:
            r = results[0]
            get = r.get if hasattr(r, "get") else (lambda k, d=None: d)
            texts = get("rec_texts") or []
            scores = get("rec_scores") or []
            polys = get("rec_polys")
            if polys is None:
                polys = get("dt_polys") or []
            for i, (t, box) in enumerate(zip(texts, polys)):
                conf = float(scores[i]) if i < len(scores) else 1.0
                lines.append((str(t).strip(), conf,
                              np.array(box, dtype=np.float32).reshape(-1, 2)))
        return lines
    # PaddleOCR 2.x
    result = ocr.ocr(image_path, cls=True)
    if result and result[0]:
        for box, (text, conf) in result[0]:
            lines.append((str(text).strip(), float(conf),
                          np.array(box, dtype=np.float32)))
    return lines


def _draw_box(img, box: np.ndarray, color, label: str):
    pts = box.astype(int).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, 3)
    x, y = box.astype(int)[0]
    cv2.putText(img, label, (int(x), max(22, int(y) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def paddleocr_extract_text(image_path: str, out_path: str | None = None,
                           ocr: "PaddleOCR | None" = None) -> dict:
    """OCR an Albanian ID/passport crop and locate + validate two fields.

    Returns the personal number and expiry date, each with the OCR bounding
    box, checks whether the card is still in date, and writes an annotated copy
    with the two boxes drawn (green = valid expiry, red = expired).
    """
    if ocr is None:
        ocr = make_ocr()
    # every OCR line with its 4-point bounding box, API-version agnostic
    lines = _ocr_lines(ocr, image_path)

    # --- personal number (letter + 8 digits + letter) with its box ---
    # The "score" reported for each field is the OCR engine's own recognition
    # confidence for that text line (PaddleOCR rec_score, 0..1) -- a real model
    # output, not a hand-rolled heuristic.
    personal_number, personal_box, personal_conf = "Not Found", None, None
    for text, conf, box in lines:
        m = PERSONAL_RE.search(text.replace(" ", "").upper())
        if m:
            personal_number, personal_box, personal_conf = m.group(0), box, conf
            break

    # --- expiry date with its box ---
    # An ID carries date-of-birth, date-of-issue and date-of-expiry; the expiry
    # is always the LATEST of them, so that is the robust pick. If a line also
    # carries an expiry keyword we prefer that line's date.
    dated = []       # (date_obj, date_str, box, conf) for every dated line
    keyworded = []   # subset whose line also mentions an expiry keyword
    for text, conf, box in lines:
        parsed = _parse_date(text)
        if not parsed:
            continue
        d_obj, d_str = parsed
        dated.append((d_obj, d_str, box, conf))
        if any(k in text.lower() for k in EXPIRY_KEYWORDS):
            keyworded.append((d_obj, d_str, box, conf))

    expiry_obj = expiry_str = expiry_box = expiry_conf = None
    if keyworded:
        expiry_obj, expiry_str, expiry_box, expiry_conf = max(
            keyworded, key=lambda t: t[0])
    elif dated:
        expiry_obj, expiry_str, expiry_box, expiry_conf = max(
            dated, key=lambda t: t[0])

    # --- validity check against today ---
    today = date.today()
    if expiry_obj is not None:
        days = (expiry_obj - today).days
        valid = days >= 0
        status = (f"VALID - expires in {days} day(s) "
                  f"({expiry_obj:%d-%m-%Y})" if valid else
                  f"EXPIRED - {abs(days)} day(s) ago "
                  f"({expiry_obj:%d-%m-%Y})")
    else:
        valid, status = None, "expiry date not found - cannot verify validity"

    p_conf_txt = f"{personal_conf:.3f}" if personal_conf is not None else "-"
    e_conf_txt = f"{expiry_conf:.3f}" if expiry_conf is not None else "-"
    print(f"Personal number: {personal_number}  (OCR conf {p_conf_txt})")
    print(f"Expiry date:     {expiry_str or 'Not Found'}  (OCR conf {e_conf_txt})")
    print(f"Validity:        {status}")

    # --- draw the two boxes on a copy for visual verification ---
    img = cv2.imread(image_path)
    if img is not None:
        if personal_box is not None:
            _draw_box(img, personal_box, (255, 0, 0),
                      f"personal no: {personal_number} ({p_conf_txt})")
        if expiry_box is not None:
            color = (0, 170, 0) if valid else (0, 0, 255)
            _draw_box(img, expiry_box, color,
                      f"expiry: {expiry_str} ({e_conf_txt})")
        if out_path is None:
            p = Path(image_path)
            out_path = str(p.with_name(p.stem + "_fields.png"))
        cv2.imwrite(out_path, img)
        print(f"Annotated ->     {out_path}")

    return {
        "image": image_path,
        "personal_number": personal_number,
        "personal_box": None if personal_box is None else personal_box.tolist(),
        "personal_conf": personal_conf,
        "expiry_date": expiry_str,
        "expiry_box": None if expiry_box is None else expiry_box.tolist(),
        "expiry_conf": expiry_conf,
        "valid": valid,
        "status": status,
        "annotated": out_path,
    }


def _conf_txt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def write_fields_report(records: list[dict], report_path: Path) -> Path:
    """Self-contained HTML: annotated boxes + OCR confidence + validity.

    The only numeric score shown is the OCR engine's recognition confidence
    (PaddleOCR rec_score) for each field -- a standard model output.
    """
    report_dir = report_path.parent
    n = len(records)
    n_valid = sum(1 for r in records if r["valid"] is True)
    n_expired = sum(1 for r in records if r["valid"] is False)
    n_unknown = sum(1 for r in records if r["valid"] is None)

    parts = ["""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ID field extraction report</title><style>
*{box-sizing:border-box}body{margin:0;font-family:Arial,Helvetica,sans-serif;
background:#f6f7f9;color:#17202a}header{padding:16px 20px;background:#fff;
border-bottom:1px solid #d7dde5}h1{margin:0 0 6px;font-size:20px}
.summary{color:#5d6b7a;font-size:14px;display:flex;gap:14px;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));
gap:16px;padding:18px}.card{background:#fff;border:1px solid #d7dde5;
border-radius:10px;overflow:hidden;box-shadow:0 3px 12px rgba(28,39,54,.06)}
.card img{display:block;width:100%;height:auto;background:#fff}
.body{padding:12px 14px}.name{font-weight:700;font-size:13px;
overflow-wrap:anywhere;margin-bottom:8px}
.row{display:flex;justify-content:space-between;gap:10px;font-size:14px;
padding:3px 0}.row .k{color:#5d6b7a}.row .v{font-weight:700;
font-variant-numeric:tabular-nums;text-align:right;overflow-wrap:anywhere}
.conf{color:#5d6b7a;font-weight:400;font-size:12px}
.badge{display:inline-block;margin-top:10px;padding:4px 10px;border-radius:6px;
font-weight:700;font-size:13px;border:1px solid}
.ok{color:#1f7a4d;background:#eaf6ef;border-color:#9fcfb6}
.bad{color:#9b2d2d;background:#fbecec;border-color:#e1a8a8}
.unk{color:#5d6b7a;background:#eef2f6;border-color:#d7dde5}</style></head><body>
<header><h1>ID field extraction &mdash; personal number &amp; expiry</h1>
<div class="summary">"""]
    parts.append(
        f"<span>{n} image(s)</span><span>{n_valid} valid</span>"
        f"<span>{n_expired} expired</span><span>{n_unknown} undetermined</span>"
        "</div></header><div class=\"grid\">")

    for r in records:
        cls, badge = ("unk", html.escape(r["status"]))
        if r["valid"] is True:
            cls = "ok"
        elif r["valid"] is False:
            cls = "bad"
        name = html.escape(Path(r["image"]).name)
        pn = html.escape(str(r["personal_number"]))
        ed = html.escape(str(r["expiry_date"] or "Not Found"))
        pc = _conf_txt(r.get("personal_conf"))
        ec = _conf_txt(r.get("expiry_conf"))
        img_html = ""
        if r.get("annotated"):
            src = html.escape(os.path.relpath(r["annotated"], report_dir)
                              .replace("\\", "/"))
            img_html = f'<img src="{src}" alt="{name}">'
        parts.append(
            f'<div class="card">{img_html}<div class="body">'
            f'<div class="name">{name}</div>'
            f'<div class="row"><span class="k">Personal number</span>'
            f'<span class="v">{pn} <span class="conf">conf {pc}</span></span></div>'
            f'<div class="row"><span class="k">Expiry date</span>'
            f'<span class="v">{ed} <span class="conf">conf {ec}</span></span></div>'
            f'<div class="badge {cls}">{badge}</div>'
            f'</div></div>')

    parts.append("</div></body></html>")
    report_path.write_text("".join(parts), encoding="utf-8")
    return report_path


def main():
    ap = argparse.ArgumentParser(
        description="Read Albanian ID/passport personal number + expiry date "
                    "(with bounding boxes) and check the expiry is still valid.")
    ap.add_argument("input_dir", help="folder of ID/passport crop images")
    ap.add_argument("-o", "--output", default="id_fields",
                    help="output folder for annotated images + report")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        sys.exit(f"Not a folder: {in_dir}")
    paths = iter_images(in_dir)
    if not paths:
        sys.exit(f"No images found in {in_dir}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # initialise PaddleOCR once and reuse it across every image
    ocr = make_ocr()
    records = []
    for p in paths:
        print(f"\n=== {p.name} ===")
        out = str(out_dir / f"{p.stem}_fields.png")
        records.append(paddleocr_extract_text(str(p), out, ocr=ocr))

    # summary.csv, like the detection / reconstruction stages
    summary = out_dir / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "image", "personal_number", "personal_conf",
            "expiry_date", "expiry_conf", "valid", "status"])
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    report = write_fields_report(records, out_dir / "nid_report.html")
    ok = sum(1 for r in records if r["valid"] is True)
    exp = sum(1 for r in records if r["valid"] is False)
    print(f"\nProcessed {len(records)} image(s): {ok} valid, {exp} expired "
          f"-> {out_dir}/")
    print(f"Summary table -> {summary}")
    print(f"HTML report   -> {report}")

def save_page_debug(debug_dir: Path, stem: str, page_no: int, img: np.ndarray,
                    candidates, best_quad, score: float) -> Path:
    dbg = img.copy()
    for src, quad, sc in candidates:
        color = _SRC_COLOR.get(src, (150, 150, 150))
        cv2.polylines(dbg, [quad.astype(int)], True, color, 3)
        x, y = quad.astype(int)[0]
        cv2.putText(dbg, f"{src} {sc:.2f}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if best_quad is not None:
        cv2.polylines(dbg, [best_quad.astype(int)], True, (0, 255, 255), 6)
    label = f"page {page_no}  best={score:.3f}" + \
            ("" if best_quad is not None else "  (no ID/passport)")
    cv2.putText(dbg, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (0, 0, 255), 4)
    if max(dbg.shape[:2]) > 1400:
        s = 1400 / max(dbg.shape[:2])
        dbg = cv2.resize(dbg, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    debug_dir.mkdir(parents=True, exist_ok=True)
    out = debug_dir / f"{stem}_p{page_no:02d}_score{score:.3f}.jpg"
    cv2.imwrite(str(out), dbg)
    return out


if __name__ == "__main__":
    main()




