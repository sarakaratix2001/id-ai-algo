"""
Stage 1 (ML cascade) — detect and crop ONLY the ID / passport.

This layers modern learned models on top of the classical detector in
detection.py, combining three complementary tools:

  1. OpenCV contour detection      (detection.find_card_quad)
        fast, exact corners when the card has clear borders on a clean page.
  2. Grounding DINO (open-vocab)   zero-shot object detection
        finds an "identity card / passport" by text prompt, with no training
        and no labelled data -- handles pages where no clean rectangle exists.
  3. SAM 2 mask refinement         Segment-Anything-style masking
        turns a detector box into a pixel-accurate mask, so a card lying on a
        cluttered / low-contrast background is cropped tightly and de-skewed.

Every candidate quadrilateral -- from contours, from Grounding DINO, and from
SAM 2 -- is scored by the SAME identity heuristic used by the classical
pipeline (detection.score_card_candidate: portrait blob + MRZ + colour). The
highest-scoring quad wins, then detection.build_final_card_image performs the
shared warp / orient / normalise. So the models only *propose* regions; the
proven scoring decides which one is really the ID.

YOLOv8 / YOLOv9 / RT-DETR fit the exact same "propose boxes" slot as Grounding
DINO -- see YoloDetector / RTDetrDetector below. They are wired but disabled by
default because a COCO-pretrained weight has no "ID card" class; point them at a
fine-tuned weight to enable.

Usage:
  python detect_transform.py scans/ -o stage1_ml/ --debug
  python detect_transform.py scans/ -o stage1_ml/ --no-sam    # boxes only
  python detect_transform.py scans/ -o stage1_ml/ --no-ml     # classical only
  python detect_transform.py scans/ -o stage1_ml/ --gdino-model IDEA-Research/grounding-dino-base
  # detect_ml.py is a thin wrapper that calls this module's main().
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

import detection as D


# ---------------------------------------------------------------- geometry

def box_to_quad(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def pad_box(box, w: int, h: int, frac: float = 0.04):
    x0, y0, x1, y1 = box
    px = (x1 - x0) * frac
    py = (y1 - y0) * frac
    return (max(0.0, x0 - px), max(0.0, y0 - py),
            min(w - 1.0, x1 + px), min(h - 1.0, y1 + py))


def score_quad(img: np.ndarray, page_gray: np.ndarray,
               quad: np.ndarray, solidity: float = 0.97) -> float:
    """Score any proposed quad with the classical identity heuristic.

    Contour candidates arrive with real shape descriptors; learned boxes are
    clean rectangles, so we derive extent from the quad itself and pass a high
    solidity. This keeps a single, comparable score across all sources.
    """
    h, w = page_gray.shape[:2]
    side_w, side_h = D.quad_dimensions(quad)
    if min(side_w, side_h) < 20:
        return 0.0
    quad_area = abs(cv2.contourArea(quad.astype(np.float32)))
    area_frac = quad_area / float(h * w)
    if area_frac < 0.004 or area_frac > 0.92:
        return 0.0
    ratio = max(side_w, side_h) / max(min(side_w, side_h), 1.0)
    bx, by, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    extent = quad_area / max(bw * bh, 1.0)
    return D.score_card_candidate(img, page_gray, quad, ratio, area_frac,
                                  solidity, extent)


def _valid_card_rect(rect) -> bool:
    (cw, ch) = rect[1]
    if min(cw, ch) < 24:
        return False
    ratio = max(cw, ch) / min(cw, ch)
    return 1.15 < ratio < 2.35


def tighten_to_card(page: np.ndarray, quad: np.ndarray,
                    sam: "Sam2Refiner | None" = None,
                    pad: float = 0.06) -> np.ndarray:
    """Snap ANY card quad to the card's exact oriented 4 corners.

    A detector box or a loose contour is often bigger than the card or not
    aligned to its edges, so warping it leaves the card tilted with white
    triangular gaps. This recovers the card's true rotated rectangle so the
    downstream perspective warp deskews it with the corners flush to the crop.

    Generalisable, not tied to any one document:
      1. Card-on-page: the card is the connected 'not-white' region inside the
         padded box -> largest contour -> minAreaRect gives the oriented corners.
      2. Fallback for cluttered / non-white backgrounds: SAM 2 segments the card
         from the box prompt and its mask's minAreaRect gives the corners.
      3. If neither yields a card-shaped rectangle, the original quad is kept
         unchanged (never makes a good crop worse).
    """
    h, w = page.shape[:2]
    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    px, py = int(bw * pad), int(bh * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
    crop = page[y0:y1, x0:x1]
    if crop.size == 0:
        return quad

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    white = (gray > 236) & (sat < 22)
    card = (~white).astype(np.uint8) * 255
    k = max(5, int(min(crop.shape[:2]) * 0.03))
    card = cv2.morphologyEx(card, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    card = cv2.morphologyEx(card, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(card, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        region = crop.shape[0] * crop.shape[1]
        rect = cv2.minAreaRect(c)
        if 0.15 < cv2.contourArea(c) / region < 0.99 and _valid_card_rect(rect):
            box = cv2.boxPoints(rect).astype(np.float32)
            box[:, 0] += x0
            box[:, 1] += y0
            return D.order_corners(box)

    if sam is not None:
        try:
            quad_s = sam.refine(page, (x, y, x + bw, y + bh))
        except Exception:
            quad_s = None
        if quad_s is not None:
            sw, sh = D.quad_dimensions(quad_s)
            if _valid_card_rect(((0, 0), (sw, sh), 0)):
                return quad_s

    return quad


# ---------------------------------------------------------------- ML backends

class GroundingDinoDetector:
    """Zero-shot open-vocabulary detector -- the PRIMARY learned detector.

    WHEN: loaded whenever the ML cascade is on (i.e. not --no-ml). Runs on
    every page and proposes candidate ID/passport boxes.

    WHY WE NEED IT: the classical contour detector only works when the card has
    a clean rectangular border on a plain page; it fails on cluttered pages,
    photos-of-a-card, or forms where no crisp rectangle exists. A learned
    detector fills that gap -- but COCO-pretrained YOLO/RT-DETR have no "ID card"
    class and we have no labelled data to fine-tune one. Grounding DINO is
    open-vocabulary: it detects whatever the text PROMPT names, so it finds an
    ID/passport with zero training. That is exactly why it is the default over
    the YOLO/RT-DETR slots below.

    It only PROPOSES rough axis-aligned boxes; Sam2Refiner turns those into
    tight oriented crops, and score_quad decides which proposal is the real ID.
    """

    PROMPT = "an identity card. a passport. an id card. a driver license."

    def __init__(self, model_id: str, device: str, box_threshold: float = 0.25,
                 text_threshold: float = 0.20):
        import torch
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
        self.torch = torch
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            model_id).to(device).eval()

    def _post_process(self, outputs, input_ids, target_size):
        """Call post_process across the arg names used by different versions."""
        pp = self.processor.post_process_grounded_object_detection
        attempts = [
            dict(input_ids=input_ids, threshold=self.box_threshold,
                 text_threshold=self.text_threshold,
                 target_sizes=[target_size]),
            dict(input_ids=input_ids, box_threshold=self.box_threshold,
                 text_threshold=self.text_threshold,
                 target_sizes=[target_size]),
            dict(threshold=self.box_threshold,
                 text_threshold=self.text_threshold,
                 target_sizes=[target_size]),
        ]
        last = None
        for kwargs in attempts:
            try:
                return pp(outputs, **kwargs)
            except TypeError as exc:
                last = exc
        raise last

    def detect(self, img_bgr: np.ndarray):
        from PIL import Image
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, text=self.PROMPT,
                                return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        results = self._post_process(outputs, inputs["input_ids"],
                                     (pil.height, pil.width))[0]
        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        out = []
        for (x0, y0, x1, y1), s in zip(boxes, scores):
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            out.append(((float(x0), float(y0), float(x1), float(y1)),
                        float(s)))
        return out


class Sam2Refiner:
    """Segment-Anything-2 box-prompted masking -> tight oriented quad.

    WHEN: loaded when the cascade is on and --no-sam is NOT set. It runs after
    a detector (Grounding DINO / YOLO / RT-DETR) has proposed a box; it never
    detects on its own -- it always needs a box prompt.

    WHY WE NEED IT: a detector box is axis-aligned and usually a bit loose, so
    warping it to a rectangle leaves the card tilted with white gaps at the
    corners, and it may include background. SAM 2 segments the exact card
    pixels inside the prompt box, and minAreaRect on that mask yields the card's
    true ROTATED four corners. Feeding those to the perspective warp deskews the
    card so the corners sit flush -- and it recovers the card even on
    low-contrast / cluttered backgrounds where an edge/contour approach can't
    trace a border. In short: Grounding DINO answers "where is the ID?",
    SAM 2 answers "what are its exact corners?".

    We use SAM 2 (not SAM 1) because it produces sharper object masks at lower
    cost; Sam2Model + Sam2Processor come from transformers, so no extra install.
    """

    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import Sam2Processor, Sam2Model
        self.torch = torch
        self.device = device
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(device).eval()

    def refine(self, img_bgr: np.ndarray, box):
        from PIL import Image
        h, w = img_bgr.shape[:2]
        px_box = pad_box(box, w, h, 0.03)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        # input_boxes is [batch][boxes][xyxy]; one image, one prompt box.
        inputs = self.processor(images=pil, input_boxes=[[list(px_box)]],
                                return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        masks = self.processor.post_process_masks(
            outputs.pred_masks, original_sizes=[[h, w]])
        mask = masks[0]
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask)
        # collapse leading (num_boxes) dims, keep the (num_masks, H, W) block
        mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])
        # SAM 2 emits 3 candidate masks per box; prefer the highest-IoU one,
        # falling back to the largest by area if scores are unavailable.
        idx = 0
        iou = getattr(outputs, "iou_scores", None)
        if iou is not None:
            iou = iou.detach().cpu().numpy().reshape(-1)
            if iou.shape[0] == mask.shape[0]:
                idx = int(np.argmax(iou))
        else:
            idx = int(np.argmax(mask.reshape(mask.shape[0], -1).sum(axis=1)))
        binary = (mask[idx] > 0).astype(np.uint8) * 255
        return self._quad_from_mask(binary)

    @staticmethod
    def _quad_from_mask(binary: np.ndarray):
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 400:
            return None
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return D.order_corners(approx)
        rect = cv2.minAreaRect(cnt)
        return D.order_corners(cv2.boxPoints(rect))


# ---------------------------------------------------------------- optional
# YOLOv8/YOLOv9 and RT-DETR occupy the same "propose boxes" slot. They need a
# weight whose classes include an ID card; a plain COCO weight will not detect
# one. Point --yolo-weight / --rtdetr-model at a fine-tuned checkpoint to use.

class YoloDetector:
    """YOLOv8/YOLOv9 box proposer -- OPTIONAL, off unless --yolo-weight is set.

    WHEN: only if you pass --yolo-weight pointing at an ultralytics checkpoint.
    It then runs alongside Grounding DINO as an extra box proposer.

    WHY: it occupies the same "propose boxes" slot as Grounding DINO but is fast
    and easy to deploy -- IF you have a weight whose classes include an ID card.
    A stock COCO YOLO has no such class and would detect nothing useful, which
    is why it is disabled by default. Enable it once you fine-tune a YOLO on ID
    cards and want its speed/precision instead of (or on top of) the zero-shot
    detector.
    """

    def __init__(self, weight: str, device: str, conf: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(weight)
        self.device = device
        self.conf = conf

    def detect(self, img_bgr: np.ndarray):
        res = self.model.predict(img_bgr, conf=self.conf, device=self.device,
                                 verbose=False)[0]
        out = []
        for b in res.boxes:
            x0, y0, x1, y1 = b.xyxy[0].tolist()
            out.append(((x0, y0, x1, y1), float(b.conf[0])))
        return out


class RTDetrDetector:
    """RT-DETR box proposer -- OPTIONAL, off unless --rtdetr-model is set.

    WHEN: only if you pass --rtdetr-model (a HuggingFace RT-DETR/RT-DETRv2 id).
    Same extra-proposer slot as YoloDetector; you'd use one or the other.

    WHY: RT-DETR is a strong transformer detector, but like YOLO it needs a
    weight fine-tuned to include an ID-card class -- a COCO checkpoint won't
    detect one. Wired here (via transformers, no extra install) so you can drop
    in a fine-tuned RT-DETR later without touching the cascade logic.
    """

    def __init__(self, model_id: str, device: str, threshold: float = 0.3):
        import torch
        from transformers import AutoProcessor, AutoModelForObjectDetection
        self.torch = torch
        self.device = device
        self.threshold = threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(
            model_id).to(device).eval()

    def detect(self, img_bgr: np.ndarray):
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold,
            target_sizes=[(pil.height, pil.width)])[0]
        out = []
        for box, s in zip(res["boxes"].cpu().numpy(),
                          res["scores"].cpu().numpy()):
            out.append((tuple(map(float, box)), float(s)))
        return out


# ------------ cascade -------------- #

class Cascade:
    """Orchestrator that owns the model backends and runs them per page.

    WHEN: built once in main() from the CLI args, then reused for every page of
    every document.

    WHY: it decides -- from the flags -- which backends to load and hold in
    memory (loading is slow, so we do it once), then on each page it gathers
    candidate quads from the classical contour detector, the learned
    detector(s), and SAM 2, and hands them to the scorer. It is the glue that
    turns the individual classes above into one detect-and-crop pipeline.

    Backends held:
      self.gdino  Grounding DINO   -- primary learned detector (unless --no-ml)
      self.sam    Sam2Refiner      -- box -> exact oriented corners (unless --no-sam)
      self.extra  YOLO or RT-DETR  -- only if --yolo-weight / --rtdetr-model given
    """

    def __init__(self, args):
        self.args = args
        self.gdino = None
        self.sam = None
        self.extra = None
        if args.use_ml:
            device = self._pick_device(args.device)
            self.device = device
            print(f"[ml] loading Grounding DINO ({args.gdino_model}) "
                  f"on {device} ...", file=sys.stderr)
            self.gdino = GroundingDinoDetector(
                args.gdino_model, device, args.box_threshold)
            if not args.no_sam:
                print(f"[ml] loading SAM 2 ({args.sam_model}) on {device} ...",
                      file=sys.stderr)
                self.sam = Sam2Refiner(args.sam_model, device)
            if args.yolo_weight:
                self.extra = YoloDetector(args.yolo_weight, device)
            elif args.rtdetr_model:
                self.extra = RTDetrDetector(args.rtdetr_model, device)

    @staticmethod
    def _pick_device(pref: str) -> str:
        if pref != "auto":
            return pref
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def detect_page(self, img: np.ndarray):
        """Return (best_quad|None, best_score, candidates) for one page."""
        page_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        candidates = []  # (source, quad, score)

        quad_c, score_c = D.detect_page(img)
        if quad_c is not None:
            candidates.append(("contour", quad_c, float(score_c)))

        detectors = [d for d in (self.gdino, self.extra) if d is not None]
        for det in detectors:
            src = "gdino" if det is self.gdino else "learned"
            try:
                boxes = det.detect(img)
            except Exception as exc:
                print(f"    [warn] {src} detect failed: {exc}", file=sys.stderr)
                continue
            h, w = img.shape[:2]
            for box, det_conf in boxes:
                quad_b = box_to_quad(*pad_box(box, w, h, 0.02))
                candidates.append((f"{src}", quad_b,
                                   score_quad(img, page_gray, quad_b)))
                if self.sam is not None:
                    try:
                        quad_s = self.sam.refine(img, box)
                    except Exception as exc:
                        print(f"    [warn] sam2 refine failed: {exc}",
                              file=sys.stderr)
                        quad_s = None
                    if quad_s is not None:
                        candidates.append(("sam2", quad_s,
                                           score_quad(img, page_gray, quad_s)))

        if not candidates:
            return None, 0.0, []
        best = max(candidates, key=lambda c: c[2])
        best_quad = best[1] if best[2] >= D.MIN_CARD_SCORE else None
        return best_quad, best[2], candidates


# ---------------------------------------------------------------- debug draw

_SRC_COLOR = {
    "contour": (0, 200, 0),
    "gdino": (255, 120, 0),
    "learned": (255, 200, 0),
    "sam2": (200, 0, 200),
}


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


def write_debug_report(out_dir: Path, reports: list[dict]) -> Path:
    """Self-contained HTML QA report for the ML cascade: per document, the final
    crop plus every scanned page's candidate overlay (contour=green, gdino=blue,
    sam2=magenta, winner=yellow) with detection scores. Replaces the report that
    used to live in detection.py so the cascade owns its own reporting."""
    import html as _html
    from urllib.parse import quote as _quote

    def esc(v):
        return _html.escape(str(v), quote=True)

    def url(p):
        return _quote(str(p).replace("\\", "/"), safe="/._-()")

    ok = sum(1 for d in reports if d["status"] == "ok")
    pages_total = sum(len(d["pages"]) for d in reports)
    parts = ["""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ID detection debug report</title><style>
*{box-sizing:border-box}body{margin:0;font-family:Arial,Helvetica,sans-serif;
background:#f6f7f9;color:#17202a}header{padding:16px 20px;background:#fff;
border-bottom:1px solid #d7dde5}h1{margin:0 0 6px;font-size:20px}
.summary{color:#5d6b7a;font-size:14px;display:flex;gap:14px;flex-wrap:wrap}
section{padding:16px 20px;border-bottom:1px solid #e6eaf0}
.head{display:flex;justify-content:space-between;gap:12px;align-items:baseline}
h2{margin:0;font-size:16px;overflow-wrap:anywhere}
.badge{padding:3px 9px;border-radius:6px;font-weight:700;font-size:12px;
border:1px solid}.ok{color:#1f7a4d;background:#eaf6ef;border-color:#9fcfb6}
.no_card,.unreadable{color:#9b2d2d;background:#fbecec;border-color:#e1a8a8}
.meta{color:#5d6b7a;font-size:13px;margin:4px 0 10px}
.final{max-width:420px;border:1px solid #d7dde5;border-radius:8px;margin-bottom:10px}
.final img{display:block;width:100%;border-radius:8px}
.pages{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
gap:12px}.page{border:1px solid #d7dde5;border-radius:8px;overflow:hidden;
background:#fff}.page.best{border-color:#4faf7d;box-shadow:0 0 0 2px #4faf7d55}
.page img{display:block;width:100%}.pfoot{padding:6px 8px;font-size:12px;
display:flex;justify-content:space-between;color:#5d6b7a}
.score{font-weight:700;font-variant-numeric:tabular-nums;color:#17202a}</style>
</head><body><header><h1>ID detection debug report</h1><div class="summary">"""]
    parts.append(f"<span>{len(reports)} documents</span><span>{ok} detected</span>"
                 f"<span>{pages_total} pages</span>"
                 f"<span>threshold {D.MIN_CARD_SCORE:.2f}</span></div></header>")

    for d in reports:
        st = esc(d["status"])
        parts.append(f'<section><div class="head"><h2>{esc(d["document"])}</h2>'
                     f'<span class="badge {st}">{st}</span></div>')
        score = d.get("detection_score")
        parts.append(f'<div class="meta">{d["total_pages"]} page(s)'
                     + (f' | best page {d["best_page"]}' if d.get("best_page") else '')
                     + (f' | score {score:.3f}' if isinstance(score, float) else '')
                     + (f' | {esc(d["crop_type"])}' if d.get("crop_type") else '')
                     + '</div>')
        if d.get("output_image"):
            parts.append(f'<div class="final"><img src="{url(d["output_image"])}" '
                         f'alt="final crop"></div>')
        if d["pages"]:
            parts.append('<div class="pages">')
            for pg in d["pages"]:
                cls = "page best" if pg.get("is_best") else "page"
                img = (f'<img src="{url(pg["debug_image"])}" alt="page">'
                       if pg.get("debug_image") else '')
                parts.append(f'<div class="{cls}">{img}<div class="pfoot">'
                             f'<span>page {pg["page_no"]}</span>'
                             f'<span class="score">{float(pg["score"]):.3f}</span>'
                             f'</div></div>')
            parts.append('</div>')
        parts.append('</section>')
    parts.append('</body></html>')
    report_path = out_dir / "debug_report.html"
    report_path.write_text("".join(parts), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------- driver

def detect_document(cascade: Cascade, path: Path, out_dir: Path, pdf_dpi: int,
                    debug: bool):
    best = None  # (score, page_no, img, quad)
    n_pages = 0
    per_page = []
    page_records = []

    for page_no, img in D.load_pages(path, pdf_dpi=pdf_dpi):
        n_pages += 1
        quad, score, candidates = cascade.detect_page(img)
        per_page.append((page_no, score))
        debug_image = ""
        if debug:
            debug_path = save_page_debug(out_dir / "pages", path.stem,
                                         page_no, img, candidates, quad, score)
            debug_image = debug_path.relative_to(out_dir).as_posix()
        page_records.append({
            "page_no": page_no,
            "score": round(float(score), 5),
            "has_card": quad is not None,
            "is_best": False,
            "debug_image": debug_image,
        })
        if quad is not None and (best is None or score > best[0]):
            best = (score, page_no, img, quad)

    if per_page:
        scores = "  ".join(f"p{p}={s:.3f}" for p, s in per_page)
        print(f"  {path.name}: scanned {n_pages} page(s) -> {scores}",
              file=sys.stderr)

    if n_pages == 0:
        row = {"document": path.name, "status": "unreadable", "card_page": "",
               "total_pages": 0, "detection_score": "", "crop_type": "",
               "output_image": ""}
        report = {"document": path.name, "status": "unreadable",
                  "best_page": None, "total_pages": 0, "detection_score": None,
                  "crop_type": "", "output_image": "", "pages": page_records}
        return row, report
    if best is None:
        print(f"  [FAIL] ID or passport is missing: {path.name}",
              file=sys.stderr)
        row = {"document": path.name, "status": "no_card", "card_page": "",
               "total_pages": n_pages, "detection_score": "", "crop_type": "",
               "output_image": ""}
        report = {"document": path.name, "status": "no_card", "best_page": None,
                  "total_pages": n_pages, "detection_score": None,
                  "crop_type": "", "output_image": "", "pages": page_records}
        return row, report

    score, page_no, img, quad = best
    for page in page_records:
        page["is_best"] = page["page_no"] == page_no

    page_path = out_dir / f"{path.stem}.png"
    final_img, crop_type = D.build_final_card_image(img, quad)
    # deskew: snap the winning region to the card's true oriented corners so
    # the final warp has no tilt-gaps. General across documents, but skip open
    # passports -- their data-page split must run on the original spread, not a
    # deskewed full-spread rectangle.
    if final_img is not None and crop_type != "passport_details":
        tq = tighten_to_card(img, quad, sam=cascade.sam)
        if tq is not quad:
            final_t, type_t = D.build_final_card_image(img, tq)
            if final_t is not None and type_t != "passport_details":
                final_img, crop_type = final_t, type_t
    if final_img is None:
        final_img, crop_type = img, "page_fallback"
    cv2.imwrite(str(page_path), final_img)
    output_image = page_path.relative_to(out_dir).as_posix()

    tag = "B&W" if D.is_grayscale(img) else "color"
    print(f"  [ok]   {path.name}: card on page {page_no}/{n_pages} "
          f"({tag}, {crop_type})")
    detection_score = round(float(score), 5)
    row = {"document": path.name, "status": "ok", "card_page": page_no,
           "total_pages": n_pages, "detection_score": detection_score,
           "crop_type": crop_type, "output_image": output_image}
    report = {"document": path.name, "status": "ok", "best_page": page_no,
              "total_pages": n_pages, "detection_score": detection_score,
              "crop_type": crop_type, "output_image": output_image,
              "pages": page_records}
    return row, report


def write_summary_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    """Write the summary CSV, falling back to a timestamped name if the target
    is locked (commonly: the file is open in Excel). A locked summary.csv should
    never throw away a whole detection run with a PermissionError."""
    try:
        handle = path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        import time
        alt = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        print(f"  [warn] {path.name} is locked (open elsewhere?); "
              f"writing {alt.name} instead", file=sys.stderr)
        handle = alt.open("w", newline="", encoding="utf-8")
        path = alt
    with handle as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Stage 1 ML cascade: contour + Grounding DINO + SAM 2.")
    # Which flag drives which class (see Cascade.__init__):
    #   --no-ml        -> load NO learned models; classical contour only.
    #   --gdino-model  -> GroundingDinoDetector (the always-on learned detector).
    #   --box-threshold-> GroundingDinoDetector confidence cutoff for a box.
    #   --no-sam       -> skip Sam2Refiner; keep the raw detector boxes (no
    #                     deskew, corners not snapped to the card).
    #   --sam-model    -> which SAM 2 checkpoint Sam2Refiner loads.
    #   --yolo-weight  -> load YoloDetector (extra proposer; needs an ID weight).
    #   --rtdetr-model -> load RTDetrDetector instead (same extra slot).
    #   --device       -> where all of the above run (auto picks cuda if present).
    ap.add_argument("input_dir")
    ap.add_argument("-o", "--output", default="stage1_ml")
    ap.add_argument("--pdf-dpi", type=int, default=300)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-ml", dest="use_ml", action="store_false",
                    help="classical contour detection only")
    ap.add_argument("--no-sam", action="store_true",
                    help="skip SAM 2 mask refinement (boxes only)")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--sam-model", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--box-threshold", type=float, default=0.25)
    ap.add_argument("--yolo-weight", default="",
                    help="ultralytics weight fine-tuned for ID cards")
    ap.add_argument("--rtdetr-model", default="",
                    help="RT-DETR HF id fine-tuned for ID cards")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = sorted(p for p in in_dir.iterdir()
                  if p.suffix.lower() in D.ALL_EXTS)
    if not docs:
        sys.exit(f"No supported files found in {in_dir}")

    cascade = Cascade(args)

    rows, reports = [], []
    for p in docs:
        row, report = detect_document(cascade, p, out_dir, args.pdf_dpi,
                                      args.debug)
        rows.append(row)
        reports.append(report)

    summary_path = write_summary_csv(
        out_dir / "summary.csv",
        ["document", "status", "card_page", "total_pages",
         "detection_score", "crop_type", "output_image"], rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nDetected cards in {ok}/{len(docs)} documents -> {out_dir}/")
    print(f"Summary table -> {summary_path}")
    if args.debug:
        report_path = write_debug_report(out_dir, reports)
        print(f"Interactive debug report -> {report_path}")


if __name__ == "__main__":
    main()
