"""Enhance a folder of normalized ID/passport crops and score the results.

Default path, for normal/good crops:
  NAFNet denoise -> Swin2SR x4

Optional document-reconstruction path, for damaged halftone/copier scans only:
  smooth card surface -> extract cleaned ink/text mask -> x4 crisp mask overlay

The script automatically routes damaged halftone/copier scans to reconstruction
and keeps already-good crops on the normal enhancement path.

Examples:
  python enhance_folder.py stage1_interactive -o enhanced_images
  python enhance_folder.py stage1_interactive -o enhanced_images --reconstruct-mode auto
  python enhance_folder.py stage1_interactive -o enhanced_images --no-denoise --sr-backend lanczos
"""

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
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAF = HERE / "NAFNet"
if not NAF.exists():
    NAF = HERE / "NAFnet"
EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff")


def iter_images(in_dir: Path):
    return sorted(p for ext in EXTS for p in in_dir.glob(ext))


def copy_images(paths: list[Path], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, out_dir / path.name)
    return out_dir


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def band_score(value: float, low: float, good_low: float,
               good_high: float, high: float) -> float:
    value = float(value)
    if value <= low or value >= high:
        return 0.0
    if good_low <= value <= good_high:
        return 1.0
    if value < good_low:
        return clamp01((value - low) / max(good_low - low, 1e-6))
    return clamp01((high - value) / max(high - good_high, 1e-6))


def denoise_stage(in_dir: Path, tmp_dir: Path, opt_path: Path):
    sys.path.insert(0, str(NAF))
    cwd = os.getcwd()
    os.chdir(NAF)  # yml pretrain path is relative to NAFNet/
    sys.argv = ["demo.py", "-opt", str(opt_path),
                "--input_path", "x", "--output_path", "y"]
    import torch
    from basicsr.models import create_model
    from basicsr.train import parse_options
    from basicsr.utils import (FileClient, imfrombytes, img2tensor,
                               tensor2img, imwrite)

    opt = parse_options(is_train=False)
    opt["num_gpu"] = torch.cuda.device_count()
    opt["dist"] = False
    model = create_model(opt)
    fc = FileClient("disk")
    imgs = [str(p) for p in iter_images(in_dir)]
    print(f"[denoise] {len(imgs)} image(s)", flush=True)
    for p in imgs:
        img = imfrombytes(fc.get(p, None), float32=True)
        img = img2tensor(img, bgr2rgb=True, float32=True)
        model.feed_data(data={"lq": img.unsqueeze(0)})
        if model.opt["val"].get("grids", False):
            model.grids()
        model.test()
        if model.opt["val"].get("grids", False):
            model.grids_inverse()
        out = tensor2img([model.get_current_visuals()["result"]])
        imwrite(out, str(tmp_dir / Path(p).name))
        print("  denoised", Path(p).name, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    os.chdir(cwd)


def descreen_image(bgr, strength: float):
    """Light smoothing for normal SR-only use. Not enabled by default."""
    import cv2
    import numpy as np

    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return bgr

    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (max(1, w // 2), max(1, h // 2)),
                       interpolation=cv2.INTER_AREA)
    smooth = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    smooth = cv2.fastNlMeansDenoisingColored(smooth, None, 7, 6, 7, 21)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 55, 145)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    keep = (edges.astype(np.float32) / 255.0)[..., None]
    alpha = strength * (1.0 - 0.75 * keep)
    out = bgr.astype(np.float32) * (1.0 - alpha)
    out += smooth.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def descreen_stage(in_dir: Path, out_dir: Path, strength: float):
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = iter_images(in_dir)
    print(f"[descreen] {len(imgs)} image(s), strength={strength:.2f}",
          flush=True)
    for p in imgs:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  skipped unreadable {p.name}", flush=True)
            continue
        cv2.imwrite(str(out_dir / p.name), descreen_image(img, strength))
        print("  descreened", p.name, flush=True)


def clean_component_mask(mask, min_area: int = 12,
                         max_area_frac: float = 0.12):
    """Drop isolated copier dots but keep character/border-sized components."""
    import cv2
    import numpy as np

    mask = (mask > 0).astype(np.uint8) * 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    h, w = mask.shape[:2]
    max_area = int(max(1, h * w * max_area_frac))

    for label in range(1, n_labels):
        x, y, bw, bh, area = stats[label]
        long_side = max(bw, bh)
        short_side = min(bw, bh)
        if area < min_area or area > max_area:
            continue
        if short_side <= 2 and area < min_area * 3:
            continue
        if long_side >= 5:
            out[labels == label] = 255
    return out


def ink_stroke_mask(gray, preserve_more: bool):
    """Find ink strokes while rejecting most single-pixel halftone texture."""
    import cv2
    import numpy as np

    h, w = gray.shape[:2]
    small_side = max(1, min(h, w))
    bg_kernel = max(17, min(51, (small_side // 18) | 1))
    bg = cv2.medianBlur(gray, bg_kernel)
    blackhat = cv2.subtract(bg, gray)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)
    percentile = 88 if preserve_more else 91
    cutoff = max(10 if preserve_more else 12,
                 int(np.percentile(blackhat, percentile)))
    strong_dark = (blackhat >= cutoff).astype(np.uint8) * 255

    block = max(21, min(71, ((small_side // 10) | 1)))
    if block % 2 == 0:
        block += 1
    adaptive = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 10)

    mask = cv2.bitwise_or(strong_dark, cv2.bitwise_and(adaptive, strong_dark))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((2, 2), np.uint8))
    min_area = 14 if preserve_more else 22
    mask = clean_component_mask(mask, min_area=min_area, max_area_frac=0.10)

    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 90, 210)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges = cv2.bitwise_and(edges, mask)
    edges = clean_component_mask(edges, min_area=min_area + 6,
                                 max_area_frac=0.08)

    mask = cv2.bitwise_or(mask, edges)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((2, 2), np.uint8))


def smooth_card_surface(bgr, strength: float):
    """Create a low-texture surface while preserving broad card lighting."""
    import cv2

    strength = max(0.0, min(1.0, float(strength)))
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (max(1, w // 4), max(1, h // 4)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.fastNlMeansDenoisingColored(small, None, 9, 7, 7, 21)
    surface = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    surface = cv2.GaussianBlur(surface, (0, 0), 0.8)
    return cv2.addWeighted(bgr, 1.0 - strength, surface, strength, 0)


def reconstruct_lanczos_stage(in_dir: Path, out_dir: Path, scale: int,
                              surface_strength: float, ink_strength: float,
                              preserve_more_text: bool):
    import cv2
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = iter_images(in_dir)
    print(f"[document-reconstruct x{scale}] {len(imgs)} image(s), "
          f"surface={surface_strength:.2f}, ink={ink_strength:.2f}",
          flush=True)
    for p in imgs:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  skipped unreadable {p.name}", flush=True)
            continue

        h, w = bgr.shape[:2]
        surface = smooth_card_surface(bgr, surface_strength)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = ink_stroke_mask(gray, preserve_more_text)

        out_size = (w * scale, h * scale)
        surface_up = cv2.resize(surface, out_size,
                                interpolation=cv2.INTER_LANCZOS4)
        mask_up = cv2.resize(mask, out_size, interpolation=cv2.INTER_NEAREST)
        mask_up = cv2.morphologyEx(mask_up, cv2.MORPH_CLOSE,
                                   np.ones((3, 3), np.uint8))

        alpha = (mask_up.astype(np.float32) / 255.0)[..., None]
        alpha *= max(0.0, min(1.0, ink_strength))
        ink = surface_up.astype(np.float32) * (1.0 - 0.58 * ink_strength)
        out = surface_up.astype(np.float32) * (1.0 - alpha) + ink * alpha

        cv2.imwrite(str(out_dir / p.name), np.clip(out, 0, 255).astype(np.uint8))
        print(f"  {p.name} {w}x{h}->{w*scale}x{h*scale}", flush=True)


def lanczos_stage(in_dir: Path, out_dir: Path, scale: int):
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = iter_images(in_dir)
    print(f"[resize x{scale}] {len(imgs)} image(s) with Lanczos", flush=True)
    for p in imgs:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  skipped unreadable {p.name}", flush=True)
            continue
        h, w = img.shape[:2]
        out = cv2.resize(img, (w * scale, h * scale),
                         interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(out_dir / p.name), out)
        print(f"  {p.name} {w}x{h}->{w*scale}x{h*scale}", flush=True)


def sr_stage(in_dir: Path, out_dir: Path, model_id: str):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import (AutoImageProcessor,
                              Swin2SRForImageSuperResolution)

    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = Swin2SRForImageSuperResolution.from_pretrained(model_id).to(dev).eval()

    def run(pil, device, half):
        inp = proc(images=pil, return_tensors="pt")
        px = inp["pixel_values"].to(device)
        m = model.to(device)
        if half:
            px = px.half()
            m = m.half()
        with torch.inference_mode():
            o = m(pixel_values=px)
        rec = o.reconstruction.data.squeeze().float().cpu().clamp(0, 1).numpy()
        return np.moveaxis(rec, 0, -1)

    imgs = [str(p) for p in iter_images(in_dir)]
    print(f"[super-res x4] {len(imgs)} image(s) on {dev}", flush=True)
    for p in imgs:
        pil = Image.open(p).convert("RGB")
        width, height = pil.size
        start = time.time()
        try:
            arr = run(pil, dev, True)
            mode = f"{dev}/fp16"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            arr = run(pil, "cpu", False)
            mode = "cpu/fp32"
        arr = arr[:height * 4, :width * 4]
        Image.fromarray((arr * 255.0).round().astype(np.uint8)).save(
            str(out_dir / Path(p).name))
        print(f"  {Path(p).name} {width}x{height}->{width*4}x{height*4} "
              f"{time.time()-start:.1f}s {mode}", flush=True)


def forced_reconstruct(name: str, patterns: list[str],
                       reconstruct_all: bool) -> bool:
    return reconstruct_all or any(fnmatch.fnmatch(name, pattern)
                                  for pattern in patterns)


def read_image_for_score(path: Path):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def normalize_for_metrics(img, max_side: int = 1200):
    import cv2

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def routing_statistics(img) -> dict[str, float]:
    import cv2
    import numpy as np

    img = normalize_for_metrics(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float((edges > 0).mean())
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    high_freq = cv2.absdiff(gray, blur)
    granularity = float(high_freq.std())

    block = max(21, min(61, ((min(gray.shape[:2]) // 10) | 1)))
    if block % 2 == 0:
        block += 1
    text = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, 10)
    text = clean_component_mask(text, min_area=8, max_area_frac=0.10)
    text_density = float((text > 0).mean())

    return {
        "laplacian_var": round(lap_var, 3),
        "contrast_std": round(contrast, 3),
        "edge_density": round(edge_density, 5),
        "text_density": round(text_density, 5),
        "granularity": round(granularity, 3),
    }


def metric_value(value, digits: int = 5):
    if value is None or value == "":
        return ""
    value = float(value)
    if math.isinf(value):
        return "inf"
    return round(value, digits)


def calculate_psnr(img1, img2):
    import numpy as np

    diff = img1.astype(np.float64) - img2.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _ssim_channel(img1, img2):
    import cv2
    import numpy as np

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    if min(img1.shape[:2]) < 11:
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1 = img1.var()
        sigma2 = img2.var()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    else:
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = kernel @ kernel.T
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1 = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2 = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2)
        return float(np.mean(numerator / denominator))

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2)
    return float(numerator / denominator)


def calculate_ssim(img1, img2):
    import numpy as np

    if img1.ndim == 2:
        return _ssim_channel(img1, img2)
    return float(np.mean([
        _ssim_channel(img1[..., idx], img2[..., idx])
        for idx in range(img1.shape[2])
    ]))


_NIQE_ERROR = None


def calculate_niqe_optional(img):
    """Use the bundled BasicSR/NAFNet NIQE implementation when available."""
    global _NIQE_ERROR
    if _NIQE_ERROR:
        return "", f"NIQE unavailable: {_NIQE_ERROR}"

    cwd = os.getcwd()
    added_path = False
    try:
        if str(NAF) not in sys.path:
            sys.path.insert(0, str(NAF))
            added_path = True
        os.chdir(NAF)
        from basicsr.metrics.niqe import calculate_niqe

        value = calculate_niqe(img, crop_border=0, input_order="HWC",
                               convert_to="y")
        return metric_value(value), ""
    except Exception as exc:
        _NIQE_ERROR = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        return "", f"NIQE unavailable: {_NIQE_ERROR}"
    finally:
        os.chdir(cwd)
        if added_path:
            try:
                sys.path.remove(str(NAF))
            except ValueError:
                pass


_LPIPS_MODEL = None
_LPIPS_ERROR = None


def calculate_lpips_optional(img1, img2):
    """Use the standard LPIPS perceptual metric if the package is installed."""
    global _LPIPS_MODEL, _LPIPS_ERROR
    if _LPIPS_ERROR:
        return "", f"LPIPS unavailable: {_LPIPS_ERROR}"
    try:
        import cv2
        import numpy as np
        import torch
        import lpips

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if _LPIPS_MODEL is None:
            _LPIPS_MODEL = lpips.LPIPS(net="alex").eval()
        _LPIPS_MODEL = _LPIPS_MODEL.to(device)

        def to_tensor(bgr):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            arr = rgb.astype(np.float32) / 127.5 - 1.0
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.inference_mode():
            value = _LPIPS_MODEL(to_tensor(img1), to_tensor(img2))
        return metric_value(float(value.detach().cpu().item())), ""
    except Exception as exc:
        _LPIPS_ERROR = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        return "", f"LPIPS unavailable: {_LPIPS_ERROR}"


def reconstruction_need(path: Path, threshold: float) -> dict:
    """Decide from image statistics whether reconstruction is needed.

    This avoids filename-specific routing. The risk value is only an internal
    routing signal for damaged copier/halftone texture; it is not an image
    quality or generative-model evaluation metric.
    """
    img = read_image_for_score(path)
    if img is None:
        return {"decision": False, "risk": 0.0, "reason": "unreadable"}

    metrics = routing_statistics(img)
    granularity = float(metrics["granularity"])
    edge_density = float(metrics["edge_density"])
    lap_var = float(metrics["laplacian_var"])
    text_density = float(metrics["text_density"])
    contrast = float(metrics["contrast_std"])

    granularity_risk = clamp01((granularity - 13.0) / 10.0)
    excessive_edges = clamp01((edge_density - 0.28) / 0.12)
    text_clutter_risk = clamp01((text_density - 0.24) / 0.12)
    low_contrast_risk = clamp01((18.0 - contrast) / 12.0)
    lap_risk = clamp01(
        (math.log1p(lap_var) - math.log1p(2500.0)) /
        max(math.log1p(10000.0) - math.log1p(2500.0), 1e-6)
    )
    risk = (0.42 * granularity_risk + 0.28 * excessive_edges +
            0.15 * text_clutter_risk + 0.10 * lap_risk +
            0.05 * low_contrast_risk)
    decision = (
        risk >= threshold and
        granularity_risk >= 0.25 and
        (excessive_edges >= 0.20 or text_clutter_risk >= 0.25 or
         lap_risk >= 0.25)
    )
    reason = (
        f"auto_reconstruction_risk={risk:.3f}; "
        f"granularity={granularity:.3f}; "
        f"edge_density={edge_density:.5f}; "
        f"text_density={text_density:.5f}; "
        f"laplacian_var={lap_var:.3f}; "
        f"contrast_std={contrast:.3f}"
    )
    return {
        "decision": bool(decision),
        "risk": round(risk, 5),
        "reason": reason,
        "granularity": round(granularity, 3),
        "edge_density": round(edge_density, 5),
        "text_density": round(text_density, 5),
        "laplacian_var": round(lap_var, 3),
        "contrast_std": round(contrast, 3),
    }


def score_pair(src_path: Path, out_path: Path, mode: str, pipeline: str,
               reference_path: Path | None = None) -> dict:
    import cv2

    src = read_image_for_score(src_path)
    out = read_image_for_score(out_path)
    if src is None or out is None:
        return {
            "image": src_path.name,
            "mode": mode,
            "pipeline": pipeline,
            "status": "missing_or_unreadable",
        }

    src_h, src_w = src.shape[:2]
    metric_notes = []
    reference_type = "input_crop"
    reference_image = src_path.name
    reference = src
    if reference_path is not None:
        reference_type = "reference_dir"
        reference_image = reference_path.name
        reference = read_image_for_score(reference_path)
        if reference is None:
            reference_type = "reference_missing_using_input_crop"
            reference_image = src_path.name
            reference = src
            metric_notes.append(f"missing reference image: {reference_path.name}")

    ref_h, ref_w = reference.shape[:2]
    out_h, out_w = out.shape[:2]
    comparable = out
    if (out_w, out_h) != (ref_w, ref_h):
        comparable = cv2.resize(out, (ref_w, ref_h),
                                interpolation=cv2.INTER_AREA)

    psnr = calculate_psnr(reference, comparable)
    ssim = calculate_ssim(reference, comparable)
    lpips_value, lpips_note = calculate_lpips_optional(reference, comparable)
    reference_niqe, reference_niqe_note = calculate_niqe_optional(reference)
    output_niqe, output_niqe_note = calculate_niqe_optional(out)
    niqe_delta = ""
    if reference_niqe != "" and output_niqe != "":
        niqe_delta = metric_value(float(output_niqe) - float(reference_niqe))
    for note in (lpips_note, reference_niqe_note, output_niqe_note):
        if note and note not in metric_notes:
            metric_notes.append(note)

    return {
        "image": src_path.name,
        "mode": mode,
        "pipeline": pipeline,
        "status": "ok",
        "input_width": src_w,
        "input_height": src_h,
        "output_width": out_w,
        "output_height": out_h,
        "reference_image": reference_image,
        "reference_type": reference_type,
        "reference_width": ref_w,
        "reference_height": ref_h,
        "reference_psnr": metric_value(psnr),
        "reference_ssim": metric_value(ssim),
        "reference_lpips": lpips_value,
        "reference_niqe": reference_niqe,
        "output_niqe": output_niqe,
        "niqe_delta": niqe_delta,
        "metric_reference": "output resized to reference for PSNR/SSIM/LPIPS",
        "metric_notes": "; ".join(metric_notes),
        "output_image": out_path.name,
    }


def read_detection_summary(input_dir: Path) -> dict[str, dict]:
    summary_path = input_dir / "summary.csv"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        row.get("output_image", ""): row
        for row in rows
        if row.get("output_image")
    }


def find_reference_image(reference_dir: Path | None, src_path: Path) -> Path | None:
    if reference_dir is None:
        return None
    direct = reference_dir / src_path.name
    if direct.exists():
        return direct
    for candidate in iter_images(reference_dir):
        if candidate.stem == src_path.stem:
            return candidate
    return reference_dir / src_path.name


def metric_pairs(input_dir: Path, out_dir: Path,
                 reference_dir: Path | None) -> tuple[list[tuple[Path, Path]], int]:
    pairs = []
    missing = 0
    for src_path in iter_images(input_dir):
        out_path = out_dir / src_path.name
        if not out_path.exists():
            continue
        ref_path = find_reference_image(reference_dir, src_path)
        if ref_path is None:
            ref_path = src_path
        elif not ref_path.exists():
            missing += 1
            continue
        pairs.append((ref_path, out_path))
    return pairs, missing


def unavailable_distribution_rows(note: str, reference_type: str,
                                  pair_count: int) -> list[dict]:
    metrics = ("fid", "kid_mean", "kid_std")
    return [
        {
            "metric": metric,
            "value": "",
            "status": "unavailable",
            "reference_type": reference_type,
            "num_image_pairs": pair_count,
            "note": note,
        }
        for metric in metrics
    ]


def calculate_distribution_metrics(input_dir: Path, out_dir: Path,
                                   reference_dir: Path | None) -> list[dict]:
    """Calculate standard set-level generative metrics when deps exist."""
    pairs, missing = metric_pairs(input_dir, out_dir, reference_dir)
    reference_type = "reference_dir" if reference_dir is not None else "input_crop"
    base_note = (
        "FID/KID are distribution metrics; lower is better. "
        "Use a held-out real or ground-truth reference set for proper GAN "
        "evaluation."
    )
    if reference_dir is None:
        base_note += " No --reference-dir was supplied, so input crops are the reference distribution."
    if missing:
        base_note += f" Skipped {missing} output(s) with no matching reference image."
    if len(pairs) < 2:
        return unavailable_distribution_rows(
            base_note + " Need at least two image pairs.", reference_type,
            len(pairs))

    try:
        import cv2
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def image_tensor(path: Path):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"unreadable image: {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            return tensor.to(device=device, dtype=torch.uint8)

        fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        kid = KernelInceptionDistance(
            subset_size=max(2, min(50, len(pairs))),
            normalize=False,
        ).to(device)

        for ref_path, out_path in pairs:
            fid.update(image_tensor(ref_path), real=True)
            fid.update(image_tensor(out_path), real=False)
            kid.update(image_tensor(ref_path), real=True)
            kid.update(image_tensor(out_path), real=False)

        fid_value = metric_value(float(fid.compute().detach().cpu().item()))
        kid_mean, kid_std = kid.compute()
        kid_mean = metric_value(float(kid_mean.detach().cpu().item()))
        kid_std = metric_value(float(kid_std.detach().cpu().item()))
        return [
            {
                "metric": "fid",
                "value": fid_value,
                "status": "ok",
                "reference_type": reference_type,
                "num_image_pairs": len(pairs),
                "note": base_note,
            },
            {
                "metric": "kid_mean",
                "value": kid_mean,
                "status": "ok",
                "reference_type": reference_type,
                "num_image_pairs": len(pairs),
                "note": base_note,
            },
            {
                "metric": "kid_std",
                "value": kid_std,
                "status": "ok",
                "reference_type": reference_type,
                "num_image_pairs": len(pairs),
                "note": base_note,
            },
        ]
    except Exception as exc:
        note = (
            f"{base_note} FID/KID unavailable: "
            f"{type(exc).__name__}: {str(exc).splitlines()[0]}. "
            "Install torchmetrics with its image dependencies to compute them."
        )
        return unavailable_distribution_rows(note, reference_type, len(pairs))


def write_distribution_metrics(out_dir: Path, rows: list[dict]) -> Path:
    path = out_dir / "generative_metrics.csv"
    fieldnames = [
        "metric", "value", "status", "reference_type", "num_image_pairs",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_score_outputs(input_dir: Path, out_dir: Path,
                        mode_by_name: dict[str, str],
                        pipeline_by_name: dict[str, str],
                        decision_by_name: dict[str, dict],
                        reference_dir: Path | None = None):
    rows = []
    detection_by_output = read_detection_summary(input_dir)
    for src_path in iter_images(input_dir):
        out_path = out_dir / src_path.name
        mode = mode_by_name.get(src_path.name, "normal")
        pipeline = pipeline_by_name.get(src_path.name, "")
        reference_path = find_reference_image(reference_dir, src_path)
        row = score_pair(src_path, out_path, mode, pipeline, reference_path)
        decision = decision_by_name.get(src_path.name, {})
        row["reconstruction_decision"] = decision.get("decision", "")
        row["auto_reconstruction_risk"] = decision.get("risk", "")
        row["reconstruction_reason"] = decision.get("reason", "")
        det = detection_by_output.get(src_path.name, {})
        row["source_document"] = det.get("document", "")
        row["detection_status"] = det.get("status", "")
        row["detection_score"] = det.get("detection_score", "")
        row["card_page"] = det.get("card_page", "")
        row["crop_type"] = det.get("crop_type", "")
        rows.append(row)

    csv_path = out_dir / "enhancement_scores.csv"
    fieldnames = [
        "image", "source_document", "mode", "pipeline", "status",
        "reconstruction_decision", "auto_reconstruction_risk",
        "reconstruction_reason",
        "detection_status", "detection_score", "card_page", "crop_type",
        "input_width", "input_height", "output_width", "output_height",
        "reference_image", "reference_type", "reference_width",
        "reference_height", "reference_psnr", "reference_ssim",
        "reference_lpips", "reference_niqe", "output_niqe", "niqe_delta",
        "metric_reference", "metric_notes", "output_image",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    distribution_rows = calculate_distribution_metrics(
        input_dir, out_dir, reference_dir)
    generative_path = write_distribution_metrics(out_dir, distribution_rows)
    report_path = out_dir / "enhancement_report.html"
    write_score_html(report_path, rows, distribution_rows)
    print(f"Enhancement scores -> {csv_path}")
    print(f"Generative metrics -> {generative_path}")
    print(f"Enhancement report -> {report_path}")


def write_score_html(report_path: Path, rows: list[dict],
                     distribution_rows: list[dict]):
    def cell(value):
        return html.escape("" if value is None else str(value))

    table_rows = []
    for row in rows:
        image = cell(row.get("image", ""))
        output = cell(row.get("output_image", row.get("image", "")))
        table_rows.append(
            "<tr>"
            f"<td>{image}</td>"
            f"<td>{cell(row.get('detection_score', ''))}</td>"
            f"<td>{cell(row.get('mode', ''))}</td>"
            f"<td>{cell(row.get('auto_reconstruction_risk', ''))}</td>"
            f"<td>{cell(row.get('pipeline', ''))}</td>"
            f"<td>{cell(row.get('reference_type', ''))}</td>"
            f"<td>{cell(row.get('reference_psnr', ''))}</td>"
            f"<td>{cell(row.get('reference_ssim', ''))}</td>"
            f"<td>{cell(row.get('reference_lpips', ''))}</td>"
            f"<td>{cell(row.get('output_niqe', ''))}</td>"
            f"<td>{cell(row.get('metric_notes', ''))}</td>"
            f"<td><a href=\"{output}\"><img src=\"{output}\" alt=\"{image}\"></a></td>"
            "</tr>"
        )

    distribution_table_rows = []
    for row in distribution_rows:
        distribution_table_rows.append(
            "<tr>"
            f"<td>{cell(row.get('metric', ''))}</td>"
            f"<td>{cell(row.get('value', ''))}</td>"
            f"<td>{cell(row.get('status', ''))}</td>"
            f"<td>{cell(row.get('reference_type', ''))}</td>"
            f"<td>{cell(row.get('num_image_pairs', ''))}</td>"
            f"<td>{cell(row.get('note', ''))}</td>"
            "</tr>"
        )

    report_path.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Enhancement Metrics</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f8; color: #161616; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #222; color: white; position: sticky; top: 0; }}
    img {{ width: 260px; max-height: 180px; object-fit: contain; background: #eee; }}
    .note {{ max-width: 980px; line-height: 1.45; }}
    code {{ background: #eee; padding: 2px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>Enhancement Metrics</h1>
  <p class="note">
    Per-image metrics use standard restoration/IQA names: PSNR and SSIM are
    full-reference fidelity metrics, LPIPS is perceptual distance when installed,
    and NIQE is a no-reference image quality metric when available. If no
    <code>--reference-dir</code> is supplied, the input crop is the reference,
    so PSNR/SSIM/LPIPS measure fidelity to the input, not ground-truth
    reconstruction quality. The auto reconstruction risk is only a routing
    debug value.
  </p>
  <h2>Set-Level Generative Metrics</h2>
  <table>
    <thead>
      <tr>
        <th>Metric</th><th>Value</th><th>Status</th><th>Reference</th><th>Pairs</th><th>Note</th>
      </tr>
    </thead>
    <tbody>
      {''.join(distribution_table_rows)}
    </tbody>
  </table>
  <h2>Per-Image Metrics</h2>
  <table>
    <thead>
      <tr>
        <th>Image</th><th>Detection</th><th>Mode</th><th>Auto Risk</th><th>Pipeline</th><th>Reference</th>
        <th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>Output NIQE</th><th>Notes</th><th>Output</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>
""", encoding="utf-8")


def pipeline_name(args, mode: str) -> str:
    denoise = "NAFNet -> " if not args.no_denoise else ""
    if mode in {"auto_reconstruct", "document_reconstruct"}:
        return denoise + "document-reconstruct -> Lanczos"
    if args.descreen:
        return denoise + f"descreen({args.descreen_strength:.2f}) -> {args.sr_backend}"
    return denoise + args.sr_backend


def main():
    ap = argparse.ArgumentParser(
        description="Enhance ID crops with automatic damaged-document reconstruction")
    ap.add_argument("input_dir")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--sr-model",
                    default="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr")
    ap.add_argument("--opt",
                    default=str(NAF / "options/test/SIDD/NAFNet-width64.yml"))
    ap.add_argument("--no-denoise", action="store_true",
                    help="skip NAFNet denoise")
    ap.add_argument("--sr-backend", choices=("swin2sr", "lanczos"),
                    default="swin2sr",
                    help="normal-image upscaler")
    ap.add_argument("--scale", type=int, default=4,
                    help="output scale for Lanczos and reconstruction modes")
    ap.add_argument("--descreen", action="store_true",
                    help="light descreen before normal SR; off by default")
    ap.add_argument("--descreen-strength", type=float, default=0.35,
                    help="0..1 light descreen strength")
    ap.add_argument("--reconstruct-mode", choices=("auto", "off", "all"),
                    default="auto",
                    help="auto routes damaged crops to reconstruction")
    ap.add_argument("--auto-reconstruct-threshold", type=float, default=0.40,
                    help="higher is more conservative for auto reconstruction")
    ap.add_argument("--document-reconstruct", action="store_true",
                    help="legacy alias for --reconstruct-mode all")
    ap.add_argument("--document-pattern", action="append", default=[],
                    help="manual override glob for forcing reconstruction")
    ap.add_argument("--reconstruct-strength", type=float, default=0.72,
                    help="0..1 smoothing strength for document reconstruction")
    ap.add_argument("--ink-strength", type=float, default=0.95,
                    help="0..1 darkness of restored text strokes")
    ap.add_argument("--preserve-more-text", action="store_true",
                    help="keep more tiny text strokes, with more residual dots")
    ap.add_argument("--reference-dir", default="",
                    help="optional clean/ground-truth reference folder with matching filenames")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output).resolve()
    opt_path = Path(args.opt).resolve()
    reference_dir = Path(args.reference_dir).resolve() if args.reference_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_by_name: dict[str, str] = {}
    pipeline_by_name: dict[str, str] = {}
    decision_by_name: dict[str, dict] = {}
    reconstruct_mode = "all" if args.document_reconstruct else args.reconstruct_mode

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        current = in_dir
        if not args.no_denoise:
            current = tmp_root / "denoised"
            current.mkdir(parents=True, exist_ok=True)
            denoise_stage(in_dir, current, opt_path)

        all_images = iter_images(current)
        print(f"[reconstruct] {len(all_images)} input image(s) from {in_dir}",
              flush=True)
        if not all_images:
            print(f"[warn] no images found in {in_dir} -- NO reconstructed "
                  f"images will be written (only metrics on any existing "
                  f"outputs run). Check the input folder path/contents.",
                  file=sys.stderr)
        doc_images = []
        for image in all_images:
            forced = forced_reconstruct(image.name, args.document_pattern,
                                        reconstruct_mode == "all")
            decision = {
                "decision": False,
                "risk": "",
                "reason": "manual off",
            }
            mode = "normal"
            if forced:
                decision = {
                    "decision": True,
                    "risk": 1.0,
                    "reason": "manual force",
                }
                mode = "document_reconstruct"
            elif reconstruct_mode == "auto":
                decision = reconstruction_need(
                    image, args.auto_reconstruct_threshold)
                if decision["decision"]:
                    mode = "auto_reconstruct"
            if decision["decision"]:
                doc_images.append(image)
            mode_by_name[image.name] = mode
            decision_by_name[image.name] = decision

        doc_names = {p.name for p in doc_images}
        normal_images = [p for p in all_images if p.name not in doc_names]

        for image in normal_images:
            pipeline_by_name[image.name] = pipeline_name(args, "normal")
        for image in doc_images:
            pipeline_by_name[image.name] = pipeline_name(
                args, mode_by_name[image.name])

        if normal_images:
            normal_dir = copy_images(normal_images, tmp_root / "normal")
            if args.descreen:
                descreened = tmp_root / "normal_descreened"
                descreen_stage(normal_dir, descreened, args.descreen_strength)
                normal_dir = descreened
            if args.sr_backend == "swin2sr":
                sr_stage(normal_dir, out_dir, args.sr_model)
            else:
                lanczos_stage(normal_dir, out_dir, max(1, int(args.scale)))

        if doc_images:
            doc_dir = copy_images(doc_images, tmp_root / "document")
            reconstruct_lanczos_stage(
                doc_dir, out_dir, max(1, int(args.scale)),
                args.reconstruct_strength, args.ink_strength,
                args.preserve_more_text)

    write_score_outputs(in_dir, out_dir, mode_by_name, pipeline_by_name,
                        decision_by_name, reference_dir)
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
