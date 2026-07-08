# ID Detection And Enhancement Workflow

This project has two main stages:

1. Detect and crop the ID/passport from source documents with `detect_ml.py`.
2. Enhance or reconstruct the cropped images with `enhance_folder.py`.

Both stages write score files so you can inspect what happened.

## Stage 1: Detect ID/Passport

Run detection on a folder of PDFs or images:

```powershell
python detect_ml.py faturim -o stage1_interactive --debug
```

Useful options:

```powershell
python detect_ml.py faturim -o stage1_interactive --no-sam
python detect_ml.py faturim -o stage1_interactive --no-ml
python detect_ml.py faturim -o stage1_interactive --device cuda
```

Detection outputs:

- `stage1_interactive/<document>.png`: normalized crop for each detected ID/passport.
- `stage1_interactive/summary.csv`: one row per source document.
- `stage1_interactive/debug_report.html`: visual report when `--debug` is used.
- `stage1_interactive/pages/*.jpg`: page-level debug images with candidate boxes and scores.

Detection score columns:

- `status`: `ok`, `no_card`, or `unreadable`.
- `card_page`: selected page number.
- `detection_score`: confidence from the identity-document heuristic.
- `crop_type`: `id_card`, `passport_details`, or fallback type.
- `output_image`: cropped image sent to enhancement.

The detection score is a 0..1 heuristic. Higher is better. It combines card shape, portrait/photo evidence, MRZ/text evidence, color/detail, and page context.

## Stage 2: Enhance Crops

Default enhancement is automatic:

```powershell
python enhance_folder.py stage1_interactive -o enhanced_images
```

Default behavior:

```text
Good/normal crops -> NAFNet denoise -> Swin2SR x4
Damaged halftone/copier crops -> document reconstruction -> Lanczos x4
```

If you want a lightweight local run without neural SR:

```powershell
python enhance_folder.py stage1_interactive -o enhanced_images --no-denoise --sr-backend lanczos --scale 4
```

Enhancement outputs:

- `enhanced_images/<document>.png`: enhanced image.
- `enhanced_images/enhancement_scores.csv`: per-image standard restoration/IQA metrics.
- `enhanced_images/generative_metrics.csv`: set-level FID/KID metrics when the required metric packages are installed.
- `enhanced_images/enhancement_report.html`: visual metric report with thumbnails.

## Damaged Halftone Scans

Document reconstruction is selected automatically from image statistics, not from the filename. The auto routing risk looks for the combination that usually breaks neural/photo super-resolution: high fine-grain texture, too many tiny edges, and dense copier/halftone structure.

Recommended general run:

```powershell
python enhance_folder.py stage1_interactive -o enhanced_images --reconstruct-mode auto
```

Useful auto-selection controls:

```powershell
python enhance_folder.py stage1_interactive -o enhanced_images --auto-reconstruct-threshold 0.55
python enhance_folder.py stage1_interactive -o enhanced_images --reconstruct-mode off
python enhance_folder.py stage1_interactive -o enhanced_images --reconstruct-mode all
```

Manual filename patterns are only an override for experiments:

```powershell
python enhance_folder.py stage1_interactive -o enhanced_images --document-pattern "*damaged*"
```

Use `--reconstruct-mode all` only when every input has the same damaged halftone/copy texture.

## Enhancement Metrics Report

`enhancement_scores.csv` merges detection results with standard image reconstruction metrics when the input folder contains `summary.csv`.

Important columns:

- `detection_score`: copied from `stage1_interactive/summary.csv`.
- `mode`: `normal`, `auto_reconstruct`, or `document_reconstruct`.
- `auto_reconstruction_risk`: automatic routing value only; this is not an evaluation metric.
- `reconstruction_reason`: raw feature values used by auto routing.
- `pipeline`: the enhancement path used for that image.
- `reference_type`: `input_crop` by default, or `reference_dir` when you pass clean references.
- `reference_psnr`: PSNR against the reference image. Higher is better.
- `reference_ssim`: SSIM against the reference image. Higher is better.
- `reference_lpips`: LPIPS perceptual distance when `lpips` is installed. Lower is better.
- `reference_niqe`: NIQE for the reference image when the bundled BasicSR/NAFNet NIQE dependencies are available. Lower is usually better.
- `output_niqe`: NIQE for the enhanced output. Lower is usually better.
- `niqe_delta`: `output_niqe - reference_niqe`.

PSNR, SSIM, and LPIPS are full-reference metrics. If you do not provide a reference folder, the script uses the input crop as the reference and the metrics only measure fidelity to the input, not true ground-truth reconstruction quality.

For proper reconstruction evaluation with clean targets, use matching filenames:

```powershell
python enhance_folder.py stage1_interactive -o enhanced_images --reference-dir clean_reference_images
```

`generative_metrics.csv` contains set-level GAN/generative metrics:

- `fid`: Frechet Inception Distance. Lower is better.
- `kid_mean`: Kernel Inception Distance mean. Lower is better.
- `kid_std`: Kernel Inception Distance standard deviation.

FID and KID are distribution metrics, not single-image scores. They require an Inception feature extractor through `torchmetrics` and its image dependencies. If those packages are not installed, the CSV will mark them as `unavailable` instead of writing a fake value. With only a few ID crops, FID/KID are unstable; use a held-out real or ground-truth reference set for serious GAN-style evaluation.

## Suggested Workflow

1. Detect and crop:

```powershell
python detect_ml.py faturim -o stage1_interactive --debug
```

2. Inspect detection scores:

```text
stage1_interactive/summary.csv
stage1_interactive/debug_report.html
```

3. Enhance normally:

```powershell
python enhance_folder.py stage1_interactive -o enhanced_images
```

4. Enhance with automatic reconstruction selection:

```powershell
python enhance_folder.py stage1_interactive -o enhanced_images --reconstruct-mode auto
```

5. Inspect enhancement metrics:

```text
enhanced_images/enhancement_scores.csv
enhanced_images/generative_metrics.csv
enhanced_images/enhancement_report.html
```

## Practical Notes

- For clean images, `auto` should keep the normal path. Use `--reconstruct-mode off` if you want to disable reconstruction completely.
- Higher `--auto-reconstruct-threshold` makes auto routing more conservative.
- For damaged scans, higher `--reconstruct-strength` removes more dots but can blur or remove small letters.
- `--preserve-more-text` keeps more letters/numbers but also keeps more residual dot texture.
- `--ink-strength` darkens restored text strokes.
- `--sr-backend swin2sr` may sharpen useful detail on clean images, but can amplify halftone dots on damaged scans.
