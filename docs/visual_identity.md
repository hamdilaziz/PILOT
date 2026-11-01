# README visuals

PILOT uses a document-and-token visual identity: localization corners around a letter P,
a restrained blue/cyan palette and a schematic document-to-sequence illustration.
All banners, diagrams and animation files are stored in the repository.

## Assets

| File | Purpose |
| --- | --- |
| `assets/readme/hero-light.png`, `hero-dark.png` | README hero, with theme selection |
| `assets/readme/architecture-light.png`, `architecture-dark.png` | Model overview |
| `assets/readme/demo.gif` | Animated replay of a recorded prediction |
| `assets/readme/demo-static.png` | Static, full-result alternative |
| `assets/brand/pilot-mark.svg` | Editable project mark |
| `assets/brand/social-preview.png` | 1280 × 640 repository sharing image |

The SVG sources accompany the hero and architecture PNGs. The README uses PNGs for consistent
font rendering across platforms; `<picture>` selects the light or dark variant. No custom CSS
or JavaScript is required in the README.

## Rebuild

Asset generation is separate from inference and training. From the repository root:

```bash
pip install pillow cairosvg tqdm
python tools/build_readme_assets.py
```

Install the system Cairo library if required by CairoSVG. The generator uses DejaVu Sans and
DejaVu Sans Mono; install those fonts when rebuilding. SVG sources retain editable text.
The generated PNGs are rendered at twice their SVG view-box resolution.

The animation uses only `input_samples/iam.png` and `outputs/iam_ocr_with_boxes.json`.
Each revealed box and text line comes from that stored model output. It does not run inference,
change predictions or imply ground-truth correctness. Frame durations are presentation choices,
not measured latency. The first GIF frame shows the full result so a paused animation is useful.
The linked static PNG also shows the complete result.

The hero's document and token stream are schematic illustrations, not benchmark predictions.
Rebuild the GIF whenever changing its source prediction. Check the actual result at README display
size, along with the light and dark hero variants, before committing new media.

## Repository sharing image

Upload `assets/brand/social-preview.png` under repository **Settings → Social preview → Edit →
Upload an image**. Committing the image does not itself change that repository setting.
The file has a solid background, a 1280 × 640 canvas and a size below 1 MB.
See [GitHub's social preview instructions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview).

GitHub supports [theme-aware images](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/quickstart-for-writing-on-github)
and [GIF/PNG/SVG assets](https://docs.github.com/en/repositories/working-with-files/using-files/working-with-non-code-files).
SVG animation and inline scripts are not supported. Visitors can also
[disable GIF autoplay](https://docs.github.com/en/account-and-profile/how-tos/account-settings/managing-accessibility-settings).
