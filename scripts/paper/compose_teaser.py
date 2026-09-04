#!/usr/bin/env python3
"""Compose the paper teaser from selected lossless RTX renders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

WIDTH = 2400
HEIGHT = 700
MARGIN = 10
GUTTER = 4
HEADER_HEIGHT = 78
LEFT_WIDTH = 840
RIGHT_X = 862
RIGHT_WIDTH = 1528
FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
# The flat-lay camera is already composed to the square frame.  Keeping the
# full frame preserves sleeve tips and, critically, the shared metric scale.
ASSET_CROP = (0.0, 0.0, 1.0, 1.0)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-order", type=Path)
    parser.add_argument(
        "--style",
        choices=("baseline", "refined", "refined_tall", "refined_margin_labels"),
        default="baseline",
    )
    return parser.parse_args()


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(f"teaser font is unavailable: {path}")
    return ImageFont.truetype(str(path), size=size)


def _image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.fit(
            source.convert("RGB"),
            size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


def _panel(
    canvas: Image.Image,
    path: Path,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    canvas.paste(_image(path, (right - left, bottom - top)), (left, top))


def _asset_panel(
    canvas: Image.Image,
    path: Path,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    # Every asset render uses the same square camera and world scale.  Keep the
    # same crop for every image so relative garment size survives composition.
    # In particular, do not fit each foreground independently because that
    # would make physically small garments appear as large as wide garments.
    with Image.open(path) as source:
        source = source.convert("RGB")
        crop = tuple(
            round(fraction * dimension)
            for fraction, dimension in zip(
                ASSET_CROP,
                (source.width, source.height, source.width, source.height),
                strict=True,
            )
        )
        image = source.crop(crop)
        image.thumbnail(
            (right - left, bottom - top),
            resample=Image.Resampling.LANCZOS,
        )
        background = Image.new(
            "RGB",
            (right - left, bottom - top),
            image.getpixel((0, 0)),
        )
        background.paste(
            image,
            (
                (background.width - image.width) // 2,
                (background.height - image.height) // 2,
            ),
        )
    canvas.paste(background, (left, top))


def _tag(
    canvas: Image.Image,
    text: str,
    position: tuple[int, int],
    *,
    font: ImageFont.FreeTypeFont,
    padding: tuple[int, int] = (17, 10),
) -> None:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x, y = position
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0] + 2 * padding[0]
    height = bounds[3] - bounds[1] + 2 * padding[1]
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=13,
        fill=(22, 25, 29, 218),
    )
    draw.text(
        (x + padding[0], y + padding[1] - bounds[1]),
        text,
        font=font,
        fill=(255, 255, 255, 255),
    )
    canvas.alpha_composite(overlay)


def _vertical_tag(
    canvas: Image.Image,
    text: str,
    center: tuple[int, int],
    *,
    font: ImageFont.FreeTypeFont,
) -> None:
    draw = ImageDraw.Draw(canvas)
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0] + 12
    height = bounds[3] - bounds[1] + 8
    label = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.rounded_rectangle(
        (0, 0, width - 1, height - 1),
        radius=7,
        fill=(255, 255, 255, 238),
        outline=(205, 209, 214, 255),
        width=1,
    )
    label_draw.text(
        (6, 4 - bounds[1]),
        text,
        font=font,
        fill=(54, 59, 66, 255),
    )
    label = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.alpha_composite(
        label,
        (
            center[0] - label.width // 2,
            center[1] - label.height // 2,
        ),
    )


def compose(
    renders: Path,
    plan: Path,
    output: Path,
    *,
    style: str = "baseline",
    asset_order: Path | None = None,
) -> None:
    if style == "baseline":
        height = HEIGHT
        header_height = HEADER_HEIGHT
        title_size, subtitle_size, robot_size, row_size = 29, 18, 23, 25
        left_title = "Garment Asset Diversity"
        left_subtitle = "24 geometries  ·  metric-scale flat lays"
        right_title = "Cross-Embodiment Cloth Manipulation"
        row_labels = ("Flattening", "Folding")
        row_padding = (11, 5)
        row_offset = 8
    elif style in {"refined", "refined_tall", "refined_margin_labels"}:
        height = 700 if style == "refined" else 760
        header_height = 74
        title_size, subtitle_size, robot_size, row_size = 27, 17, 21, 19
        left_title = "Diverse Garment Assets"
        left_subtitle = "24 geometries  ·  shared metric scale"
        right_title = "Cross-Embodiment Manipulation"
        row_labels = ("Flatten", "Fold")
        row_padding = (9, 4)
        row_offset = 6
    else:
        raise ValueError(f"unknown teaser composition style: {style}")

    canvas = Image.new("RGBA", (WIDTH, height), (255, 255, 255, 255))
    title_font = _font("DejaVuSans-Bold.ttf", title_size)
    subtitle_font = _font("DejaVuSans.ttf", subtitle_size)
    robot_font = _font("DejaVuSans-Bold.ttf", robot_size)
    row_font = _font("DejaVuSans-Bold.ttf", row_size)
    render_plan = json.loads(plan.read_text(encoding="utf-8"))
    asset_names = tuple(job["name"] for job in render_plan["jobs"] if job.get("group") == "assets")
    if len(asset_names) != 24:
        raise ValueError(f"paper teaser requires 24 asset jobs, got {len(asset_names)}")
    if asset_order is not None:
        order = json.loads(asset_order.read_text(encoding="utf-8"))
        ordered_names = tuple(order.get("asset_names", ()))
        if len(ordered_names) != len(asset_names) or set(ordered_names) != set(asset_names):
            raise ValueError("asset order must contain every planned asset exactly once")
        asset_names = ordered_names
    asset_columns = 6
    asset_rows = 4
    asset_width = (LEFT_WIDTH - (asset_columns - 1) * GUTTER) // asset_columns
    asset_height = (height - header_height - MARGIN - (asset_rows - 1) * GUTTER) // asset_rows
    for index, name in enumerate(asset_names):
        row, column = divmod(index, asset_columns)
        left = MARGIN + column * (asset_width + GUTTER)
        top = header_height + row * (asset_height + GUTTER)
        _asset_panel(
            canvas,
            renders / "assets" / f"{name}_00.png",
            (left, top, left + asset_width, top + asset_height),
        )
    header_draw = ImageDraw.Draw(canvas)
    header_draw.text(
        (MARGIN + 2, 2),
        left_title,
        fill=(24, 27, 31, 255),
        font=title_font,
    )
    header_draw.text(
        (MARGIN + 3, 42),
        left_subtitle,
        fill=(91, 98, 107, 255),
        font=subtitle_font,
    )
    right_title_box = header_draw.textbbox((0, 0), right_title, font=title_font)
    header_draw.text(
        (
            RIGHT_X + (RIGHT_WIDTH - (right_title_box[2] - right_title_box[0])) // 2,
            0,
        ),
        right_title,
        fill=(24, 27, 31, 255),
        font=title_font,
    )
    header_draw.line(
        (MARGIN, header_height - 3, LEFT_WIDTH, header_height - 3),
        fill=(205, 209, 214, 255),
        width=2,
    )
    header_draw.line(
        (RIGHT_X, header_height - 3, WIDTH - MARGIN, header_height - 3),
        fill=(205, 209, 214, 255),
        width=2,
    )
    header_draw.line(
        (LEFT_WIDTH + 10, MARGIN, LEFT_WIDTH + 10, height - MARGIN),
        fill=(222, 225, 229, 255),
        width=2,
    )

    column_specs = (
        ("piper", "PIPER"),
        ("aloha2", "ALOHA"),
        ("x5", "ARX X5"),
        ("fr3_duo", "Franka"),
        ("ur5e_2f85", "UR5e"),
    )
    tile_width = (RIGHT_WIDTH - 4 * GUTTER) // 5
    task_tile_height = (height - header_height - MARGIN - GUTTER) // 2
    for index, (_, robot) in enumerate(column_specs):
        left = RIGHT_X + index * (tile_width + GUTTER)
        robot_box = header_draw.textbbox((0, 0), robot, font=robot_font)
        robot_width = robot_box[2] - robot_box[0]
        header_draw.text(
            (left + (tile_width - robot_width) // 2, 43 - robot_box[1]),
            robot,
            fill=(24, 27, 31, 255),
            font=robot_font,
        )

    for row, task in enumerate(("flatten", "fold")):
        top = header_height + row * (task_tile_height + GUTTER)
        for index, (stem, _) in enumerate(column_specs):
            left = RIGHT_X + index * (tile_width + GUTTER)
            _panel(
                canvas,
                renders / "task_grid" / f"{stem}_{task}_00.png",
                (left, top, left + tile_width, top + task_tile_height),
            )
        if style == "refined_margin_labels":
            _vertical_tag(
                canvas,
                row_labels[row].upper(),
                (LEFT_WIDTH + 10, top + task_tile_height // 2),
                font=row_font,
            )
        else:
            _tag(
                canvas,
                row_labels[row],
                (RIGHT_X + row_offset, top + row_offset),
                font=row_font,
                padding=row_padding,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output, optimize=True)


def main() -> None:
    args = _arguments()
    compose(
        args.renders.expanduser().resolve(),
        args.plan.expanduser().resolve(),
        args.output.expanduser().resolve(),
        style=args.style,
        asset_order=(args.asset_order.expanduser().resolve() if args.asset_order else None),
    )


if __name__ == "__main__":
    main()
