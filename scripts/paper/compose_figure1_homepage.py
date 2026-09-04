#!/usr/bin/env python3
"""Compose the full-width homepage version of ClothDojo Figure 1."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

LEFT_WIDTH = 3000
WIDTH = 5030
FINAL_WIDTH_IN = 7.0
TEX_POINTS_PER_INCH = 72.27
PX_PER_PT = WIDTH / (FINAL_WIDTH_IN * TEX_POINTS_PER_INCH)
PANEL_HEADER_H = 108
TITLE_PT = 7.5
SECONDARY_PT = 5.2
BG = (255, 255, 255)
INK = (27, 33, 40)
MUTED = (87, 98, 108)
TEAL = (22, 137, 139)
ORANGE = (222, 139, 48)
BLUE = (49, 111, 167)
GREEN = (55, 146, 92)
FONT_ROOT = Path("/usr/share/fonts/truetype/lato")


def _font(size_pt: float, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Return a font sized in points at the final 7-inch publication width."""
    name = "Lato-Bold.ttf" if bold else "Lato-Regular.ttf"
    return ImageFont.truetype(str(FONT_ROOT / name), round(size_pt * PX_PER_PT))


def _fit(path: Path, size: tuple[int, int]) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as source:
        return ImageOps.fit(source.convert("RGB"), size, Image.Resampling.LANCZOS)


def _split_frame(nocs: Path, rgb: Path, size: tuple[int, int]) -> Image.Image:
    width, height = size
    left = _fit(nocs, size)
    right = _fit(rgb, size)
    split = width // 2
    frame = left.copy()
    frame.paste(right.crop((split, 0, width, height)), (split, 0))
    ImageDraw.Draw(frame).line((split, 0, split, height), fill="white", width=4)
    return frame


def _panel_header(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    number: int,
    title: str,
    note: str,
) -> None:
    height = PANEL_HEADER_H
    center_y = y + height // 2
    title_font = _font(TITLE_PT, bold=True)
    note_font = _font(SECONDARY_PT)
    # Align the title to the full module edge, including the vertical row labels.
    title_x = 10
    note_x = 2987
    number_text = f"{number:02d}"
    draw.text(
        (title_x, center_y), number_text, font=title_font, fill=INK, anchor="lm"
    )
    number_box = draw.textbbox(
        (title_x, center_y), number_text, font=title_font, anchor="lm"
    )
    title_start = number_box[2] + 20
    draw.text((title_start, center_y), title, font=title_font, fill=INK, anchor="lm")
    draw.text(
        (note_x, center_y),
        note,
        font=note_font,
        fill=INK,
        anchor="rm",
    )
    title_box = draw.textbbox(
        (title_start, center_y), title, font=title_font, anchor="lm"
    )
    note_box = draw.textbbox((note_x, center_y), note, font=note_font, anchor="rm")
    if title_box[2] + 32 > note_box[0]:
        raise ValueError(f"panel header text overlaps: {title!r} / {note!r}")


def _draw_a_row(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    nocs: list[Path],
    rgb: list[Path],
    y: int,
    tile_x: int,
    tile_w: int,
    tile_h: int,
    gap: int,
) -> None:
    for index, (nocs_path, rgb_path) in enumerate(zip(nocs, rgb, strict=True)):
        x = tile_x + index * (tile_w + gap)
        canvas.paste(_split_frame(nocs_path, rgb_path, (tile_w, tile_h)), (x, y))
        draw.rectangle((x, y, x + tile_w, y + tile_h), outline="white", width=3)


def _draw_b_row(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    paths: list[Path],
    y: int,
    tile_x: int,
    tile_w: int,
    tile_h: int,
    gap: int,
) -> None:
    for index, path in enumerate(paths):
        x = tile_x + index * (tile_w + gap)
        canvas.paste(_fit(path, (tile_w, tile_h)), (x, y))
        draw.rectangle((x, y, x + tile_w, y + tile_h), outline="white", width=3)


def _vertical_task_label(
    canvas: Image.Image,
    *,
    y: int,
    task: str,
    gutter_w: int,
    tile_h: int,
) -> None:
    """Place a restrained 90-degree task label beside an image row."""
    strip = Image.new("RGBA", (tile_h, gutter_w), (0, 0, 0, 0))
    strip_draw = ImageDraw.Draw(strip)
    font = _font(SECONDARY_PT)
    text_box = strip_draw.textbbox((0, 0), task.upper(), font=font)
    if text_box[2] - text_box[0] > tile_h - 28:
        raise ValueError(f"vertical task label does not fit: {task!r}")
    strip_draw.text(
        (tile_h / 2, gutter_w / 2 - 3),
        task.upper(),
        font=font,
        fill=(*INK, 255),
        anchor="mm",
    )
    rotated = strip.rotate(90, expand=True)
    canvas.paste(rotated, (10, y), rotated)


def _evaluation_icon(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    subtitle: str,
    color: tuple[int, int, int],
    kind: str,
) -> None:
    x0, y0, x1, y1 = box
    icon_x = x0 + 44
    icon_y = (y0 + y1) // 2
    if kind == "axes":
        for offset in (-24, 0, 24):
            draw.line(
                (icon_x - 24, icon_y + offset, icon_x + 24, icon_y + offset), fill=color, width=5
            )
            draw.ellipse(
                (
                    icon_x - 8 + offset // 2,
                    icon_y + offset - 8,
                    icon_x + 8 + offset // 2,
                    icon_y + offset + 8,
                ),
                fill=color,
            )
    elif kind == "metrics":
        draw.arc(
            (icon_x - 34, icon_y - 34, icon_x + 34, icon_y + 34), 200, 520, fill=color, width=7
        )
        draw.line((icon_x, icon_y, icon_x + 23, icon_y - 21), fill=color, width=6)
        draw.ellipse((icon_x - 6, icon_y - 6, icon_x + 6, icon_y + 6), fill=color)
    else:
        for index, height in enumerate((28, 47, 68)):
            bx = icon_x - 35 + index * 27
            draw.rounded_rectangle(
                (bx, icon_y + 35 - height, bx + 18, icon_y + 35), radius=4, fill=color
            )
    title_font = _font(SECONDARY_PT)
    title_xy = (x0 + 100, icon_y)
    title_box = draw.textbbox(title_xy, title, font=title_font, anchor="lm")
    if title_box[2] > x1 - 22:
        raise ValueError(f"evaluation title does not fit: {title!r}")
    draw.text(
        title_xy,
        title,
        font=title_font,
        fill=INK,
        anchor="lm",
    )
    if subtitle:
        draw.multiline_text(
            (x0 + 100, y0 + 92),
            subtitle,
            font=_font(6.0),
            fill=MUTED,
            spacing=4,
        )


def _dataset_grid(
    paths: list[Path],
    *,
    size: tuple[int, int],
    columns: int = 5,
) -> Image.Image:
    """Compose 25 rendered states, or sample them from legacy videos."""
    import cv2

    if len(paths) != 25:
        raise ValueError(f"expected 25 dataset videos, got {len(paths)}")
    gap = 9
    width, height = size
    tile_w = (width - gap * (columns - 1)) // columns
    rows = len(paths) // columns
    tile_h = (height - gap * (rows - 1)) // rows
    grid = Image.new("RGB", size, (238, 241, 243))
    for index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".png":
            with Image.open(path) as source:
                image = source.convert("RGB")
        else:
            capture = cv2.VideoCapture(str(path))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                capture.release()
                raise RuntimeError(f"cannot read video frames: {path}")
            fraction = 0.98
            capture.set(cv2.CAP_PROP_POS_FRAMES, round((frame_count - 1) * fraction))
            ok, frame = capture.read()
            capture.release()
            if not ok:
                raise RuntimeError(f"cannot sample {path} at {fraction:.2f}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        image = ImageOps.fit(image, (tile_w, tile_h), Image.Resampling.LANCZOS)
        row, column = divmod(index, columns)
        x = column * (tile_w + gap)
        y = row * (tile_h + gap)
        grid.paste(image, (x, y))
    return grid


def _right_pipeline(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    top: int,
    grid_bottom: int,
    videos: list[Path],
) -> None:
    x0 = LEFT_WIDTH + 18
    x1 = WIDTH - 12
    grid_x = x0 + 12
    grid_y = top + PANEL_HEADER_H + 6
    grid_w = x1 - x0 - 24
    grid_h = grid_bottom - grid_y
    eval_top = grid_y + grid_h + 72
    eval_card_y = eval_top + 72
    eval_card_h = 150

    title_x = grid_x
    header_y = top + PANEL_HEADER_H // 2
    title_font = _font(TITLE_PT, bold=True)
    draw.text((title_x, header_y), "03", font=title_font, fill=INK, anchor="lm")
    number_box = draw.textbbox(
        (title_x, header_y), "03", font=title_font, anchor="lm"
    )
    draw.text(
        (number_box[2] + 20, header_y),
        "Benchmark Dataset",
        font=title_font,
        fill=INK,
        anchor="lm",
    )
    draw.text(
        (grid_x + grid_w, header_y),
        "500+ garments  ·  5K+ trajectories",
        font=_font(SECONDARY_PT),
        fill=INK,
        anchor="rm",
    )

    dataset_grid = _dataset_grid(videos, size=(grid_w, grid_h))
    canvas.paste(dataset_grid, (grid_x, grid_y))

    eval_header_y = eval_top + title_font.size // 2
    draw.text(
        (title_x, eval_header_y), "04", font=title_font, fill=INK, anchor="lm"
    )
    number_box = draw.textbbox(
        (title_x, eval_header_y), "04", font=title_font, anchor="lm"
    )
    draw.text(
        (number_box[2] + 20, eval_header_y),
        "Benchmark Evaluation",
        font=title_font,
        fill=INK,
        anchor="lm",
    )
    card_gap = 16
    card_x0 = grid_x
    card_w = (grid_w - 2 * card_gap) // 3
    _evaluation_icon(
        draw,
        (card_x0, eval_card_y, card_x0 + card_w, eval_card_y + eval_card_h),
        title="Generalization",
        subtitle="",
        color=TEAL,
        kind="axes",
    )
    _evaluation_icon(
        draw,
        (
            card_x0 + card_w + card_gap,
            eval_card_y,
            card_x0 + 2 * card_w + card_gap,
            eval_card_y + eval_card_h,
        ),
        title="Diagnostics",
        subtitle="",
        color=ORANGE,
        kind="metrics",
    )
    _evaluation_icon(
        draw,
        (
            card_x0 + 2 * (card_w + card_gap),
            eval_card_y,
            grid_x + grid_w,
            eval_card_y + eval_card_h,
        ),
        title="Policy Ranking",
        subtitle="",
        color=GREEN,
        kind="performance",
    )


def compose(
    a_nocs: Path,
    a_nocs_group: str,
    a_rgb: Path,
    b_replay: Path,
    dataset_video_dir: Path,
    output: Path,
) -> None:
    a_unfold_nocs = [
        a_nocs / "unfold" / a_nocs_group / f"unfold_{index:02d}.png" for index in range(5)
    ]
    a_fold_nocs = [a_nocs / "fold" / a_nocs_group / f"fold_{index:02d}.png" for index in range(5)]
    a_unfold_rgb = [
        a_rgb / "unfold/cam_high_95_rgb_selected5" / f"unfold_{index:02d}.png" for index in range(5)
    ]
    a_fold_rgb = [
        a_rgb / "fold/cam_high_95_rgb_selected5" / f"fold_{index:02d}.png" for index in range(5)
    ]
    robots = (
        ("piper", "Piper"),
        ("aloha2", "ALOHA 2"),
        ("x5", "ARX X5"),
        ("fr3_duo", "Franka FR3"),
        ("ur5e_2f85", "UR5e"),
    )
    b_unfold = [b_replay / f"b_{robot}_flatten_00.png" for robot, _ in robots]
    b_fold = [b_replay / f"b_{robot}_fold_00.png" for robot, _ in robots]
    dataset_frames = sorted(dataset_video_dir.glob("[0-9][0-9]_*.png"))
    dataset_videos = sorted(dataset_video_dir.glob("[0-9][0-9]_*.mp4"))
    dataset_media = dataset_frames if len(dataset_frames) == 25 else dataset_videos
    if len(dataset_media) != 25:
        raise FileNotFoundError(
            f"expected 25 dataset PNGs or videos in {dataset_video_dir}, "
            f"found {len(dataset_frames)} PNGs and {len(dataset_videos)} videos"
        )
    all_paths = (
        *a_unfold_nocs,
        *a_fold_nocs,
        *a_unfold_rgb,
        *a_fold_rgb,
        *b_unfold,
        *b_fold,
    )
    missing = [path for path in all_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    margin = 12
    gutter_w = 64
    label_w = 50
    tile_x = margin + gutter_w
    gap = 14
    tile_w = (LEFT_WIDTH - margin - tile_x - 4 * gap) // 5
    tile_h = 420
    a_row_gap = 18
    b_row_gap = 18
    header_gap = 6

    a_header_y = 14
    a_flat_y = a_header_y + PANEL_HEADER_H + header_gap
    a_fold_y = a_flat_y + tile_h + a_row_gap

    b_header_y = a_fold_y + tile_h + 30
    b_flat_y = b_header_y + PANEL_HEADER_H + header_gap
    b_fold_y = b_flat_y + tile_h + b_row_gap

    height = b_fold_y + tile_h + 8
    canvas = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(canvas)

    _panel_header(
        draw,
        y=a_header_y,
        number=1,
        title="Privileged Trajectory Agent",
        note="continuous rollouts  ·  NOCS  |  RGB",
    )
    _draw_a_row(
        canvas,
        draw,
        nocs=a_unfold_nocs,
        rgb=a_unfold_rgb,
        y=a_flat_y,
        tile_x=tile_x,
        tile_w=tile_w,
        tile_h=tile_h,
        gap=gap,
    )
    _vertical_task_label(
        canvas,
        y=a_flat_y,
        task="Flatten",
        gutter_w=label_w,
        tile_h=tile_h,
    )
    _vertical_task_label(
        canvas,
        y=a_fold_y,
        task="Fold",
        gutter_w=label_w,
        tile_h=tile_h,
    )
    _draw_a_row(
        canvas,
        draw,
        nocs=list(reversed(a_fold_nocs)),
        rgb=list(reversed(a_fold_rgb)),
        y=a_fold_y,
        tile_x=tile_x,
        tile_w=tile_w,
        tile_h=tile_h,
        gap=gap,
    )
    _panel_header(
        draw,
        y=b_header_y,
        number=2,
        title="Retargeting & Visual Randomization",
        note="5 embodiments   ·   1K+ appearances",
    )
    _draw_b_row(
        canvas,
        draw,
        paths=b_unfold,
        y=b_flat_y,
        tile_x=tile_x,
        tile_w=tile_w,
        tile_h=tile_h,
        gap=gap,
    )
    _vertical_task_label(
        canvas,
        y=b_flat_y,
        task="Flatten",
        gutter_w=label_w,
        tile_h=tile_h,
    )
    _vertical_task_label(
        canvas,
        y=b_fold_y,
        task="Fold",
        gutter_w=label_w,
        tile_h=tile_h,
    )
    _draw_b_row(
        canvas,
        draw,
        paths=b_fold,
        y=b_fold_y,
        tile_x=tile_x,
        tile_w=tile_w,
        tile_h=tile_h,
        gap=gap,
    )

    _right_pipeline(
        canvas,
        draw,
        top=a_header_y,
        grid_bottom=b_flat_y + 4 * tile_h // 5,
        videos=dataset_media,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True, dpi=(450, 450))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-nocs", type=Path, required=True)
    parser.add_argument("--a-nocs-group", required=True)
    parser.add_argument("--a-rgb", type=Path, required=True)
    parser.add_argument("--b-replay", type=Path, required=True)
    parser.add_argument("--dataset-video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compose(
        args.a_nocs.expanduser().resolve(),
        args.a_nocs_group,
        args.a_rgb.expanduser().resolve(),
        args.b_replay.expanduser().resolve(),
        args.dataset_video_dir.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
