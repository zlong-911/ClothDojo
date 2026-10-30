"""Compose a source-audited camera pullback over 200 recorded trajectories.

CPU only. Inputs are 320x240/10 Hz excerpts of the original source videos.
The 25 fps output preserves 1x time; each sampled source frame may repeat.
"""
from pathlib import Path
import json
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    cv2.setNumThreads(1)
    items = json.loads((ROOT / "outputs/presentation/website_rgb_pool_20260928/manifest.json").read_text())
    assert len(items) == 200
    assert len({item["source_video"] for item in items}) == 200
    assert all(item["task"] == "flatten" for item in items[:100])
    assert all(item["task"] == "fold" for item in items[100:])
    captures = []
    encoder = None
    output = ROOT / "website/assets/curated/generation-200.mp4"
    try:
        for index in range(200):
            source = ROOT / f"outputs/website_media_20261030/reorganized/thumb-clips/{index:03d}.mp4"
            cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
            if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - 10) > 0.01:
                cap.release()
                raise ValueError(f"Expected readable 10 Hz excerpt: {source}")
            captures.append(cap)
        encoder = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x720",
            "-r", "25", "-i", "-", "-an", "-c:v", "libx264", "-threads", "4",
            "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ], stdin=subprocess.PIPE)
        world = np.empty((2400, 6400, 3), np.uint8)
        previous = -1
        for frame in range(300):
            source_frame = frame * 10 // 25
            if source_frame != previous:
                for index, cap in enumerate(captures):
                    ok, image = cap.read()
                    if not ok or image.shape != (240, 320, 3):
                        raise RuntimeError(f"Invalid source {index} at frame {source_frame}")
                    row, col = divmod(index, 20)
                    world[row*240:(row+1)*240, col*320:(col+1)*320] = image
                previous = source_frame
            progress = np.clip((frame / 25 - 1.5) / 4, 0, 1)
            eased = progress * progress * (3 - 2 * progress)
            fraction = 0.3 + 0.7 * eased
            width, height = round(6400 * fraction), round(2400 * fraction)
            x, y = (6400-width)//2, (2400-height)//2
            image = cv2.resize(world[y:y+height, x:x+width], (1920,720), interpolation=cv2.INTER_AREA)
            encoder.stdin.write(image.tobytes())
            if frame % 50 == 0:
                print(f"composed {frame}/300", flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Scale encoder failed")
        encoder = None
        print("200 source trajectories;", len({x["asset_id"] for x in items}), "unique garments", flush=True)
    finally:
        for cap in captures:
            cap.release()
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            encoder.wait()


if __name__ == "__main__":
    main()
