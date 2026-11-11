from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Iterable

try:
    import cv2
except ImportError as exc:  # pragma: no cover - handled at runtime with a friendly error.
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - handled at runtime with a friendly error.
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - handled at runtime with a friendly error.
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None

try:
    import pyvirtualcam
except ImportError as exc:  # pragma: no cover - handled at runtime with a friendly error.
    pyvirtualcam = None  # type: ignore[assignment]
    _PYVIRTUALCAM_IMPORT_ERROR = exc
else:
    _PYVIRTUALCAM_IMPORT_ERROR = None


DEFAULT_ASCII_CHARS = " .:-=+*#%@"
CHAR_ASPECT_RATIO = 0.43


@dataclass(frozen=True)
class OutputSize:
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="camera-asci",
        description="Capture a webcam, render it as ASCII in the terminal, and stream it to a virtual camera.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index to open with OpenCV.")
    parser.add_argument("--capture-width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--capture-height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target processing and virtual camera FPS.")
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1920,
        help="Virtual camera output width used for the ASCII canvas.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=1080,
        help="Virtual camera output height used for the ASCII canvas.",
    )
    parser.add_argument(
        "--ascii-chars",
        default=DEFAULT_ASCII_CHARS,
        help="ASCII gradient ordered from darkest to lightest.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=0,
        help="Font size for the virtual camera canvas. 0 means auto-fit.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the webcam horizontally for a selfie-style preview.",
    )
    parser.add_argument(
        "--terminal-only",
        action="store_true",
        help="Skip the virtual camera stream and only render to the terminal.",
    )
    return parser.parse_args()


def load_monospace_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "DejaVuSansMono.ttf",
        "Consolas.ttf",
        "consola.ttf",
        "Courier New.ttf",
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        r"C:\Windows\Fonts\lucon.ttf",
        r"/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        r"/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    ]
    for candidate in candidates:
        with contextlib.suppress(Exception):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def open_capture(index: int, width: int, height: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if os.name == "nt" and hasattr(cv2, "CAP_DSHOW") else 0
    capture = cv2.VideoCapture(index, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise SystemExit(f"Unable to open webcam index {index}.")
    return capture


def estimate_terminal_ascii_size(frame_shape: tuple[int, int, int], terminal: OutputSize) -> tuple[int, int]:
    frame_height, frame_width = frame_shape[:2]
    cols = max(1, terminal.width)
    rows = max(1, int(cols * (frame_height / max(frame_width, 1)) * CHAR_ASPECT_RATIO))
    if rows > terminal.height:
        rows = max(1, terminal.height)
        cols = max(1, int(rows * (frame_width / max(frame_height, 1)) / CHAR_ASPECT_RATIO))
        cols = min(cols, terminal.width)
    return cols, rows


def frame_to_ascii(frame: np.ndarray, cols: int, rows: int, gradient: str) -> list[str]:
    if len(gradient) < 2:
        raise ValueError("ASCII gradient must contain at least two characters.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    mean = float(gray.mean())
    gain = min(4.0, max(1.0, 110.0 / max(mean, 1.0)))
    gray = cv2.convertScaleAbs(gray, alpha=gain, beta=0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = np.clip((gray.astype(np.float32) / 255.0) ** 0.8 * 255.0, 0, 255).astype(np.uint8)

    resized = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    index = (resized.astype(np.float32) * (len(gradient) - 1) / 255.0).astype(np.int32)
    lut = np.asarray(list(gradient))
    ascii_matrix = lut[index]
    return ["".join(row) for row in ascii_matrix]


def render_ascii_to_image(
    ascii_lines: Iterable[str],
    canvas: OutputSize,
    font_size: int,
) -> np.ndarray:
    lines = list(ascii_lines)
    font = load_monospace_font(font_size)
    probe = Image.new("RGB", (1, 1), (0, 0, 0))
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((0, 0), "M", font=font)
    char_width = max(1, bbox[2] - bbox[0])
    char_height = max(1, bbox[3] - bbox[1])
    line_height = max(1, char_height)
    max_line_len = max((len(line) for line in lines), default=1)

    text_image = Image.new("RGB", (max(1, char_width * max_line_len), max(1, line_height * len(lines))), (0, 0, 0))
    draw = ImageDraw.Draw(text_image)
    y = 0
    for line in lines:
        draw.text((0, y), line, font=font, fill=(255, 255, 255))
        y += line_height

    fitted = text_image.resize((canvas.width, canvas.height), resample=Image.Resampling.NEAREST)
    return np.asarray(fitted, dtype=np.uint8)


def choose_font_size(canvas: OutputSize, cols: int, rows: int, explicit: int) -> int:
    if explicit > 0:
        return explicit
    width_based = max(8, canvas.width // max(cols, 1))
    height_based = max(8, canvas.height // max(rows, 1))
    return max(8, min(width_based, height_based))


def read_quit_key() -> bool:
    if os.name == "nt":
        import msvcrt

        if not msvcrt.kbhit():
            return False
        key = msvcrt.getwch()
        return key.lower() == "q"

    import select

    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return False
    key = sys.stdin.read(1)
    return key.lower() == "q"


def choose_mode() -> str:
    print("1 - Live Camera")
    print("2 - Photo to ASCII")
    while True:
        choice = input("Choose an option (1/2): ").strip()
        if choice in {"1", "2"}:
            return "live" if choice == "1" else "photo"
        print("Invalid choice, enter 1 or 2.")


def run_photo_ascii(args: argparse.Namespace) -> int:
    path = input("Enter photo path: ").strip().strip('"')
    if not path:
        raise SystemExit("No path given.")
    if not os.path.isfile(path):
        raise SystemExit(f"File not found: {path}")

    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Unable to read image: {path}")

    terminal = OutputSize(*shutil.get_terminal_size(fallback=(80, 24)))
    cols, rows = estimate_terminal_ascii_size(frame.shape, terminal)
    ascii_lines = frame_to_ascii(frame, cols, rows, args.ascii_chars)
    print()
    print("\n".join(ascii_lines))
    return 0


def setup_terminal() -> None:
    sys.stdout.write("\033[2J\033[H\033[?25l")
    sys.stdout.flush()


def restore_terminal() -> None:
    sys.stdout.write("\033[?25h\033[0m\n")
    sys.stdout.flush()


def main() -> int:
    args = parse_args()

    if args.fps <= 0:
        raise SystemExit("--fps must be greater than zero.")

    missing: list[str] = []
    if cv2 is None:
        missing.append("opencv-python")
    if np is None:
        missing.append("numpy")
    if Image is None:
        missing.append("pillow")
    if missing:
        raise SystemExit(
            "Missing required dependencies: "
            + ", ".join(missing)
            + ". Install them with: pip install -r requirements.txt"
        )

    mode = choose_mode()

    if mode == "photo":
        return run_photo_ascii(args)

    if not args.terminal_only and pyvirtualcam is None:
        raise SystemExit(
            "pyvirtualcam is not installed. Install requirements first: pip install -r requirements.txt"
        ) from _PYVIRTUALCAM_IMPORT_ERROR

    capture = open_capture(args.camera_index, args.capture_width, args.capture_height)
    camera_size = OutputSize(args.camera_width, args.camera_height)

    setup_terminal()
    frame_period = 1.0 / args.fps
    next_tick = time.perf_counter()

    try:
        camera_context = contextlib.nullcontext(None)
        if not args.terminal_only:
            try:
                camera_context = pyvirtualcam.Camera(
                    width=camera_size.width,
                    height=camera_size.height,
                    fps=args.fps,
                )
            except Exception as exc:  # pragma: no cover - depends on host system driver availability.
                platform_hint = (
                    "On Windows, install or enable OBS Virtual Camera. "
                    "On Linux, load v4l2loopback. On macOS, ensure the pyvirtualcam backend is available."
                )
                raise SystemExit(f"Virtual camera unavailable: {exc}\n{platform_hint}") from exc

        with camera_context as virtual_camera:
            while True:
                ok, frame = capture.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                if args.mirror:
                    frame = cv2.flip(frame, 1)

                terminal = OutputSize(*shutil.get_terminal_size(fallback=(80, 24)))
                cols, rows = estimate_terminal_ascii_size(frame.shape, terminal)
                ascii_lines = frame_to_ascii(frame, cols, rows, args.ascii_chars)

                sys.stdout.write("\033[H\033[J")
                sys.stdout.write("\n".join(ascii_lines))
                sys.stdout.flush()

                if virtual_camera is not None:
                    font_size = choose_font_size(camera_size, cols, rows, args.font_size)
                    rgb_frame = render_ascii_to_image(ascii_lines, camera_size, font_size)
                    virtual_camera.send(rgb_frame)

                if read_quit_key():
                    break

                next_tick += frame_period
                sleep_for = next_tick - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        with contextlib.suppress(Exception):
            capture.release()
        restore_terminal()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
