# Camera ASCII

Capture a webcam, convert it to real-time ASCII art, render it in the terminal, and broadcast the result as a virtual camera.

## Features

- Startup menu: `1 - Live Camera`, `2 - Photo to ASCII`
- Live webcam capture with OpenCV
- ASCII terminal preview with low-flicker ANSI updates
- Virtual camera output via `pyvirtualcam`
- Convert any photo file to ASCII art
- Clean shutdown with `q` or `Ctrl+C`

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

You get a menu:

```
1 - Live Camera
2 - Photo to ASCII
```

**1 - Live Camera** — streams your webcam as ASCII in the terminal (and to the virtual camera unless `--terminal-only` is passed). Press `q` or `Ctrl+C` to exit.

**2 - Photo to ASCII** — asks for a photo path, converts it to ASCII, and prints it sized to your terminal:

```
Enter photo path: C:\path\to\photo.jpg
```

Use the built-in defaults for camera size, output size, contrast, and rendering.

## Virtual Camera Notes

`pyvirtualcam` needs a system virtual camera backend.

- Windows: enable OBS Virtual Camera or another supported virtual camera driver
- Linux: load `v4l2loopback`
- macOS: ensure the `pyvirtualcam` backend is available on your system

If the backend is missing, the app exits with a diagnostic message before streaming starts.

## How It Works

1. Capture frames from the webcam with `cv2.VideoCapture` (or load a photo from disk)
2. Downscale and map pixels to an ASCII gradient
3. Print the ASCII matrix directly into the terminal
4. Draw the same ASCII onto a PIL canvas and stream it to the virtual camera (live mode only)
