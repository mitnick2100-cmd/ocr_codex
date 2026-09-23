#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8n + IMX519 + Arducam focuser example.

Usage:
    python3 yolo.py
    python3 yolo.py --model ../yolov8n.pt --device 0
"""

import argparse
import os
import time
from pathlib import Path
from queue import Empty

import cv2
from ultralytics import YOLO

from Autofocus import AutoFocus
from Focuser import Focuser
from JetsonCamera import Camera


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FRAMERATE = 17
DEFAULT_IMGSZ = (736, 1280)  # (height, width)

# IMX519 感測器可用模式；15 FPS 使用 3840x2160 模式並由 Argus 降速輸出。
FPS_OPTIONS = (9, 15, 17, 60, 120)
SENSOR_MODES = {
    9: (4656, 3496),
    15: (3840, 2160),
    17: (3840, 2160),
    60: (1920, 1080),
    120: (1280, 720),
}

EXPOSURE_STEP_MS = 1.0
EXPOSURE_MIN_MS = 0.1
EXPOSURE_MAX_RATIO = 0.95
EXPOSURE_SENSOR_MAX_MS = 683.709

FOCUS_STEP = 50
FOCUS_SETTLE_TIME = 0.05

KEY_LEFT = (2424832, 65361, 81)
KEY_UP = (2490368, 65362, 82)
KEY_RIGHT = (2555904, 65363, 83)
KEY_DOWN = (2621440, 65364, 84)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8n IMX519 camera + focuser example.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="IMX519 display width")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="IMX519 display height")
    parser.add_argument(
        "--framerate",
        type=int,
        choices=FPS_OPTIONS,
        default=DEFAULT_FRAMERATE,
        help="IMX519 FPS. Supported: {}. Default: {}".format(
            ", ".join(map(str, FPS_OPTIONS)), DEFAULT_FRAMERATE
        ),
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 for Jetson GPU or cpu")
    parser.add_argument("-i", "--i2c-bus", type=int, default=10, help="Focuser I2C bus. Default: 10")
    parser.add_argument("--exposure", type=float, default=None, help="Initial exposure time in ms")
    parser.add_argument("--gain", default="1 1", help='Fixed gain range, e.g. "1 1"; set "" for auto gain')
    parser.add_argument("--snapshot-dir", default="snapshots", help="Directory for images saved by C")
    return parser.parse_args()


def resolve_model_path(model):
    requested = Path(model)
    if requested.exists() or requested.is_absolute():
        return str(requested)

    script_dir = Path(__file__).resolve().parent
    project_root_model = script_dir.parent / model
    if project_root_model.exists():
        return str(project_root_model)

    return model


def max_exposure_ms(framerate):
    if framerate <= 0:
        raise ValueError("framerate must be greater than zero")
    frame_limit_ms = (1000.0 / framerate) * EXPOSURE_MAX_RATIO
    return min(frame_limit_ms, EXPOSURE_SENSOR_MAX_MS)


def sensor_mode_for_fps(framerate):
    try:
        return SENSOR_MODES[framerate]
    except KeyError as exc:
        raise ValueError(
            "Unsupported FPS {}. Choose one of {}".format(
                framerate, ", ".join(map(str, FPS_OPTIONS))
            )
        ) from exc


def clamp_exposure_ms(exposure_ms, framerate):
    if exposure_ms is None:
        return None
    return min(max(float(exposure_ms), EXPOSURE_MIN_MS),
               max_exposure_ms(framerate))


def reopen_camera_with_exposure(camera, cam_state, focuser=None):
    exposure_ms = cam_state["exposure_ms"]
    exposure_ns = None if exposure_ms is None else int(exposure_ms * 1_000_000)

    camera.reconfigure(
        framerate=cam_state["framerate"],
        exposure_time_ns=exposure_ns,
        gain_range=cam_state["gain_range"],
        ae_lock=exposure_ms is not None,
        capture_width=cam_state["capture_width"],
        capture_height=cam_state["capture_height"],
    )

    if focuser is not None:
        focuser.reinit()


def is_arrow_key(key):
    return (
        key in KEY_LEFT
        or key in KEY_UP
        or key in KEY_RIGHT
        or key in KEY_DOWN
    )


def draw_text(frame, text, origin, color=(255, 255, 255), scale=0.65):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def draw_overlay(frame, elapsed_ms, cam_state, focuser, auto_focus, last_key):
    fps = 1000.0 / max(elapsed_ms, 1e-3)
    exposure_ms = cam_state["exposure_ms"]
    exposure_text = "Exposure: Auto"
    if exposure_ms is not None:
        exposure_text = "Exposure: {:.2f} ms (~1/{:.0f}s)".format(
            exposure_ms,
            1000.0 / exposure_ms,
        )

    if auto_focus.is_finished():
        autofocus_text = "AutoFocus: Idle (best={})".format(auto_focus.best_position)
    else:
        autofocus_text = "AutoFocus: Running"

    lines = [
        "YOLOv8n {:.1f} ms {:.1f} FPS".format(elapsed_ms, fps),
        "Focus: {}".format(focuser.get(Focuser.OPT_FOCUS)),
        autofocus_text,
        exposure_text,
        "Camera FPS: {} ({}x{})".format(
            cam_state["framerate"],
            cam_state["capture_width"],
            cam_state["capture_height"],
        ),
        "Last key: {}".format(last_key) if last_key >= 0 else "Last key: None",
        "Q/Esc exit | C snapshot | A auto exposure | [/] shutter | -/= FPS",
        "Up/Down or W/S focus | Enter/F autofocus | R reset focus",
    ]

    for index, line in enumerate(lines):
        color = (0, 255, 0) if index == 0 else (255, 255, 255)
        draw_text(frame, line, (20, 35 + index * 30), color=color)


def change_framerate(camera, cam_state, focuser, direction):
    current_index = FPS_OPTIONS.index(cam_state["framerate"])
    new_index = max(0, min(len(FPS_OPTIONS) - 1, current_index + direction))
    new_fps = FPS_OPTIONS[new_index]
    if new_fps == cam_state["framerate"]:
        return

    capture_width, capture_height = sensor_mode_for_fps(new_fps)
    cam_state["framerate"] = new_fps
    cam_state["capture_width"] = capture_width
    cam_state["capture_height"] = capture_height
    cam_state["exposure_ms"] = clamp_exposure_ms(
        cam_state["exposure_ms"], new_fps
    )
    reopen_camera_with_exposure(camera, cam_state, focuser)


def handle_key(key, camera, cam_state, focuser, auto_focus, snapshot_state, frame, snapshot_dir):
    key8 = key & 0xFF

    if key8 in (ord("-"), ord("_")):
        change_framerate(camera, cam_state, focuser, -1)
    elif key8 in (ord("="), ord("+")):
        change_framerate(camera, cam_state, focuser, 1)
    elif key in KEY_RIGHT or key8 == ord("]"):
        current = cam_state["exposure_ms"] if cam_state["exposure_ms"] is not None else 0.0
        cam_state["exposure_ms"] = min(
            current + EXPOSURE_STEP_MS,
            max_exposure_ms(cam_state["framerate"]),
        )
        reopen_camera_with_exposure(camera, cam_state, focuser)
    elif key in KEY_LEFT or key8 == ord("["):
        if cam_state["exposure_ms"] is None:
            cam_state["exposure_ms"] = max_exposure_ms(cam_state["framerate"])
        cam_state["exposure_ms"] = max(
            cam_state["exposure_ms"] - EXPOSURE_STEP_MS,
            EXPOSURE_MIN_MS,
        )
        reopen_camera_with_exposure(camera, cam_state, focuser)
    elif key in KEY_UP or key8 in (ord("w"), ord("W")):
        focuser.set(Focuser.OPT_FOCUS, focuser.get(Focuser.OPT_FOCUS) + FOCUS_STEP)
    elif key in KEY_DOWN or key8 in (ord("s"), ord("S")):
        focuser.set(Focuser.OPT_FOCUS, focuser.get(Focuser.OPT_FOCUS) - FOCUS_STEP)
    elif key8 in (ord("r"), ord("R")):
        focuser.reset(Focuser.OPT_FOCUS)
    elif key8 in (10, 13) or key in (65421, ord("f"), ord("F")):
        auto_focus.start_focus()
    elif key8 in (ord("c"), ord("C")):
        os.makedirs(snapshot_dir, exist_ok=True)
        filename = os.path.join(
            snapshot_dir,
            "image{:04d}.jpg".format(snapshot_state["count"]),
        )
        cv2.imwrite(filename, frame)
        snapshot_state["count"] += 1
        print("Saved image:", filename)
    elif key8 in (ord("a"), ord("A")):
        cam_state["exposure_ms"] = None
        reopen_camera_with_exposure(camera, cam_state, focuser)


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model)
    gain_range = args.gain if args.gain else None
    capture_width, capture_height = sensor_mode_for_fps(args.framerate)
    exposure_ms = clamp_exposure_ms(args.exposure, args.framerate)
    if args.exposure is not None and exposure_ms != args.exposure:
        print("Exposure clamped from {:.3f} ms to {:.3f} ms for {} FPS".format(
            args.exposure, exposure_ms, args.framerate
        ))
    exposure_ns = int(exposure_ms * 1_000_000) if exposure_ms is not None else None
    cam_state = {
        "framerate": args.framerate,
        "exposure_ms": exposure_ms,
        "gain_range": gain_range,
        "capture_width": capture_width,
        "capture_height": capture_height,
    }

    print("Opening IMX519 camera: {}x{} @ {} FPS (sensor {}x{})".format(
        args.width,
        args.height,
        args.framerate,
        capture_width,
        capture_height,
    ))
    camera = Camera(
        width=args.width,
        height=args.height,
        framerate=args.framerate,
        exposure_time_ns=exposure_ns,
        gain_range=gain_range,
        ae_lock=exposure_ms is not None,
        capture_width=capture_width,
        capture_height=capture_height,
    )

    print("Using I2C bus:", args.i2c_bus)
    focuser = Focuser(args.i2c_bus)
    auto_focus = AutoFocus(
        focuser,
        camera,
        roi=(0.4, 0.4, 0.2, 0.2),
        focus_step=FOCUS_STEP,
        move_settle_time=FOCUS_SETTLE_TIME,
        decline_patience=5,
        verbose=False,
    )

    print("Loading YOLO model:", model_path)
    model = YOLO(model_path)

    window_name = "YOLOv8n IMX519"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)
    last_key = -1
    snapshot_state = {"count": 0}

    try:
        while True:
            try:
                frame = camera.getFrame(timeout=1.0)
            except Empty:
                print("Waiting for camera frame...")
                continue

            if frame is None:
                continue

            start = time.time()
            results = model.predict(
                source=frame,
                imgsz=DEFAULT_IMGSZ,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            elapsed_ms = (time.time() - start) * 1000.0

            annotated = results[0].plot()
            draw_overlay(annotated, elapsed_ms, cam_state, focuser, auto_focus, last_key)

            cv2.imshow(window_name, annotated)
            key = cv2.waitKeyEx(1)
            if key == -1:
                continue

            last_key = key
            if key == 27 or ((key & 0xFF) in (ord("q"), ord("Q")) and not is_arrow_key(key)):
                break

            handle_key(
                key,
                camera,
                cam_state,
                focuser,
                auto_focus,
                snapshot_state,
                annotated,
                args.snapshot_dir,
            )
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
