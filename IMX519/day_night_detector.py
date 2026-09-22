#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use the IMX519 camera input to classify the scene as day or night by average
pixel brightness.

Usage examples:
    python3 day_night_detector.py
    python3 day_night_detector.py --threshold 75
    python3 day_night_detector.py --no-show
    python3 day_night_detector.py --samples 10 --interval 2.0
"""

import argparse
import time
from queue import Empty

import cv2

from JetsonCamera import Camera


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FRAMERATE = 15
DEFAULT_THRESHOLD = 80.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect day/night from IMX519 average pixel brightness."
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                        help="IMX519 display width")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                        help="IMX519 display height")
    parser.add_argument("--framerate", type=int, default=DEFAULT_FRAMERATE,
                        help="Camera framerate")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Brightness threshold. >= threshold is day, below is night.")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of frames averaged for each decision")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between each day/night decision")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not show camera preview window")
    parser.add_argument("--once", action="store_true",
                        help="Print one decision and exit")
    return parser.parse_args()


def average_brightness(frame):
    """Return the mean grayscale brightness in the 0-255 range."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def classify_day_night(brightness, threshold):
    return "day" if brightness >= threshold else "night"


def read_average_brightness(camera, samples):
    values = []
    last_frame = None

    while len(values) < samples:
        try:
            frame = camera.getFrame(timeout=2.0)
        except Empty:
            print("Warning: timeout while waiting for IMX519 frame")
            continue

        if frame is None:
            continue

        last_frame = frame
        values.append(average_brightness(frame))

    return sum(values) / len(values), last_frame


def draw_overlay(frame, brightness, state, threshold):
    color = (0, 255, 255) if state == "day" else (255, 120, 0)
    label = "DAY" if state == "day" else "NIGHT"
    lines = [
        "Brightness: {:.1f}".format(brightness),
        "Threshold: {:.1f}".format(threshold),
        "Result: {}".format(label),
        "Press Q or ESC to quit",
    ]

    y = 32
    for line in lines:
        cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)
        y += 34


def main():
    args = parse_args()
    samples = max(1, args.samples)

    camera = Camera(
        width=args.width,
        height=args.height,
        framerate=args.framerate,
    )

    print("IMX519 day/night detector started")
    print("brightness >= {:.1f}: day, brightness < {:.1f}: night".format(
        args.threshold,
        args.threshold,
    ))

    try:
        while True:
            brightness, frame = read_average_brightness(camera, samples)
            state = classify_day_night(brightness, args.threshold)
            print("average_brightness={:.1f}, result={}".format(brightness, state))

            if not args.no_show and frame is not None:
                preview = frame.copy()
                draw_overlay(preview, brightness, state, args.threshold)
                cv2.imshow("IMX519 Day/Night Detector", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break

            if args.once:
                break

            time.sleep(max(0.0, args.interval))
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        camera.close()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
