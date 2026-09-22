#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect day/night from a video file by average pixel brightness.

Usage examples:
    python test.py --video ../bus_image/bus_1.mp4
    python test.py --video ../bus_image/bus_1.mp4 --threshold 75
    python test.py --video ../bus_image/bus_1.mp4 --no-show
    python test.py --video ../bus_image/bus_1.mp4 --samples 10 --interval 0.5
"""

import argparse
import time

import cv2


DEFAULT_THRESHOLD = 35.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect day/night from video average pixel brightness."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Brightness threshold. >= threshold is day, below is night.")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of frames averaged for each decision")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between each printed day/night decision")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not show video preview window")
    parser.add_argument("--loop", action="store_true",
                        help="Loop video when it reaches the end")
    return parser.parse_args()


def average_brightness(frame):
    """Return the mean grayscale brightness in the 0-255 range."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def classify_day_night(brightness, threshold):
    return "day" if brightness >= threshold else "night"


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


def read_average_brightness(cap, samples, loop):
    values = []
    last_frame = None

    while len(values) < samples:
        ok, frame = cap.read()

        if not ok or frame is None:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        last_frame = frame
        values.append(average_brightness(frame))

    if not values:
        return None, None

    return sum(values) / len(values), last_frame


def main():
    args = parse_args()
    samples = max(1, args.samples)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video: {}".format(args.video))

    print("Video day/night detector started")
    print("video: {}".format(args.video))
    print("brightness >= {:.1f}: day, brightness < {:.1f}: night".format(
        args.threshold,
        args.threshold,
    ))

    try:
        while True:
            brightness, frame = read_average_brightness(cap, samples, args.loop)
            if brightness is None:
                print("End of video")
                break

            state = classify_day_night(brightness, args.threshold)
            print("average_brightness={:.1f}, result={}".format(brightness, state))

            if not args.no_show and frame is not None:
                preview = frame.copy()
                draw_overlay(preview, brightness, state, args.threshold)
                cv2.imshow("Video Day/Night Detector", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break

            time.sleep(max(0.0, args.interval))
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
