#!/usr/bin/env python3
"""IMX519 / Jetson automatic refocus experiment; no focus key required.

Run beside JetsonCamera.py and Focuser.py. Fixed central ROI, not object
tracking. Motion gating is a heuristic, not a motion/defocus classifier.
"""
import argparse
from collections import deque
from queue import Empty
from statistics import median
import time


class FocusController:
    """Single-threaded motor owner; UI and sampling share one frame stream."""

    def __init__(self, move, position, args):
        self.move = move
        self.position = position
        self.args = args
        self.turbo = getattr(args, "af_speed", "fast") == "turbo"
        self.fast = getattr(args, "af_speed", "fast") != "safe"
        self.guard_frames = 2 if self.fast else 3
        self.sample_count = 2 if self.fast else 3
        self.coarse_step = 50 if self.turbo or not self.fast else 100
        self.local_radius = 100 if self.fast else 150
        self.last_duration = None
        self.move_count = 0
        self.reference = None
        self.state = "WARMUP"
        self.ready_at = time.monotonic() + 2.0
        self.stable_since = None
        self.low_since = None
        self.history = deque(maxlen=5)
        self.samples = []
        self.results = {}
        self.active = False
        self.attempts = 0
        self.reason = "Waiting for camera"

    def go(self, position, now):
        self.position = max(0, min(1000, int(position)))
        command_started = time.monotonic()
        self.move(self.position)
        command_duration = time.monotonic() - command_started
        self.move_count += 1
        # Wait after I2C completion. Arrival timing is not an exposure timestamp.
        self.ready_at = now + command_duration + max(
            self.args.settle, self.guard_frames / self.args.framerate)
        self.samples = []

    def scan(self, points, state, now):
        self.pending = iter(sorted(set(points)))
        self.results = {}
        self.state = state
        self.go(next(self.pending), now)

    def start(self, now, score):
        self.active = True
        self.origin = self.position
        self.before = score
        self.started = now
        self.move_count = 0
        self.attempts += 1
        self.full = self.reference is None
        self.reason = "Automatic scan #{}".format(self.attempts)
        if self.turbo and not self.full:
            self.scan([self.origin, max(0, self.origin - 40),
                       min(1000, self.origin + 40)], "PROBE", now)
            print(self.reason, flush=True)
            return
        if self.full:
            self.search_step = self.coarse_step
            points = range(0, 1001, self.search_step)
        else:
            self.search_step = 50
            lo = max(0, self.origin - self.local_radius)
            hi = min(1000, self.origin + self.local_radius)
            points = list(range(lo, hi + 1, 50)) + [hi, self.origin]
        self.scan(points, "COARSE", now)
        print(self.reason, flush=True)

    def full_scan(self, now):
        self.full = True
        self.search_step = self.coarse_step
        self.scan(range(0, 1001, self.search_step), "COARSE", now)

    def refine(self, best, steps, now, radius=None):
        self.refine_steps = list(steps)
        step = self.refine_steps.pop(0)
        radius = step if radius is None else radius
        points = list(range(max(0, best - radius), min(1000, best + radius) + 1, step))
        self.scan(points + [best], "REFINE", now)

    def finish(self, now, success, score=0):
        self.last_duration = now - self.started
        if success:
            self.reference = score
            self.reason = "Focus verified"
        else:
            self.go(self.origin, now)
            self.reason = "No reliable improvement; restored previous position"
        self.active = False
        self.state = "COOLDOWN"
        self.ready_at = now + (self.args.cooldown if success else max(5, self.args.cooldown))
        self.stable_since = self.low_since = None
        self.history.clear()
        print("{}: position={}, score={:.1f}, elapsed={:.2f}s, moves={}".format(
            self.reason, self.position, score, self.last_duration, self.move_count), flush=True)

    def update(self, now, score, motion, valid):
        if self.active and now - self.started > 60:
            self.finish(now, False)
            return
        if now < self.ready_at:
            return
        if not valid or motion:
            self.stable_since = self.low_since = None
            self.history.clear()
            self.samples = []
            if not self.active:
                self.state = "HOLD"
                self.reason = "Motion / dark / saturated; waiting"
            return
        if self.active:
            self.samples.append(score)
            required = 3 if self.state == "VERIFY" else self.sample_count
            if len(self.samples) < required:
                return
            value = median(self.samples)
            if self.state == "VERIFY":
                # Flat scenes must not be declared a successful focus lock.
                improved = self.reference is None or value > self.before * 1.08
                if self.turbo and not self.full and not (
                        value >= self.args.min_sharpness and value >= self.peak * 0.8 and improved):
                    self.full_scan(now)
                    return
                self.finish(now, value >= self.args.min_sharpness and
                            value >= self.peak * 0.8 and improved, value)
                return
            self.results[self.position] = value
            if self.state == "WALK":
                # Walk in 40-unit increments until improvement stops; no full sweep.
                self.walk_steps += 1
                if value > self.walk_peak * 1.03:
                    self.walk_best, self.walk_peak = self.position, value
                    nxt = self.position + self.direction * 40
                    if 0 <= nxt <= 1000:
                        if self.walk_steps >= 4:
                            self.full_scan(now)
                            return
                        self.go(nxt, now)
                        return
                self.refine(self.walk_best, [20, 10], now)
                return
            nxt = next(self.pending, None)
            if nxt is not None:
                self.go(nxt, now)
                return
            best = max(self.results, key=self.results.get)
            peak = self.results[best]
            if self.state == "PROBE":
                if peak < max(self.args.min_sharpness, self.before * 1.08):
                    self.full_scan(now)
                elif best == self.origin:
                    self.refine(best, [10], now, radius=50)
                else:
                    self.direction = 1 if best > self.origin else -1
                    self.walk_steps = 0
                    self.walk_best, self.walk_peak = best, peak
                    nxt = best + self.direction * 40
                    if 0 <= nxt <= 1000:
                        self.state = "WALK"
                        self.go(nxt, now)
                    else:
                        self.refine(best, [10], now, radius=50)
                return
            if self.state == "REFINE":
                if self.refine_steps:
                    self.refine(best, self.refine_steps, now)
                else:
                    self.peak = peak
                    self.state = "VERIFY"
                    self.go(best, now)
                return
            if self.state == "COARSE":
                if self.full and peak < max(self.args.min_sharpness,
                                            min(self.results.values()) * 1.1):
                    self.finish(now, False)
                    return
                boundary = best in (min(self.results), max(self.results))
                if not self.full and (boundary or peak < self.before * 1.08):
                    self.full_scan(now)
                elif self.turbo:
                    self.refine(best, [10], now, radius=50)
                else:
                    step = 25 if self.fast else 10
                    self.scan(range(max(0, best - self.search_step),
                                    min(1000, best + self.search_step) + 1, step),
                              "FINE", now)
            elif self.state == "FINE" and self.fast:
                # Keep the measured best candidate, including near travel limits.
                points = list(range(max(0, best - 20), min(1000, best + 20) + 1, 10))
                self.scan(points + [best], "POLISH", now)
            else:
                self.peak = peak
                self.state = "VERIFY"
                self.go(best, now)
            return
        self.state = "MONITOR"
        self.reason = "Monitoring central ROI"
        if self.stable_since is None:
            self.stable_since = now
        self.history.append(score)
        if len(self.history) < 5 or now - self.stable_since < 0.5:
            return
        smoothed = median(self.history)
        if self.reference is None:
            self.start(now, smoothed)
        elif smoothed < self.reference * self.args.drop_ratio:
            if self.low_since is None:
                self.low_since = now
            elif now - self.low_since >= self.args.trigger_delay:
                self.start(now, smoothed)
        else:
            self.low_since = None


def main():
    import cv2
    import numpy as np
    from JetsonCamera import Camera
    from Focuser import Focuser

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i2c-bus", type=int, default=10)
    parser.add_argument("--framerate", type=int, choices=(9, 15, 17, 60, 120), default=17)
    parser.add_argument("--exposure", type=float, default=None, help="Fixed exposure in ms")
    parser.add_argument("--gain", default="1 1", help='Gain range; use "" for automatic')
    parser.add_argument("--roi-size", type=float, default=0.3, help="Central ROI fraction")
    parser.add_argument("--drop-ratio", type=float, default=0.7)
    parser.add_argument("--trigger-delay", type=float, default=0.4)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--af-speed", choices=("turbo", "fast", "safe"), default="turbo",
                        help="turbo: directional search; fast: hierarchical search; safe: original scan")
    parser.add_argument("--settle", type=float, default=None,
                        help="Motor settle seconds; default fast=0.06, safe=0.15")
    parser.add_argument("--motion-threshold", type=float, default=25.0,
                        help="ROI optical flow speed limit, analysis pixels/second")
    parser.add_argument("--min-sharpness", type=float, default=20.0,
                        help="Minimum verified Tenengrad score; scene dependent")
    args = parser.parse_args()
    if args.settle is None:
        args.settle = 0.15 if args.af_speed == "safe" else 0.06
    if not 0.05 <= args.roi_size <= 0.9 or not 0 < args.drop_ratio < 1:
        parser.error("roi-size must be 0.05..0.9; drop-ratio must be 0..1")
    if min(args.trigger_delay, args.cooldown, args.settle,
           args.motion_threshold, args.min_sharpness) <= 0:
        parser.error("Timing and quality thresholds must be positive")
    if args.exposure is not None and not 0 < args.exposure <= 950.0 / args.framerate:
        parser.error("Exposure must be positive and <= 95% of frame interval")
    modes = {9: (4656, 3496), 15: (3840, 2160), 17: (3840, 2160),
             60: (1920, 1080), 120: (1280, 720)}
    width, height = modes[args.framerate]
    camera = Camera(width=1280, height=720, framerate=args.framerate,
                    capture_width=width, capture_height=height,
                    exposure_time_ns=None if args.exposure is None else int(args.exposure * 1e6),
                    gain_range=args.gain or None, ae_lock=args.exposure is not None)
    try:
        focuser = Focuser(args.i2c_bus)
        controller = FocusController(lambda p: focuser.set(Focuser.OPT_FOCUS, p),
                                     focuser.get(Focuser.OPT_FOCUS), args)
        previous = previous_time = None
        last_frame = time.monotonic()
        cv2.namedWindow("Automatic AF test", cv2.WINDOW_NORMAL)
        while True:
            try:
                frame = camera.getFrame(timeout=1.0)
            except Empty:
                if time.monotonic() - last_frame > 5:
                    raise RuntimeError("No camera frames for five seconds")
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
                continue
            if frame is None:
                raise RuntimeError("Camera returned an empty frame")
            now = time.monotonic()
            last_frame = now
            h, w = frame.shape[:2]
            rw, rh = int(w * args.roi_size), int(h * args.roi_size)
            x, y = (w - rw) // 2, (h - rh) // 2
            gray = cv2.cvtColor(frame[y:y+rh, x:x+rw], cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            sharpness = float(np.mean(gx * gx + gy * gy))
            speed = 0.0
            flow_known = False
            if previous is not None:
                points = cv2.goodFeaturesToTrack(previous, 80, 0.01, 7)
                if points is not None and len(points) >= 8:
                    tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
                    if tracked is not None:
                        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, tracked, None)
                        if back is not None:
                            good = ((status.ravel() == 1) & (back_status.ravel() == 1) &
                                    (np.linalg.norm(back - points, axis=2).ravel() < 1.5))
                            if good.sum() >= 8:
                                displacement = np.linalg.norm(tracked[good] - points[good], axis=2)
                                speed = float(np.median(displacement)) / max(now - previous_time, 0.001)
                                flow_known = True
            # Fallback permits startup on a defocused scene; it is NOT proof of no motion.
            difference = 0 if previous is None else float(np.mean(cv2.absdiff(previous, gray)))
            motion = speed > args.motion_threshold if flow_known else difference > 10
            previous, previous_time = gray, now
            valid = 10 < float(gray.mean()) < 245 and float(np.mean(gray >= 250)) < 0.3
            controller.update(now, sharpness, motion, valid)
            display = frame.copy()  # Never draw into a camera-owned AF image.
            cv2.rectangle(display, (x, y), (x+rw, y+rh), (0, 255, 255), 2)
            ref = controller.reference
            duration = "--" if controller.last_duration is None else "{:.2f}s".format(controller.last_duration)
            lines = ["{} | focus={} | AF count={} | {} | last={}".format(
                         controller.state, controller.position, controller.attempts, args.af_speed, duration),
                     "Sharpness={:.1f} | reference={}".format(sharpness, "none" if ref is None else "{:.1f}".format(ref)),
                     "Motion={} | speed={:.1f} px/s | flow={}".format(motion, speed, "OK" if flow_known else "UNKNOWN"),
                     controller.reason, "Keep target inside yellow box | Q / Esc: exit"]
            for i, text in enumerate(lines):
                cv2.putText(display, text, (15, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display, text, (15, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Automatic AF test", display)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
