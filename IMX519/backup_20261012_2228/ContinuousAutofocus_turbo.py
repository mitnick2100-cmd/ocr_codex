#!/usr/bin/env python3
"""IMX519 / Jetson continuous autofocus (turbo-only, no fine scan).

Run beside JetsonCamera.py and Focuser.py. Fixed central ROI, not object
tracking. Motion gating is a heuristic, not a motion/defocus classifier.

Trigger: the smoothed ROI score drops below reference x drop-ratio, or rises
above reference x rise-ratio (new / more textured target). In NO TARGET
(reference < min-sharpness) only a rise to min-sharpness or to
reference x NO_TARGET_RISE triggers. A rise-triggered scan that finds no
clearly better side accepts the (interpolated) current position as focused
instead of failing, so the lens does not wobble on lighting changes.

Turbo search:  PROBE (origin score reused from MONITOR, measure +/-STEP)
-> WALK toward the better side until the score stops improving -> interpolate
the peak from the best point and its two measured neighbours (no extra moves)
-> VERIFY.

If PROBE sees only blur (all scores < min-sharpness: lens far from focus, or a
blank scene), one coarse RESCUE sweep per trigger finds the region, then a
LOCAL +/-STEP probe hands over to WALK/interpolation. A RESCUE that finds no
texture puts the controller straight into NO TARGET (reference = current
score) so it waits for a new target instead of retrying.
"""
import argparse
from collections import deque
import math
from queue import Empty
from statistics import median
import time

FOCUS_MIN, FOCUS_MAX = 0, 1000
STEP = 40                # PROBE offset and WALK increment
WALK_GAIN = 1.03         # keep walking while score improves by > 3 %
IMPROVE_RATIO = 1.08     # a new position must beat the trigger score by 8 %
VERIFY_KEEP = 0.8        # verified score must keep 80 % of the best measured score
RESCUE_STEP = 100        # coarse sweep step when PROBE sees only blur
GUARD_FRAMES = 2         # keep 2 until the camera pipeline latency is measured
SAMPLES = 1              # frames averaged per search position
VERIFY_SAMPLES = 2
HISTORY_LEN = 3          # MONITOR median window (frames)
STABLE_TIME = 0.2        # MONITOR needs this long of stable frames before judging
SCAN_TIMEOUT = 60.0
MAX_FAILS = 3            # consecutive failures before re-baselining
NO_TARGET_RISE = 3.0     # in NO TARGET, score must triple to trigger (noise guard)


def clamp(p):
    return max(FOCUS_MIN, min(FOCUS_MAX, int(round(p))))


def interpolate_peak(results, best, step):
    """Gaussian (log-parabola) fit through best and its +/-step neighbours.

    Uses scores already measured, so it costs no lens movement. Falls back
    to `best` when a neighbour is missing or the curve is not a peak.
    """
    left, right = results.get(best - step), results.get(best + step)
    centre = results[best]
    if left is None or right is None or min(left, centre, right) <= 0:
        return best
    l, c, r = math.log(left), math.log(centre), math.log(right)
    denom = l - 2 * c + r
    if denom >= 0:
        return best
    offset = 0.5 * step * (l - r) / denom
    return clamp(best + max(-step / 2, min(step / 2, offset)))


class FocusController:
    """Single-threaded motor owner; UI and sampling share one frame stream."""

    def __init__(self, move, position, args):
        self.move = move
        self.position = position
        self.args = args
        self.last_duration = None
        self.move_count = 0
        self.reference = None
        self.state = "WARMUP"
        self.ready_at = time.monotonic() + 2.0
        self.stable_since = None
        self.low_since = None
        self.history = deque(maxlen=HISTORY_LEN)
        self.samples = []
        self.results = {}
        self.pending = iter(())
        self.active = False
        self.attempts = 0
        self.fails = 0
        self.rescued = False
        self.cause = "init"      # what triggered the current scan: init / drop / rise
        self.rescue_center = position
        self.origin = position
        self.before = 0.0
        self.started = 0.0
        self.peak = 0.0
        self.direction = 1
        self.walk_best = position
        self.walk_peak = 0.0
        self.reason = "Waiting for camera"

    # ---- motor / scan helpers -------------------------------------------
    def go(self, position, now):
        self.position = clamp(position)
        command_started = time.monotonic()
        self.move(self.position)
        command_duration = time.monotonic() - command_started
        self.move_count += 1
        # Wait after I2C completion. Arrival timing is not an exposure timestamp.
        self.ready_at = now + command_duration + max(
            self.args.settle, GUARD_FRAMES / self.args.framerate)
        self.samples = []

    def scan(self, points, state, now, known=None):
        """Visit `points`; positions already in `known` are not re-measured."""
        self.results = dict(known or {})
        self.pending = iter(sorted({clamp(p) for p in points} - set(self.results)))
        self.state = state
        self.go(next(self.pending), now)

    def lock(self, best, step, now):
        """Jump straight to the interpolated peak and verify it."""
        self.peak = self.results[best]
        self.state = "VERIFY"
        self.go(interpolate_peak(self.results, best, step), now)

    # ---- lifecycle ------------------------------------------------------
    def start(self, now, score, cause="init"):
        self.cause = cause
        self.active = True
        self.origin = self.position
        self.before = score
        self.started = now
        self.move_count = 0
        self.attempts += 1
        self.rescued = False
        self.reason = "Automatic scan #{}".format(self.attempts)
        # The origin was just measured by MONITOR (median), so only the two
        # side points cost a lens move.
        self.scan([self.origin - STEP, self.origin + STEP], "PROBE", now,
                  known={self.origin: score})
        print(self.reason, flush=True)

    def rescue(self, now):
        """PROBE saw only blur: lens is far from focus (or the scene is blank)."""
        self.rescued = True
        self.reason = "Rescue sweep (scan #{})".format(self.attempts)
        self.scan(range(FOCUS_MIN, FOCUS_MAX + 1, RESCUE_STEP), "RESCUE", now)
        print(self.reason, flush=True)

    def finish(self, now, success, score=0.0, no_target=False):
        self.last_duration = now - self.started
        if success:
            self.reference = score
            self.fails = 0
            self.reason = "Focus verified"
        else:
            self.go(self.origin, now)
            self.fails += 1
            self.reason = "No reliable improvement; restored previous position"
            # A full sweep found no texture, or repeated failures: accept the
            # current level as the new baseline (NO TARGET if it is below
            # min-sharpness) and wait for a drop/rise instead of retrying.
            # Also applies before the first successful lock (reference None).
            if no_target or self.fails >= MAX_FAILS:
                self.reference = self.before
                self.fails = 0
                self.reason += " (no target)" if no_target else " (re-baselined)"
        self.active = False
        self.state = "COOLDOWN"
        self.ready_at = now + self.args.cooldown
        self.stable_since = self.low_since = None
        self.history.clear()
        print("{}: position={}, score={:.1f}, elapsed={:.2f}s, moves={}".format(
            self.reason, self.position, score, self.last_duration, self.move_count), flush=True)

    # ---- per-frame update -----------------------------------------------
    def update(self, now, score, motion, valid):
        if self.active and now - self.started > SCAN_TIMEOUT:
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
            self._step(now, score)
        else:
            self._monitor(now, score)

    def _step(self, now, score):
        self.samples.append(score)
        required = VERIFY_SAMPLES if self.state == "VERIFY" else SAMPLES
        if len(self.samples) < required:
            return
        value = median(self.samples)

        if self.state == "VERIFY":
            # Flat scenes must not be declared a successful focus lock.
            # A rise trigger already means the scene got sharper; no further
            # improvement over the trigger score is required.
            improved = (self.reference is None or self.cause == "rise" or
                        value > self.before * IMPROVE_RATIO)
            self.finish(now, value >= self.args.min_sharpness and
                        value >= self.peak * VERIFY_KEEP and improved, value)
            return

        self.results[self.position] = value

        if self.state == "WALK":
            if value > self.walk_peak * WALK_GAIN:
                self.walk_best, self.walk_peak = self.position, value
                nxt = self.position + self.direction * STEP
                if FOCUS_MIN <= nxt <= FOCUS_MAX:
                    self.go(nxt, now)
                    return
            # Passed the peak (or hit the travel limit): it lies near walk_best.
            self.lock(self.walk_best, STEP, now)
            return

        nxt = next(self.pending, None)
        if nxt is not None:
            self.go(nxt, now)
            return
        best = max(self.results, key=self.results.get)
        peak = self.results[best]

        if self.state == "PROBE":
            # First lock: any usable sharpness is enough (nothing to beat yet).
            need = self.args.min_sharpness if self.reference is None else \
                max(self.args.min_sharpness, self.before * IMPROVE_RATIO)
            if peak < need:
                if peak < self.args.min_sharpness and not self.rescued:
                    # All three points are blur: no direction to follow.
                    self.rescue(now)
                elif self.cause == "rise" and peak >= self.args.min_sharpness:
                    # Score rose but no side is clearly better: the lens is
                    # already near the peak for the new scene. Interpolate,
                    # verify and adopt it as the new reference.
                    self.lock(self.origin, STEP, now)
                else:
                    self.finish(now, False, peak)
            else:
                self._climb(self.origin, best, peak, now)
            return

        if self.state == "RESCUE":
            if peak < max(self.args.min_sharpness, min(self.results.values()) * 1.1):
                self.finish(now, False, peak, no_target=True)
            else:
                # Peak is within +/-RESCUE_STEP/2 of best: probe +/-STEP there.
                self.rescue_center = best
                self.scan([best - STEP, best + STEP], "LOCAL", now, known={best: peak})
            return

        if self.state == "LOCAL":
            self._climb(self.rescue_center, best, peak, now)

    def _climb(self, center, best, peak, now):
        """Lock at `center` if it is best, else WALK toward the better side."""
        if best == center:
            self.lock(best, STEP, now)
            return
        self.direction = 1 if best > center else -1
        self.walk_best, self.walk_peak = best, peak
        nxt = best + self.direction * STEP
        if FOCUS_MIN <= nxt <= FOCUS_MAX:
            self.state = "WALK"
            self.go(nxt, now)
        else:
            self.lock(best, STEP, now)

    def _monitor(self, now, score):
        self.state = "MONITOR"
        self.reason = "Monitoring central ROI"
        if self.stable_since is None:
            self.stable_since = now
        self.history.append(score)
        if len(self.history) < self.history.maxlen or now - self.stable_since < STABLE_TIME:
            return
        smoothed = median(self.history)
        ref = self.reference
        if ref is None:
            self.start(now, smoothed)
            return
        if ref < self.args.min_sharpness:
            self.reason = "No target texture in ROI; waiting for a new target"
        # Drop: the locked target went out of focus. Disabled in NO TARGET:
        # a drop from a textureless baseline is just noise and would only
        # cause repeated rescue sweeps.
        drop = ref >= self.args.min_sharpness and smoothed < ref * self.args.drop_ratio
        # Rise: a new textured target appeared. Without this, a low reference
        # (e.g. re-baselined on a blank wall) would never trigger again.
        if ref < self.args.min_sharpness:
            rise = smoothed >= self.args.min_sharpness or smoothed > ref * NO_TARGET_RISE
        else:
            rise = smoothed > ref * getattr(self.args, "rise_ratio", 1.4)
        if drop or rise:
            if self.low_since is None:
                self.low_since = now
            elif now - self.low_since >= self.args.trigger_delay:
                self.start(now, smoothed, "drop" if drop else "rise")
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
    parser.add_argument("--drop-ratio", type=float, default=0.9,
                        help="Refocus when score < reference x drop-ratio")
    parser.add_argument("--rise-ratio", type=float, default=1.4,
                        help="Refocus when score > reference x rise-ratio (new target)")
    parser.add_argument("--trigger-delay", type=float, default=0.2,
                        help="Seconds a drop/rise must persist before refocusing")
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--settle", type=float, default=0.06, help="Motor settle seconds")
    parser.add_argument("--motion-threshold", type=float, default=300.0,
                        help="ROI optical flow speed limit, analysis pixels/second")
    parser.add_argument("--min-sharpness", type=float, default=20.0,
                        help="Minimum verified Tenengrad score; scene dependent")
    args = parser.parse_args()
    if not 0.05 <= args.roi_size <= 0.9 or not 0 < args.drop_ratio < 1:
        parser.error("roi-size must be 0.05..0.9; drop-ratio must be 0..1")
    if args.rise_ratio <= 1:
        parser.error("rise-ratio must be > 1")
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
            lines = ["{} | focus={} | AF count={} | turbo | last={}".format(
                         controller.state, controller.position, controller.attempts, duration),
                     "Sharpness={:.1f} | reference={}{}".format(
                         sharpness, "none" if ref is None else "{:.1f}".format(ref),
                         " (NO TARGET)" if ref is not None and ref < args.min_sharpness else ""),
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
