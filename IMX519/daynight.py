"""開機日夜判斷：程式啟動後先用白天參數量 10 秒亮度，決定白天或晚上的 gain / conf。

只在開機時判斷一次，之後不再切換（跨過黃昏 / 清晨要重開程式）。
main.py 與 nohead_onetime_main.py 共用：add_arguments / check_args 處理參數，
initial_state 給開相機用的 gain / conf，主迴圈每幀呼叫 startup_step。

亮度量測：處理畫面縮成 160x90、跳過上方 15%，取 (像素/255)^2.2 的第 75 百分位
（去掉 gamma 回到近似線性的光量；看畫面較亮的四分之一，公車停在正前方擋住大半畫面時，
旁邊露出的街景仍算得到；不取最亮的部分，夜間的車燈、LED 看板才不會被算成白天）。
場景亮度 L = 該值 / gain / (曝光 / 15 ms)，不受當下 gain、曝光設定影響；開機時再取 10 秒中位數。

門檻依 10/22（白天 gain 1）、11/04（白天 gain 3）與夜間 gain 3 共 20 支錄影決定。
gain 3 錄影換算成 gain 1 要知道 "3 3" 的實際倍數 K（只有類比 gain 為 3，ISP 數位 gain 也是 3 時為 9），
0.031 對兩種 K 都成立：白天最低 0.044（深色公車停在正前方、K=9），夜間最高 0.022（K=3），
各約有 1.43 倍餘裕；約 8 萬個開機點都沒有判斷錯誤。原本用中位數、門檻 0.028 時，
同一段公車擋住的白天在 K=9 會被判成晚上。資料只有正午與深夜，黃昏 / 清晨還要實測確認。
"""
import time

import cv2
import numpy as np

DAY = "day"
NIGHT = "night"
PROFILES = {
    DAY: {"gain": "1 1", "conf": 0.5},
    NIGHT: {"gain": "3 3", "conf": 0.3},
}
REFERENCE_EXPOSURE_MS = 15.0   # 門檻量測時的曝光；開啟日夜判斷而沒給 --exposure 時用這個值
L_NIGHT = 0.031                # 10 秒中位數低於此值判定為晚上
PERCENTILE = 75                # 每幀取線性亮度的第幾百分位
STARTUP_SAMPLES = 10           # 開機量測筆數
SAMPLE_INTERVAL = 1.0          # 每筆間隔秒數（依時間取樣，不依幀數；主迴圈速度隨 YOLO 變動）
SKIP_TOP = 0.15                # 量測時跳過畫面上方的比例
MEASURE_SIZE = (160, 90)


# ---- measurement ---------------------------------------------------------
def gain_value(gain_range: str) -> float:
    """把 nvarguscamerasrc 的 "min max" 字串轉成單一 gain；min 與 max 必須相同。"""
    parts = gain_range.split()
    if len(parts) != 2 or float(parts[0]) != float(parts[1]):
        raise ValueError(f'gain range must be fixed ("N N"), got {gain_range!r}')
    return float(parts[0])


def measure(frame: np.ndarray) -> float:
    """畫面線性亮度的第 PERCENTILE 百分位（0～1）。"""
    small = cv2.resize(frame, MEASURE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)[int(MEASURE_SIZE[1] * SKIP_TOP):]
    return float(np.percentile((gray.astype(np.float32) / 255.0) ** 2.2, PERCENTILE))


def scene_level(linear: float, gain: float, exposure_ms: float) -> float:
    """換算成與 gain、曝光無關的場景亮度 L。"""
    return linear / gain / (exposure_ms / REFERENCE_EXPOSURE_MS)


def decide(levels) -> str:
    return NIGHT if float(np.median(levels)) < L_NIGHT else DAY


class StartupProbe:
    """收集開機量測值；收滿 STARTUP_SAMPLES 筆後 update() 回傳 DAY / NIGHT，其餘時候回傳 None。"""

    def __init__(self, gain_range: str, exposure_ms: float,
                 samples: int = STARTUP_SAMPLES, interval: float = SAMPLE_INTERVAL):
        self.gain = gain_value(gain_range)
        self.exposure_ms = exposure_ms
        self.samples = samples
        self.interval = interval
        self.levels = []
        self.next_time = None
        self.result = None

    @property
    def done(self) -> bool:
        return self.result is not None

    @property
    def level(self) -> float | None:
        """目前為止的 L 中位數。"""
        return float(np.median(self.levels)) if self.levels else None

    def update(self, frame: np.ndarray, now: float) -> str | None:
        if self.done or (self.next_time is not None and now < self.next_time):
            return None
        self.levels.append(scene_level(measure(frame), self.gain, self.exposure_ms))
        self.next_time = now + self.interval
        if len(self.levels) < self.samples:
            return None
        self.result = decide(self.levels)
        return self.result


# ---- program integration -------------------------------------------------
def add_arguments(parser):
    parser.add_argument("--no-auto-daynight", action="store_true",
                        help="Do not detect day / night at startup; use --gain and --conf as given")


def check_args(args):
    """Set args.auto_daynight; with detection on, a missing --exposure becomes 15 ms."""
    args.auto_daynight = not args.no_auto_daynight
    if args.auto_daynight:
        # 門檻是在固定曝光下量的；自動曝光會把亮度拉回固定值，日夜就分不出來
        if args.exposure is None:
            args.exposure = REFERENCE_EXPOSURE_MS
        print("Day / night detection: gain / conf come from the day / night profiles "
              f"(exposure {args.exposure:g} ms); --no-auto-daynight to set them by hand")


def initial_state(args, exposure_ms):
    """cam_state entries for opening the camera: gain_range, conf, mode, probe.

    With detection on the camera always opens with the day profile; only a dark
    scene reopens it once with the night gain.
    """
    if not args.auto_daynight:
        return {"gain_range": args.gain or None, "conf": args.conf,
                "mode": "FIXED", "probe": None}
    day = PROFILES[DAY]
    return {"gain_range": day["gain"], "conf": day["conf"], "mode": "PROBING",
            "probe": StartupProbe(day["gain"], exposure_ms)}


def mode_text(cam_state):
    """日夜模式、gain、conf；開機判斷中顯示進度與目前的 L。"""
    gain = cam_state["gain_range"] or "auto"
    text = f"Mode: {cam_state['mode']} | gain {gain} | conf {cam_state['conf']:.2f}"
    probe = cam_state["probe"]
    if probe is not None:
        level = "--" if probe.level is None else f"{probe.level:.4f}"
        text += f" | probing {len(probe.levels)}/{probe.samples} s, L={level}"
    return text


def apply_profile(cam_state, mode, reopen):
    """Use a day / night profile: reopen with its gain (if different), then use its conf.

    `reopen()` reopens the camera with cam_state's settings and raises RuntimeError
    when it cannot (JetsonCamera.Camera.reconfigure gives up after 10 attempts,
    with the old capture already released). A failed switch reopens with the
    previous gain instead of ending the program. Return True when the profile is in use.
    """
    profile = PROFILES[mode]
    previous_gain = cam_state["gain_range"]
    if profile["gain"] != previous_gain:
        cam_state["gain_range"] = profile["gain"]
        try:
            reopen()
        except RuntimeError as error:
            print(f"Switching to {mode.upper()} (gain {profile['gain']}) failed: {error}; "
                  f"reopening with gain {previous_gain}")
            cam_state["gain_range"] = previous_gain
            reopen()
            return False
    cam_state["conf"] = profile["conf"]
    cam_state["mode"] = mode.upper()
    return True


def startup_step(cam_state, frame, now, reopen):
    """Feed one clean processing frame to the startup probe; apply its result when ready."""
    probe = cam_state["probe"]
    if probe is None:
        return
    result = probe.update(frame, now)
    if result is None:
        return
    cam_state["probe"] = None
    profile = PROFILES[result]
    print(f"[{time.strftime('%H:%M:%S')}] Startup: {result.upper()} "
          f"(L={probe.level:.4f}, night below {L_NIGHT}) -> "
          f"gain {profile['gain']}, conf {profile['conf']}")
    # 白天：gain 不變，只設 conf / 模式；晚上：重開一次換夜間 gain
    if not apply_profile(cam_state, result, reopen):
        apply_profile(cam_state, DAY, reopen)   # gain 已退回白天：讓 conf / 模式一致
        print("Staying with the DAY profile")


def cancel_startup(cam_state, reopen):
    """A manual exposure / FPS change skews the measurement: keep the day profile instead."""
    if cam_state["probe"] is None:
        return
    cam_state["probe"] = None
    apply_profile(cam_state, DAY, reopen)
    print("Day / night detection cancelled by a manual camera key; keeping the DAY profile")
