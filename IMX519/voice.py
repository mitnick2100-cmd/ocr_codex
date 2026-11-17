"""公車進站中文語音播報：OCR 確認號碼後，用 piper 合成「N號公車即將進站」並從預設喇叭播放。

nohead_onetime_main.py 使用：add_arguments / check_args 處理參數，建立 VoiceAnnouncer 後
在 CONFIRMED 時呼叫 announce(number)，結束時呼叫 close()。

播放流程全部在一個背景 daemon thread：啟動時開一個常駐的 piper 子程序（--json-input，
模型只載入一次，約 0.4 s），每則播報寫一行 JSON 給 piper，piper 在暫存資料夾寫出 wav 並把路徑印到
stdout，再用 paplay 播放（走 PulseAudio 預設輸出），播完刪除 wav。一次只播一則，不會互相打斷。
啟動時先合成一句不播放的暖機句子（WARMUP_TEXT），第一台車的播報就不用等第一次推論。

數字唸法（2026-11-17 實測 piper zh_CN-huayan-medium）：
1～2 位數直接交給 piper，會照整數唸（63 → 六十三）；3～4 位數先轉成國字逐字唸
（358 → 三五八、1050 → 一零五零；使用者 2026-11-17 決定，三位數 piper 本來會唸成三百五十八，
4 位數則會唸錯：1050 → 一千五十，少了「零」）。--voice-digits 時 1～2 位數也逐字唸。

piper 用官方 aarch64 單獨執行檔（~/piper/piper/），不用 pip 的 piper-tts：
pip 版會裝一般的 onnxruntime，跟 Jetson 的 onnxruntime-gpu 衝突。
任何一步失敗都只印警告並停用語音，不影響辨識主程式。
"""
import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time

VOICE_TEMPLATE = "{}號公車即將進站"
DEFAULT_PIPER_BIN = "~/piper/piper/piper"
DEFAULT_PIPER_MODEL = "~/piper/voices/zh_CN-huayan-medium.onnx"
PLAYER = ["paplay"]            # 走 PulseAudio 預設輸出（目前是 USB 耳機）
QUEUE_SIZE = 4                 # 待播佇列上限；滿了就丟掉新的
DEFAULT_COOLDOWN = 30.0        # 同一個號碼幾秒內不重複播報（ByteTrack 換 ID 時避免重唸）
SPEED_RANGE = (0.5, 2.0)       # --voice-speed 允許範圍（piper length_scale，越大越慢）
PIPER_TIMEOUT = 10.0           # 單句合成逾時（實測約 0.2～0.3 s）
PLAY_TIMEOUT = 15.0            # 單句播放逾時（實測一句約 2～2.5 s）
CHINESE_DIGITS = "零一二三四五六七八九"
# 啟動時先合成一次、不播放：第一次推論較慢（實測 0.6～0.8 s，之後約 0.25 s），
# 先暖機讓第一台車的播報不用多等。句型與實際播報相同，數字涵蓋 0～9。
WARMUP_TEXT = VOICE_TEMPLATE.format(CHINESE_DIGITS)
DIGITWISE_MIN_LEN = 3          # 幾位數以上逐字唸（358 → 三五八）；以下照整數唸（63 → 六十三）


def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] [VOICE] {message}", flush=True)


# ---- text ----------------------------------------------------------------
def spoken_number(number: str, digits: bool = False) -> str:
    """把 OCR 號碼轉成要給 piper 的文字。"""
    number = number.strip()
    if not re.fullmatch(r"\d+", number):
        return number
    if digits or len(number) >= DIGITWISE_MIN_LEN:
        return "".join(CHINESE_DIGITS[int(d)] for d in number)
    return str(int(number))    # 去掉開頭的 0，例如 "05" → "5"


def sentence(number: str, digits: bool = False) -> str:
    return VOICE_TEMPLATE.format(spoken_number(number, digits))


# ---- arguments -----------------------------------------------------------
def add_arguments(parser):
    group = parser.add_argument_group("voice announcement")
    group.add_argument("--no-voice", action="store_true",
                       help="Disable voice announcement")
    group.add_argument("--voice-cooldown", type=float, default=DEFAULT_COOLDOWN,
                       help="Seconds before the same bus number may be announced again "
                            f"(default: {DEFAULT_COOLDOWN:g})")
    group.add_argument("--voice-digits", action="store_true",
                       help="Read every bus number digit by digit "
                            "(numbers with 3 or more digits are always read digit by digit)")
    group.add_argument("--voice-speed", type=float, default=1.0,
                       help="piper length_scale; larger is slower "
                            f"({SPEED_RANGE[0]:g}-{SPEED_RANGE[1]:g}, default: 1.0)")
    group.add_argument("--piper-bin", default=DEFAULT_PIPER_BIN,
                       help=f"piper executable (default: {DEFAULT_PIPER_BIN})")
    group.add_argument("--piper-model", default=DEFAULT_PIPER_MODEL,
                       help=f"piper voice model (default: {DEFAULT_PIPER_MODEL})")


def check_args(args):
    """Validate ranges and expand ~ in paths; exits through SystemExit on bad values."""
    if args.voice_cooldown < 0:
        raise SystemExit("--voice-cooldown must be >= 0")
    if not SPEED_RANGE[0] <= args.voice_speed <= SPEED_RANGE[1]:
        raise SystemExit(f"--voice-speed must be in {SPEED_RANGE[0]:g}-{SPEED_RANGE[1]:g}")
    args.piper_bin = os.path.expanduser(args.piper_bin)
    args.piper_model = os.path.expanduser(args.piper_model)


# ---- announcer -----------------------------------------------------------
class VoiceAnnouncer:
    """背景播報器；announce() 不會卡住呼叫的執行緒。"""

    def __init__(self, args):
        self.enabled = not args.no_voice
        self.muted = False
        self.cooldown = args.voice_cooldown
        self.digits = args.voice_digits
        self.speed = args.voice_speed
        self.piper_bin = args.piper_bin
        self.piper_model = args.piper_model
        self.queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.last_spoken = {}              # number -> monotonic time（排入佇列的時間）
        self.lock = threading.Lock()
        self.proc_lock = threading.Lock()  # 保護「開新 player」與 close() 結束子程序之間的競爭
        self.stopping = threading.Event()
        self.piper = None
        self.player = None
        self.tmpdir = None
        self.thread = None
        if self.enabled:
            self.thread = threading.Thread(target=self._run, name="voice", daemon=True)
            self.thread.start()

    # -- public ------------------------------------------------------------
    def announce(self, number: str) -> bool:
        """排入一則播報；被停用、靜音、冷卻或佇列已滿時回傳 False。"""
        if not self.enabled or self.muted or self.stopping.is_set():
            return False
        number = str(number).strip()
        now = time.monotonic()
        with self.lock:
            last = self.last_spoken.get(number)
            if last is not None and now - last < self.cooldown:
                log(f"cooldown skip {number} ({now - last:.1f} s < {self.cooldown:g} s)")
                return False
            try:
                self.queue.put_nowait(number)
            except queue.Full:
                log(f"queue full, dropped {number}")
                return False
            self.last_spoken[number] = now
        return True

    def toggle(self) -> bool:
        """切換靜音；回傳切換後是否會播報。"""
        self.muted = not self.muted
        log("muted" if self.muted else "unmuted")
        return self.enabled and not self.muted

    def wait_idle(self, timeout: float | None = None) -> bool:
        """等佇列播完（單獨執行測試用）；背景執行緒結束時也會返回。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.queue.unfinished_tasks and self.thread is not None and self.thread.is_alive():
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return True

    def close(self):
        """停止播報：中斷正在播的句子、結束 piper，不等佇列播完。"""
        if self.thread is None:
            return
        with self.proc_lock:
            self.stopping.set()
            for proc in (self.player, self.piper):
                if proc is not None and proc.poll() is None:
                    proc.terminate()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=2.0)
        self.thread = None

    # -- worker ------------------------------------------------------------
    def _disable(self, reason: str):
        log(f"WARNING: {reason}; voice disabled")
        self.enabled = False

    def _start_piper(self) -> bool:
        if not os.path.isfile(self.piper_bin) or not os.access(self.piper_bin, os.X_OK):
            self._disable(f"piper not found: {self.piper_bin}")
            return False
        if not os.path.isfile(self.piper_model):
            self._disable(f"piper model not found: {self.piper_model}")
            return False
        if shutil.which(PLAYER[0]) is None:
            self._disable(f"{PLAYER[0]} not found")
            return False
        self.tmpdir = tempfile.mkdtemp(prefix="voice_")
        try:
            # start_new_session：終端機 Ctrl+C 會送 SIGINT 給整個程序群組，子程序若一起收到，
            # 會在 close() 之前先結束，背景執行緒就可能接著開下一句的 paplay 而留下孤兒程序；
            # 讓子程序不在同一群組，一律由 close() 結束。
            self.piper = subprocess.Popen(
                [self.piper_bin, "-m", self.piper_model, "--json-input",
                 "-d", self.tmpdir, "-q", "--length_scale", f"{self.speed:g}"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, encoding="utf-8", bufsize=1, start_new_session=True)
        except OSError as error:
            self._disable(f"cannot start piper: {error}")
            return False
        return True

    def _synthesize(self, text: str) -> str:
        """回傳 wav 路徑；失敗時丟 RuntimeError。"""
        if self.piper.poll() is not None:
            raise RuntimeError(f"piper exited with code {self.piper.returncode}")
        self.piper.stdin.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        self.piper.stdin.flush()
        # piper 寫完 wav 才印路徑；readline 沒有逾時，用計時執行緒在卡住時結束 piper
        timed_out = threading.Event()

        def kill():
            timed_out.set()
            self.piper.kill()

        timer = threading.Timer(PIPER_TIMEOUT, kill)
        timer.start()
        try:
            path = self.piper.stdout.readline().strip()
        finally:
            timer.cancel()
        if not path or not os.path.isfile(path):
            if timed_out.is_set():
                raise RuntimeError(f"piper timeout ({PIPER_TIMEOUT:g} s)")
            try:
                code = self.piper.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                code = "still running"
            raise RuntimeError(f"piper produced no audio (exit code {code})")
        return path

    def _play(self, path: str):
        with self.proc_lock:
            if self.stopping.is_set():
                return
            self.player = subprocess.Popen(PLAYER + [path], stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, text=True,
                                           start_new_session=True)
        try:
            _, err = self.player.communicate(timeout=PLAY_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.player.kill()
            self.player.communicate()
            log(f"WARNING: {PLAYER[0]} timeout")
            return
        if self.player.returncode != 0 and not self.stopping.is_set():
            # 播放失敗只警告、不停用：PulseAudio 暫時不在（例如耳機重新連線）時下一則可能就好了
            log(f"WARNING: {PLAYER[0]} failed ({self.player.returncode}): {err.strip()}")

    def _warmup(self) -> bool:
        """合成一句不播放的句子；失敗時停用語音並回傳 False。"""
        try:
            started = time.monotonic()
            path = self._synthesize(WARMUP_TEXT)
        except (OSError, RuntimeError, ValueError) as error:
            if not self.stopping.is_set():
                self._disable(f"piper warmup error: {error}")
            return False
        try:
            os.remove(path)
        except OSError:
            pass
        log(f"ready (piper warmup {time.monotonic() - started:.2f} s)")
        return True

    def _run(self):
        ok = self._start_piper() and self._warmup()
        try:
            while True:
                number = self.queue.get()
                try:
                    if number is None or self.stopping.is_set():
                        break
                    if not ok or not self.enabled:
                        continue           # 已停用：只消化佇列
                    text = sentence(number, self.digits)
                    try:
                        started = time.monotonic()
                        path = self._synthesize(text)
                    except (OSError, RuntimeError, ValueError) as error:
                        if not self.stopping.is_set():
                            self._disable(f"piper error: {error}")
                        continue
                    log(f"{text}  (synth {time.monotonic() - started:.2f} s)")
                    try:
                        self._play(path)
                    except OSError as error:
                        self._disable(f"cannot run {PLAYER[0]}: {error}")
                    finally:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                finally:
                    self.queue.task_done()
        finally:
            if self.piper is not None and self.piper.poll() is None:
                try:
                    self.piper.stdin.close()
                except OSError:
                    pass
                try:
                    self.piper.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.piper.kill()
            if self.tmpdir is not None:
                shutil.rmtree(self.tmpdir, ignore_errors=True)


# ---- standalone test -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Announce bus numbers in Mandarin, e.g. python3 voice.py 5 300 1050 5")
    add_arguments(parser)
    parser.add_argument("numbers", nargs="+", help="bus numbers to announce, in order")
    args = parser.parse_args()
    check_args(args)
    announcer = VoiceAnnouncer(args)
    try:
        for number in args.numbers:
            announcer.announce(number)
        announcer.wait_idle()
    except KeyboardInterrupt:
        pass
    finally:
        announcer.close()


if __name__ == "__main__":
    main()
