# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
import time
try:
    from  Queue import  Queue, Empty
except ModuleNotFoundError:
    from  queue import  Queue, Empty

import  threading
import signal
import sys


# def signal_handler(sig, frame):
#     print('You pressed Ctrl+C!')
#     sys.exit(0)
# signal.signal(signal.SIGINT, signal_handler)


def gstreamer_pipeline(
    capture_width=3840,
    capture_height=2160,
    display_width=1280,
    display_height=720,
    framerate=15,
    flip_method=0,
    exposure_time_ns=None,   # 例如 20000000 代表 20ms (1/50s)，可解決LED看板閃爍/斷字問題
    gain_range=None,         # 例如 "1 1" 鎖定增益為1，避免自動增益放大雜訊掩蓋曝光效果
    ae_lock=False,           # 是否額外鎖定自動曝光演算法
):
    """
    exposure_time_ns: 單一數字或 (min_ns, max_ns) tuple，單位為奈秒。
                       min=max 代表固定曝光時間（不受自動曝光影響）。
                       None 表示保留自動曝光。
                       注意：曝光時間不可超過 1/framerate 秒，
                       想要曝光更久就要把 framerate 相應調低。
    gain_range: 字串 "min max"，例如 "1 1"。None 表示保留自動增益。
    ae_lock: True 時加上 aelock=true。
    """
    extra_props = ""

    if exposure_time_ns is not None:
        if isinstance(exposure_time_ns, (tuple, list)):
            min_exp, max_exp = exposure_time_ns
        else:
            min_exp = max_exp = exposure_time_ns
        extra_props += ' exposuretimerange="%d %d"' % (min_exp, max_exp)

    if gain_range is not None:
        extra_props += ' gainrange="%s" ispdigitalgainrange="%s"' % (gain_range, gain_range)

    if ae_lock:
        extra_props += " aelock=true"

    return (
        "nvarguscamerasrc%s ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            extra_props,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class FrameReader(threading.Thread):
    queues = []
    _running = True
    camera = None
    def __init__(self, camera, name):
        threading.Thread.__init__(self)
        self.name = name
        self.camera = camera
        self._camera_lock = threading.Lock()

    def run(self):
        while self._running:
            with self._camera_lock:
                cam = self.camera
            ret, frame = cam.read()
            if not ret or frame is None:
                # 相機正在重新設定（例如調整曝光）或暫時讀不到影格，稍微等待再試，
                # 避免忙碌迴圈，也避免把無效影格塞進queue
                time.sleep(0.01)
                continue
            while self.queues:
                queue = self.queues.pop()
                queue.put(frame)
    
    def addQueue(self, queue):
        self.queues.append(queue)

    def getFrame(self, timeout = None):
        queue = Queue(1)
        self.addQueue(queue)
        return queue.get(timeout = timeout)

    def set_camera(self, camera):
        """執行期間切換底層的 cv2.VideoCapture，不需要重啟執行緒。"""
        with self._camera_lock:
            self.camera = camera

    def stop(self):
        self._running = False

class Previewer(threading.Thread):
    window_name = "Arducam"
    _running = True
    camera = None
    def __init__(self, camera, name):
        threading.Thread.__init__(self)
        self.name = name
        self.camera = camera
    
    def run(self):
        self._running = True
        while self._running:
            try:
                frame = self.camera.getFrame(2000)
            except Empty:
                # 相機正在重新設定（例如調整曝光），暫時沒有新影格，
                # 不要讓例外中斷這個執行緒，繼續等下一張即可
                continue
            if frame is None:
                continue
            cv2.imshow(self.window_name, frame)
            keyCode = cv2.waitKey(16) & 0xFF
        cv2.destroyWindow(self.window_name)

    def start_preview(self):
        self.start()
    def stop_preview(self):
        self._running = False

class Camera(object):
    frame_reader = None
    cap = None
    previewer = None

    def __init__(self, width=1280, height=720,
                 framerate=15, exposure_time_ns=None, gain_range=None, ae_lock=False,
                 capture_width=3840, capture_height=2160):
        self.open_camera(width, height, framerate, exposure_time_ns, gain_range,
                         ae_lock, capture_width, capture_height)

    def open_camera(self, width=1280, height=720,
                    framerate=15, exposure_time_ns=None, gain_range=None, ae_lock=False,
                    capture_width=3840, capture_height=2160):
        self._width = width
        self._height = height
        self._framerate = framerate
        self._capture_width = capture_width
        self._capture_height = capture_height

        self.cap = cv2.VideoCapture(
            gstreamer_pipeline(
                capture_width=capture_width,
                capture_height=capture_height,
                flip_method=0,
                display_width=width,
                display_height=height,
                framerate=framerate,
                exposure_time_ns=exposure_time_ns,
                gain_range=gain_range,
                ae_lock=ae_lock,
            ),
            cv2.CAP_GSTREAMER,
        )
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera!")
        if self.frame_reader is None or not self.frame_reader.is_alive():
            self.frame_reader = FrameReader(self.cap, "")
            self.frame_reader.daemon = True
            self.frame_reader.start()
        else:
            self.frame_reader.set_camera(self.cap)
        self.previewer = Previewer(self.frame_reader, "")

    def reconfigure(self, framerate=None, exposure_time_ns=None,
                     gain_range=None, ae_lock=False,
                     capture_width=None, capture_height=None):
        """
        重新建立底層 VideoCapture，可同時切換 FPS 與 IMX519 感測器擷取解析度。
        """
        framerate = framerate if framerate is not None else self._framerate
        capture_width = capture_width if capture_width is not None else self._capture_width
        capture_height = capture_height if capture_height is not None else self._capture_height

        old_cap = self.cap
        old_cap.release()
        time.sleep(0.3)

        new_cap = None
        for _ in range(10):
            candidate = cv2.VideoCapture(
                gstreamer_pipeline(
                    capture_width=capture_width,
                    capture_height=capture_height,
                    flip_method=0,
                    display_width=self._width,
                    display_height=self._height,
                    framerate=framerate,
                    exposure_time_ns=exposure_time_ns,
                    gain_range=gain_range,
                    ae_lock=ae_lock,
                ),
                cv2.CAP_GSTREAMER,
            )
            if candidate.isOpened():
                new_cap = candidate
                break
            candidate.release()
            time.sleep(0.3)

        if new_cap is None:
            raise RuntimeError("Failed to reopen camera with new settings!")

        self.frame_reader.set_camera(new_cap)
        self.cap = new_cap
        self._framerate = framerate
        self._capture_width = capture_width
        self._capture_height = capture_height

    def getFrame(self, timeout=None):
        return self.frame_reader.getFrame(timeout)

    def start_preview(self):
        self.previewer.daemon = True
        self.previewer.start_preview()

    def stop_preview(self):
        self.previewer.stop_preview()
        self.previewer.join()

    def close(self):
        self.frame_reader.stop()
        self.cap.release()

if __name__ == "__main__":
    # 範例：framerate 30fps，曝光時間固定為 20ms (1/50s)
    camera = Camera(
        framerate=30,
        exposure_time_ns=20000000,
        gain_range="1 1",
        ae_lock=True,
    )
    camera.start_preview()
    time.sleep(10)
    camera.stop_preview()
    camera.close()
