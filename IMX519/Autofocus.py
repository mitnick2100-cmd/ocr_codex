# -*- coding: utf-8 -*-
"""
自動對焦模組 (Contrast-Detection Autofocus)
=============================================

演算法：
    對焦馬達從最近端一路移動到最遠端，每一步都在畫面中央的ROI區域算一次
    Laplacian算子的方差當作「清晰度」指標 —— 影像越清晰，邊緣的高頻成分越多，
    Laplacian響應值的方差就越大，這是最常見的contrast-detection做法。

    掃描過程中一路追蹤目前最高的清晰度；一旦偵測到清晰度「連續下降」超過
    設定次數，代表已經掃過峰值，就提前中止掃描（不用整個行程跑完），
    最後把鏡頭移回全程清晰度最高的那個位置。

架構：
    AutoFocus.start_focus() 非阻塞啟動，背後開兩個thread：
      - _sweep_thread  : 移動馬達 + 取樣清晰度，結果丟進queue
      - _decide_thread : 消費queue，追蹤最佳位置 + 判斷提前終止 + 收尾移動鏡頭
    用 queue 解耦兩者，可以邊移動邊判斷，比「全部掃完再算」更快。
"""

import time
import cv2
import threading

from Focuser import Focuser

try:
    from  Queue import  Queue
except ModuleNotFoundError:
    from  queue import  Queue


def laplacian_sharpness(img):
    """用 Laplacian 算子的方差當作清晰度指標，數值越大代表影像邊緣越銳利(越清晰)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    return float(lap.var())


def crop_roi(frame, roi):
    """roi: (x, y, width, height)，四個值都是0~1的比例，取畫面中的一個子區域。"""
    h, w = frame.shape[:2]
    x0 = int(w * roi[0])
    y0 = int(h * roi[1])
    x1 = x0 + int(w * roi[2])
    y1 = y0 + int(h * roi[3])
    return frame[y0:y1, x0:x1]


class AutoFocus(object):
    """
    Contrast-detection 自動對焦控制器。

    用法：
        auto_focus = AutoFocus(focuser, camera)
        auto_focus.start_focus()              # 非阻塞，背景執行
        ...
        if auto_focus.is_finished():
            print("完成，最佳位置 =", auto_focus.best_position)
    """

    def __init__(self, focuser, camera,
                 roi=(0.4, 0.4, 0.2, 0.2),
                 focus_step=50,
                 move_settle_time=0.05,
                 decline_patience=3,
                 verbose=False):
        self.focuser = focuser
        self.camera = camera
        self.roi = roi                        # 只在畫面中央一小塊算清晰度，避免背景干擾、也比全畫面快
        self.focus_step = focus_step           # 每一步移動的對焦值
        self.move_settle_time = move_settle_time  # 每次移動後，給鏡頭穩定+等新影格的時間(秒)
        self.decline_patience = decline_patience  # 連續清晰度下降幾次就提前結束
        self.verbose = verbose

        self._sample_queue = Queue()
        self._finish_lock = threading.Lock()
        self._finished = True
        self.best_position = 0
        self.best_sharpness = 0.0

    # ---- 狀態存取 (thread-safe) ----
    def is_finished(self):
        with self._finish_lock:
            return self._finished

    def _set_finished(self, finished=True):
        with self._finish_lock:
            self._finished = finished

    # ---- 對外API ----
    def start_focus(self):
        """非阻塞啟動一次自動對焦流程；若上一次還在跑，這次呼叫會被忽略。"""
        if not self.is_finished():
            return

        self._sample_queue = Queue()
        self._set_finished(False)
        self.best_position = 0
        self.best_sharpness = 0.0

        threading.Thread(target=self._sweep_thread, daemon=True).start()
        threading.Thread(target=self._decide_thread, daemon=True).start()

    # ---- 內部：負責移動馬達 + 取樣清晰度 ----
    def _sweep_thread(self):
        max_position = self.focuser.opts[Focuser.OPT_FOCUS]["MAX_VALUE"]
        position = 0
        self.focuser.set(Focuser.OPT_FOCUS, position)

        while not self.is_finished() and position <= max_position:
            time.sleep(self.move_settle_time)

            frame = self.camera.getFrame(1000)
            if frame is None:
                continue

            roi_frame = crop_roi(frame, self.roi)
            sharpness = laplacian_sharpness(roi_frame)

            if self.verbose:
                print("[AutoFocus] position={}, sharpness={:.2f}".format(position, sharpness))

            self._sample_queue.put((position, sharpness))

            position += self.focus_step
            if position <= max_position and not self.is_finished():
                self.focuser.set(Focuser.OPT_FOCUS, position)

        # 結束訊號，讓 _decide_thread 知道沒有更多樣本了
        self._sample_queue.put((None, None))

    # ---- 內部：負責判斷峰值 + 提前終止 + 收尾移動鏡頭 ----
    def _decide_thread(self):
        samples = []
        decline_count = 0
        last_sharpness = None

        while True:
            position, sharpness = self._sample_queue.get()
            if position is None:
                break

            samples.append((position, sharpness))

            if last_sharpness is not None:
                if sharpness <= last_sharpness:
                    decline_count += 1
                else:
                    decline_count = 0
            last_sharpness = sharpness

            if decline_count >= self.decline_patience:
                # 通知 _sweep_thread 提前停止移動
                self._set_finished(True)

        if not samples:
            self._set_finished(True)
            return

        best_position, best_sharpness = max(samples, key=lambda item: item[1])
        self.best_position = best_position
        self.best_sharpness = best_sharpness

        if self.verbose:
            print("[AutoFocus] best position={}, sharpness={:.2f}".format(best_position, best_sharpness))

        self.focuser.set(Focuser.OPT_FOCUS, best_position)
        self._set_finished(True)
