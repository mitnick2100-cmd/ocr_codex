# encoding: UTF-8
'''
    Arducam programable zoom-lens controller.

    Copyright (c) 2019-4 Arducam <http://www.arducam.com>.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
    OR OTHER DEALINGS IN THE SOFTWARE.
'''

import cv2 #sudo apt-get install python-opencv
import numpy as py
import os
import sys
import time
import argparse
from JetsonCamera import Camera

from Focuser import Focuser
from Autofocus import AutoFocus

global image_count
image_count = 0

# IMX519 在目前 Jetson 驅動可穩定使用的輸出幀率。
# 15 FPS 會使用 3840x2160 感測器模式並由 Argus 降速輸出。
FPS_OPTIONS = (9, 15, 17, 60, 120)
SENSOR_MODES = {
    9: (4656, 3496),
    15: (3840, 2160),
    17: (3840, 2160),
    60: (1920, 1080),
    120: (1280, 720),
}

# 曝光時間每次按鍵調整的步進值（毫秒）
EXPOSURE_STEP_MS = 1.0
# 曝光時間下限（毫秒），避免設成0或負值
EXPOSURE_MIN_MS = 0.1
# 曝光最多使用單一畫格的 95%，保留少量時序安全裕度
EXPOSURE_MAX_RATIO = 0.95
# IMX519 驅動回報的硬體曝光上限（683709000 ns）。
EXPOSURE_SENSOR_MAX_MS = 683.709

# 對焦設定：步階固定 50，縮短等待時間以加快自動對焦掃描。
FOCUS_STEP = 50
FOCUS_SETTLE_TIME = 0.05

KEY_LEFT = (2424832, 65361, 81)
KEY_UP = (2490368, 65362, 82)
KEY_RIGHT = (2555904, 65363, 83)
KEY_DOWN = (2621440, 65364, 84)


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
    """
    依照 cam_state 目前的曝光/增益設定，重新配置相機參數。
    只會替換底層的 cv2.VideoCapture，不會重建預覽視窗，
    避免視窗重新彈出而搶走終端機的鍵盤焦點。

    調整曝光時間需要重建 Argus 擷取工作階段，這通常會讓整個相機模組
    （含對焦馬達晶片）斷電再通電，所以重新配置完成後，
    如果有帶入 focuser，會呼叫 reinit() 讓對焦功能恢復正常。
    """
    exposure_ns = None
    if cam_state['exposure_ms'] is not None:
        exposure_ns = int(cam_state['exposure_ms'] * 1000000)

    camera.reconfigure(
        framerate=cam_state['framerate'],
        exposure_time_ns=exposure_ns,
        gain_range=cam_state['gain_range'],
        ae_lock=(cam_state['exposure_ms'] is not None),
        capture_width=cam_state['capture_width'],
        capture_height=cam_state['capture_height'],
    )

    if focuser is not None:
        focuser.reinit()


def is_key(key, *values):
    return key in values or (key & 0xFF) in values


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


def draw_overlay(frame, key, focuser, cam_state, auto_focus):
    focus_value = "Focus: {}".format(focuser.get(Focuser.OPT_FOCUS))
    if auto_focus.is_finished():
        autofocus_value = "AutoFocus: Idle (best={})".format(auto_focus.best_position)
    else:
        autofocus_value = "AutoFocus: Running"

    if cam_state['exposure_ms'] is not None:
        exposure_ms = cam_state['exposure_ms']
        shutter_value = "Shutter: {:.2f} ms (~1/{:.0f}s)".format(
            exposure_ms, 1000.0 / exposure_ms
        )
    else:
        shutter_value = "Shutter: Auto"

    fps_text = "FPS: {} ({}x{})".format(
        cam_state['framerate'],
        cam_state['capture_width'],
        cam_state['capture_height'],
    )
    key_text = "Last key: {}".format(key) if key >= 0 else "Last key: None"
    lines = [
        "Arducam Controller",
        focus_value,
        autofocus_value,
        shutter_value,
        fps_text,
        key_text,
        "Q/Esc exit | C snapshot | A auto exposure | [/] shutter | -/= FPS",
        "Up/Down or W/S focus | Enter/F autofocus | R reset focus",
    ]
    for index, line in enumerate(lines):
        color = (0, 255, 255) if index == 0 else (255, 255, 255)
        draw_text(frame, line, (20, 35 + index * 30), color=color)

def parse_cmdline():
    parser = argparse.ArgumentParser(description='Arducam Controller.')

    parser.add_argument('-i', '--i2c-bus', type=int, default=10,
                        help='Set i2c bus. Default: 10')
    parser.add_argument('--width', type=int, default=1280,
                        help='Camera preview width. Default: 1280')
    parser.add_argument('--height', type=int, default=720,
                        help='Camera preview height. Default: 720')
    parser.add_argument('--framerate', type=int, choices=FPS_OPTIONS, default=17,
                        help='IMX519 FPS. Supported: {}. Default: 17'.format(
                            ', '.join(map(str, FPS_OPTIONS))))
    parser.add_argument('--exposure', type=float, default=None,
                        help='Initial exposure time in milliseconds (e.g. 20 for a 1/50s shutter). '
                             'If not set, auto exposure is used at startup.')
    parser.add_argument('--gain', type=str, default="1 1",
                        help='Fixed analog/digital gain range, e.g. "1 1". '
                             'Set to "" to leave auto gain enabled. Default: "1 1"')

    return parser.parse_args()


def change_framerate(camera, cam_state, focuser, direction):
    current_index = FPS_OPTIONS.index(cam_state['framerate'])
    new_index = max(0, min(len(FPS_OPTIONS) - 1, current_index + direction))
    new_fps = FPS_OPTIONS[new_index]
    if new_fps == cam_state['framerate']:
        return

    capture_width, capture_height = sensor_mode_for_fps(new_fps)
    cam_state['framerate'] = new_fps
    cam_state['capture_width'] = capture_width
    cam_state['capture_height'] = capture_height
    cam_state['exposure_ms'] = clamp_exposure_ms(
        cam_state['exposure_ms'], new_fps
    )
    reopen_camera_with_exposure(camera, cam_state, focuser)


# parse input key
def parseKey(k,focuser,auto_focus,camera,cam_state):
    global image_count
    focus_step = FOCUS_STEP

    key8 = k & 0xFF

    if key8 in (ord('-'), ord('_')):
        change_framerate(camera, cam_state, focuser, -1)
    elif key8 in (ord('='), ord('+')):
        change_framerate(camera, cam_state, focuser, 1)
    elif k in KEY_RIGHT or key8 == ord(']'):
        # 拉長曝光時間 -> 快門變慢，有助於拍到完整的LED看板數字
        current = cam_state['exposure_ms'] if cam_state['exposure_ms'] is not None else 0.0
        upper_bound = max_exposure_ms(cam_state['framerate'])
        cam_state['exposure_ms'] = min(current + EXPOSURE_STEP_MS, upper_bound)
        reopen_camera_with_exposure(camera, cam_state, focuser)
    elif k in KEY_LEFT or key8 == ord('['):
        # 縮短曝光時間 -> 快門變快
        if cam_state['exposure_ms'] is None:
            cam_state['exposure_ms'] = max_exposure_ms(cam_state['framerate'])
        cam_state['exposure_ms'] = max(cam_state['exposure_ms'] - EXPOSURE_STEP_MS, EXPOSURE_MIN_MS)
        reopen_camera_with_exposure(camera, cam_state, focuser)
    elif k in KEY_UP or key8 in (ord('w'), ord('W')):
        focuser.set(Focuser.OPT_FOCUS,focuser.get(Focuser.OPT_FOCUS) + focus_step)
    elif k in KEY_DOWN or key8 in (ord('s'), ord('S')):
        focuser.set(Focuser.OPT_FOCUS,focuser.get(Focuser.OPT_FOCUS) - focus_step)
    elif key8 in (ord('r'), ord('R')):
        focuser.reset(Focuser.OPT_FOCUS)
    elif key8 in (10, 13) or k in (65421, ord('f'), ord('F')):  # Enter or F
        auto_focus.start_focus()
    elif key8 in (ord('c'), ord('C')):
        #save image to file.
        cv2.imwrite("image{}.jpg".format(image_count), camera.getFrame())
        image_count += 1
    elif key8 in (ord('a'), ord('A')):
        # 切回自動曝光
        cam_state['exposure_ms'] = None
        reopen_camera_with_exposure(camera, cam_state, focuser)


def draw_menu(camera, i2c_bus, cam_state):
    focuser = Focuser(i2c_bus)
    auto_focus = AutoFocus(
        focuser, camera,
        roi=(0.4, 0.4, 0.2, 0.2),   # 對焦要看的區域 (x, y, width, height)，比例0~1
        focus_step=FOCUS_STEP,       # 每步移動量
        move_settle_time=FOCUS_SETTLE_TIME,  # 每步移動後等待秒數
        decline_patience=5,         # 連續下降幾次判定結束
        verbose=False,               # 想在終端機看每一步的清晰度數值可以打開
    )

    window_name = "Arducam Controller"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, cam_state['width'], cam_state['height'])
    last_key = -1

    while True:
        try:
            frame = camera.getFrame(timeout=1.0)
        except Exception:
            continue
        if frame is None:
            continue

        draw_overlay(frame, last_key, focuser, cam_state, auto_focus)
        cv2.imshow(window_name, frame)

        key = cv2.waitKeyEx(10)
        if key == -1:
            continue
        last_key = key
        if key == 27 or ((key & 0xFF) in (ord('q'), ord('Q')) and not is_arrow_key(key)):
            break
        parseKey(key, focuser, auto_focus, camera, cam_state)

    cv2.destroyWindow(window_name)

def main():
    args = parse_cmdline()

    gain_range = args.gain if args.gain else None
    capture_width, capture_height = sensor_mode_for_fps(args.framerate)
    exposure_ms = clamp_exposure_ms(args.exposure, args.framerate)
    if args.exposure is not None and exposure_ms != args.exposure:
        print("Exposure clamped from {:.3f} ms to {:.3f} ms for {} FPS".format(
            args.exposure, exposure_ms, args.framerate
        ))
    exposure_ns = int(exposure_ms * 1000000) if exposure_ms is not None else None

    cam_state = {
        'width': args.width,
        'height': args.height,
        'framerate': args.framerate,
        'exposure_ms': exposure_ms,   # None 代表自動曝光
        'gain_range': gain_range,
        'capture_width': capture_width,
        'capture_height': capture_height,
    }

    # open camera
    camera = Camera(
        width=args.width,
        height=args.height,
        framerate=args.framerate,
        exposure_time_ns=exposure_ns,
        gain_range=gain_range,
        ae_lock=(exposure_ms is not None),
        capture_width=capture_width,
        capture_height=capture_height,
    )
    print("Using I2C bus:", args.i2c_bus)
    try:
        draw_menu(camera, args.i2c_bus, cam_state)
    finally:
        camera.close()

    

if __name__ == "__main__":
    main()
