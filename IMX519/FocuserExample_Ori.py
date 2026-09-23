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
import curses

global image_count
image_count = 0

# 曝光時間每次按鍵調整的步進值（毫秒）
EXPOSURE_STEP_MS = 1.0
# 曝光時間下限（毫秒），避免設成0或負值
EXPOSURE_MIN_MS = 0.1
# 相對於 1/framerate 的安全係數，避免曝光時間超過單一畫格的時間
EXPOSURE_MAX_RATIO = 0.95


def max_exposure_ms(framerate):
    return (1000.0 / framerate) * EXPOSURE_MAX_RATIO


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
    )

    if focuser is not None:
        focuser.reinit()


# Rendering status bar
def RenderStatusBar(stdscr):
    height, width = stdscr.getmaxyx()
    statusbarstr = "Press 'q' to exit"
    stdscr.attron(curses.color_pair(3))
    stdscr.addstr(height-1, 0, statusbarstr)
    stdscr.addstr(height-1, len(statusbarstr), " " * (width - len(statusbarstr) - 1))
    stdscr.attroff(curses.color_pair(3))
# Rendering description
def RenderDescription(stdscr):
    focus_desc      = "Focus    : Up-Down Arrow"
    autofocus_desc  = "AutoFocus: Enter Key"
    shutter_desc    = "Shutter  : Left-Right Arrow (slower/faster), 'a' = Auto exposure"
    snapshot_desc   = "Snapshot : 'c' Key"

    desc_y = 1
    
    stdscr.addstr(desc_y + 1, 0, focus_desc, curses.color_pair(1))
    stdscr.addstr(desc_y + 2, 0, autofocus_desc, curses.color_pair(1))
    stdscr.addstr(desc_y + 3, 0, shutter_desc, curses.color_pair(1))
    stdscr.addstr(desc_y + 4, 0, snapshot_desc, curses.color_pair(1))
# Rendering  middle text
def RenderMiddleText(stdscr,k,focuser,cam_state,auto_focus):
    # get height and width of the window.
    height, width = stdscr.getmaxyx()
    # Declaration of strings
    title = "Arducam Controller"[:width-1]
    subtitle = ""[:width-1]
    keystr = "Last key pressed: {}".format(k)[:width-1]
    
    
    # Obtain device infomation
    focus_value = "Focus    : {}".format(focuser.get(Focuser.OPT_FOCUS))[:width-1]

    if auto_focus.is_finished():
        autofocus_value = "AutoFocus: Idle (best={})".format(auto_focus.best_position)[:width-1]
    else:
        autofocus_value = "AutoFocus: Running..."[:width-1]

    if cam_state['exposure_ms'] is not None:
        exposure_ms = cam_state['exposure_ms']
        shutter_value = "Shutter  : {:.2f} ms (~1/{:.0f}s)".format(
            exposure_ms, 1000.0 / exposure_ms
        )[:width-1]
    else:
        shutter_value = "Shutter  : Auto"[:width-1]
    
    if k == 0:
        keystr = "No key press detected..."[:width-1]

    # Centering calculations
    start_x_title = int((width // 2) - (len(title) // 2) - len(title) % 2)
    start_x_subtitle = int((width // 2) - (len(subtitle) // 2) - len(subtitle) % 2)
    start_x_keystr = int((width // 2) - (len(keystr) // 2) - len(keystr) % 2)
    start_x_device_info = int((width // 2) - (len("Focus    : 00000") // 2) - len("Focus    : 00000") % 2)
    start_y = int((height // 2) - 6)
    
    # Turning on attributes for title
    stdscr.attron(curses.color_pair(2))
    stdscr.attron(curses.A_BOLD)

    # Rendering title
    stdscr.addstr(start_y, start_x_title, title)

    # Turning off attributes for title
    stdscr.attroff(curses.color_pair(2))
    stdscr.attroff(curses.A_BOLD)

    # Print rest of text
    stdscr.addstr(start_y + 1, start_x_subtitle, subtitle)
    stdscr.addstr(start_y + 3, (width // 2) - 2, '-' * 4)
    stdscr.addstr(start_y + 5, start_x_keystr, keystr)
    # Print device info
    stdscr.addstr(start_y + 6, start_x_device_info, focus_value)
    stdscr.addstr(start_y + 7, start_x_device_info, autofocus_value)
    stdscr.addstr(start_y + 8, start_x_device_info, shutter_value)

def parse_cmdline():
    parser = argparse.ArgumentParser(description='Arducam Controller.')

    parser.add_argument('-i', '--i2c-bus', type=int, nargs=None, required=True,
                        help='Set i2c bus, for A02 is 6, for B01 is 7 or 8, for Jetson Xavier NX it is 9 and 10.')
    parser.add_argument('--width', type=int, default=1280,
                        help='Camera preview width. Default: 1280')
    parser.add_argument('--height', type=int, default=720,
                        help='Camera preview height. Default: 720')
    parser.add_argument('--framerate', type=int, default=15,
                        help='Camera framerate. Exposure time cannot exceed 1/framerate seconds, '
                             'lower the framerate if you need a longer exposure. Default: 15')
    parser.add_argument('--exposure', type=float, default=None,
                        help='Initial exposure time in milliseconds (e.g. 20 for a 1/50s shutter). '
                             'If not set, auto exposure is used at startup.')
    parser.add_argument('--gain', type=str, default="1 1",
                        help='Fixed analog/digital gain range, e.g. "1 1". '
                             'Set to "" to leave auto gain enabled. Default: "1 1"')

    return parser.parse_args()

# parse input key
def parseKey(k,focuser,auto_focus,camera,cam_state):
    global image_count
    focus_step  = 50

    if k == ord('r'):
        focuser.reset(Focuser.OPT_FOCUS)
    elif k == curses.KEY_UP:
        focuser.set(Focuser.OPT_FOCUS,focuser.get(Focuser.OPT_FOCUS) + focus_step)
    elif k == curses.KEY_DOWN:
        focuser.set(Focuser.OPT_FOCUS,focuser.get(Focuser.OPT_FOCUS) - focus_step)
    elif k == 10:  # Enter
        auto_focus.start_focus()
    elif k == ord('c'):
        #save image to file.
        cv2.imwrite("image{}.jpg".format(image_count), camera.getFrame())
        image_count += 1
    elif k == curses.KEY_RIGHT:
        # 拉長曝光時間 -> 快門變慢，有助於拍到完整的LED看板數字
        current = cam_state['exposure_ms'] if cam_state['exposure_ms'] is not None else 0.0
        upper_bound = max_exposure_ms(cam_state['framerate'])
        cam_state['exposure_ms'] = min(current + EXPOSURE_STEP_MS, upper_bound)
        reopen_camera_with_exposure(camera, cam_state, focuser)
    elif k == curses.KEY_LEFT:
        # 縮短曝光時間 -> 快門變快
        if cam_state['exposure_ms'] is None:
            cam_state['exposure_ms'] = max_exposure_ms(cam_state['framerate'])
        cam_state['exposure_ms'] = max(cam_state['exposure_ms'] - EXPOSURE_STEP_MS, EXPOSURE_MIN_MS)
        reopen_camera_with_exposure(camera, cam_state, focuser)
    elif k == ord('a'):
        # 切回自動曝光
        cam_state['exposure_ms'] = None
        reopen_camera_with_exposure(camera, cam_state, focuser)


# Python curses example Written by Clay McLeod
# https://gist.github.com/claymcleod/b670285f334acd56ad1c
def draw_menu(stdscr, camera, i2c_bus, cam_state):
    focuser = Focuser(i2c_bus)
    auto_focus = AutoFocus(focuser, camera)
    

    k = 0
    cursor_x = 0
    cursor_y = 0

    # Clear and refresh the screen for a blank canvas
    stdscr.clear()
    stdscr.refresh()

    # Start colors in curses
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)

    # Loop where k is the last character pressed
    while (k != ord('q')):
        # Initialization
        stdscr.clear()
        # Flush all input buffers. 
        curses.flushinp()
        # get height and width of the window.
        height, width = stdscr.getmaxyx()

        # parser input key
        parseKey(k,focuser,auto_focus,camera,cam_state)

        # Rendering some text
        whstr = "Width: {}, Height: {}".format(width, height)
        stdscr.addstr(0, 0, whstr, curses.color_pair(1))

        # render key description
        RenderDescription(stdscr)
        # render status bar
        RenderStatusBar(stdscr)
        # render middle text
        RenderMiddleText(stdscr,k,focuser,cam_state,auto_focus)
        # Refresh the screen
        stdscr.refresh()

        # Wait for next input
        k = stdscr.getch()

def main():
    args = parse_cmdline()

    gain_range = args.gain if args.gain else None
    exposure_ns = int(args.exposure * 1000000) if args.exposure is not None else None

    # cam_state 集中保存目前相機的曝光/增益/畫質設定，
    # 供 parseKey() 調整快門時重新開啟相機使用
    cam_state = {
        'width': args.width,
        'height': args.height,
        'framerate': args.framerate,
        'exposure_ms': args.exposure,   # None 代表自動曝光
        'gain_range': gain_range,
    }

    #open camera
    camera = Camera(
        width=args.width,
        height=args.height,
        framerate=args.framerate,
        exposure_time_ns=exposure_ns,
        gain_range=gain_range,
        ae_lock=(args.exposure is not None),
    )
    #open camera preview
    camera.start_preview()
    print(args.i2c_bus)
    curses.wrapper(draw_menu, camera, args.i2c_bus, cam_state)

    camera.stop_preview()
    camera.close()

    

if __name__ == "__main__":
    main()
