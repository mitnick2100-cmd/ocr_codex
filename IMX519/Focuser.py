'''
    Arducam programable zoom-lens control component.

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

import sys
import time
import os

# 優先用 smbus2 直接寫 I2C（每次約 <1ms）；原本每次寫入要開兩個 i2cset 外部程式，
# 一次移動約多花數十 ms。沒裝 smbus2 或寫入失敗時，自動退回原本的 i2cset 指令，
# 行為與原版相同。安裝：pip3 install smbus2
try:
    from smbus2 import SMBus
except ImportError:
    SMBus = None
    print("[Focuser] smbus2 not installed; falling back to i2cset (slower). "
          "Install with: pip3 install smbus2", file=sys.stderr)

_buses = {}

def _get_bus(bus):
    if SMBus is None:
        return None
    handle = _buses.get(bus)
    if handle is None:
        try:
            handle = SMBus(bus)
        except (OSError, IOError) as e:
            print("[Focuser] cannot open /dev/i2c-{} ({}); using i2cset".format(bus, e),
                  file=sys.stderr)
            return None
        _buses[bus] = handle
    return handle

def _write_reg(bus, address, reg, value):
    handle = _get_bus(bus)
    if handle is not None:
        try:
            handle.write_byte_data(address, reg, value)
            return
        except (OSError, IOError):
            # 相機重建時對焦晶片可能被斷電；關掉舊 handle，這次改用 i2cset，下次重新開啟
            try:
                handle.close()
            except Exception:
                pass
            _buses.pop(bus, None)
    os.system("i2cset -y {} 0x{:02x} 0x{:02x} 0x{:02x}".format(bus, address, reg, value))

def init(bus, address):
    _write_reg(bus, address, 0x02, 0x00)

def write(bus, address, value):
    value <<= 4
    _write_reg(bus, address, 0x00, value >> 8)
    _write_reg(bus, address, 0x01, value & 0xFF)

class Focuser:
    bus = None
    CHIP_I2C_ADDR = 0x0C

    def __init__(self, bus):
        self.focus_value = 0
        self.bus = bus
        self.verbose = False
        init(self.bus, self.CHIP_I2C_ADDR)

    def reinit(self):
        """
        重新初始化對焦馬達晶片，並把目前記住的對焦值重新寫回硬體。

        Arducam的對焦馬達晶片跟CSI相機感光元件共用同一路電源，
        當相機擷取工作階段(Argus session)被完全釋放/重新開啟時
        （例如動態調整曝光時間需要重建 cv2.VideoCapture），
        相機模組通常會被斷電再通電，導致對焦晶片回到未初始化狀態，
        必須重新呼叫 init() 才能再接受寫入指令，
        否則之後的對焦操作會被晶片忽略、看起來像是「失效」。
        """
        init(self.bus, self.CHIP_I2C_ADDR)
        # 把軟體記住的對焦值重新寫回硬體，避免斷電造成鏡頭實際位置跟畫面顯示的值不一致
        self.write(self.CHIP_I2C_ADDR, self.focus_value)
        
    def read(self):
        return self.focus_value

    def write(self, chip_addr, value):
        if value < 0:
            value = 0
        self.focus_value = value

        value = int(value / 1000.0 * 4095)

        write(self.bus, chip_addr, value)

    OPT_BASE    = 0x1000
    OPT_FOCUS   = OPT_BASE | 0x01
    OPT_ZOOM    = OPT_BASE | 0x02
    OPT_MOTOR_X = OPT_BASE | 0x03
    OPT_MOTOR_Y = OPT_BASE | 0x04
    OPT_IRCUT   = OPT_BASE | 0x05
    opts = {
        OPT_FOCUS : {
            "MIN_VALUE": 0,
            "MAX_VALUE": 1000,
            "DEF_VALUE": 0,
        },
    }
    def reset(self,opt,flag = 1):
        info = self.opts[opt]
        if info == None or info["DEF_VALUE"] == None:
            return
        self.set(opt,info["DEF_VALUE"])

    def get(self,opt,flag = 0):
        info = self.opts[opt]
        return self.read()

    def set(self,opt,value,flag = 1):
        info = self.opts[opt]
        if value > info["MAX_VALUE"]:
            value = info["MAX_VALUE"]
        elif value < info["MIN_VALUE"]:
            value = info["MIN_VALUE"]
        self.write(self.CHIP_I2C_ADDR, value)
        if self.verbose:
            print("write: {}".format(value))

pass 

def test():
    focuser = Focuser(7)
    focuser.set(Focuser.OPT_FOCUS, 0)
    time.sleep(3)
    focuser.set(Focuser.OPT_FOCUS, 1000)
    time.sleep(3)
    focuser.reset(Focuser.OPT_FOCUS)

if __name__ == "__main__":
    test()
