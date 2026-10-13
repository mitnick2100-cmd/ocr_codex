# -*- coding: utf-8 -*-
"""長曝光 LED 看板 ROI 前處理。

背景：為了取得完整 LED 數字（避開 PWM 閃爍造成的斷字），相機刻意使用長曝光，
結果是 LED 核心削白、光暈外擴、數字之間的間隙被填滿。

核心觀察：彩色 LED 對 R/G/B 三個通道而言，等同隔著強度不同的 ND 濾鏡。
主通道（例如紅字的 R）在整個光暈範圍都已飽和，但較弱的通道只有在真正的
筆畫核心才會被推高。因此把「排序後」的通道加權相加：

    E = 1 * max + 2 * mid + 3 * min

E 在主通道削白後仍會隨真實亮度單調遞增（單張影像的偽 HDR），而且弱通道
權重較大，光暈被壓暗、筆畫間隙重新出現。未過曝時弱通道很小，E 幾乎等於
主通道，不會破壞正常影像。

流程：通道加權亮度 -> 穩健拉伸 -> CLAHE -> 取代 LAB 的 L（保留 a、b 色彩）。
選用：看板玻璃反光（整片霧白）時的暗通道去霧，預設關閉。

刻意不使用：
- 自動 gamma：對單一通道是單調轉換，不會增加筆畫與背景的可分離度；在接近
  全白的影像上還會造成色階斷層。
- Top-hat / 背景扣除：光暈讓數字變成比結構元素更大的白塊，會被當成背景扣掉。
- 大半徑銳化、去卷積：在削白平台邊緣產生假影，模糊條件下反而降低分離度。
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

CHANNEL_WEIGHTS = (1.0, 2.0, 3.0)   # (max, mid, min)
STRETCH_PERCENTILES = (0.5, 99.8)
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
GLARE_THRESHOLD = 0.35               # glare_level() 超過此值才建議開啟去霧
QUALITY_HEIGHT = 64


def _channel_max_min(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐像素的最亮與最暗通道（cv2 逐元素運算，比 numpy 沿 axis=2 快）。"""
    blue, green, red = cv2.split(image)
    return cv2.max(cv2.max(blue, green), red), cv2.min(cv2.min(blue, green), red)


def channel_weighted_luma(image: np.ndarray,
                          weights: tuple[float, float, float] = CHANNEL_WEIGHTS) -> np.ndarray:
    """以排序後的 (max, mid, min) 通道加權相加，回傳拉伸到 0..255 的亮度。"""
    if image is None or image.size == 0:
        raise ValueError("ROI 不可為空")
    brightest, darkest = _channel_max_min(image)
    brightest, darkest = brightest.astype(np.float32), darkest.astype(np.float32)
    middle = image.sum(axis=2, dtype=np.float32) - brightest - darkest
    energy = weights[0] * brightest + weights[1] * middle + weights[2] * darkest
    low, high = np.percentile(energy, STRETCH_PERCENTILES)
    stretched = (energy - low) / max(float(high - low), 1e-6) * 255
    return np.clip(stretched, 0, 255).astype(np.uint8)


def _odd(value: float) -> int:
    value = max(3, int(round(value)))
    return value if value % 2 else value + 1


def glare_level(image: np.ndarray) -> float:
    """暗通道中位數（0..1）。LED 看板底色應接近黑色，數值高代表玻璃反光霧化。"""
    kernel_size = _odd(image.shape[0] * 0.08)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dark = cv2.erode(_channel_max_min(image)[1], kernel)
    cumulative = np.cumsum(np.bincount(dark.ravel(), minlength=256))
    return float(np.searchsorted(cumulative, dark.size / 2)) / 255.0


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    box = lambda x: cv2.boxFilter(x, -1, (radius, radius))
    mean_i, mean_p = box(guide), box(src)
    a = (box(guide * src) - mean_i * mean_p) / (box(guide * guide) - mean_i * mean_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def dehaze_glare(image: np.ndarray, omega: float = 0.85, t_min: float = 0.35,
                 clip_threshold: int = 242) -> np.ndarray:
    """暗通道先驗去霧，把玻璃反光視為加性的大氣光。

    與一般 DCP 的差異：
    - 削白像素（min(RGB) 已飽和）的透射率固定為 1，避免把 LED 核心「去霧」成彩色雜點。
    - t_min 提高到 0.35，雜訊放大上限約 2.9 倍，適合小尺寸、JPEG 壓縮過的 ROI。
    """
    pixels = image.astype(np.float32) / 255.0
    kernel_size = _odd(image.shape[0] * 0.08)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dark = cv2.erode(pixels.min(axis=2), kernel).ravel()
    count = max(1, dark.size // 1000)
    brightest = np.argpartition(dark, -count)[-count:]
    airlight = np.maximum(pixels.reshape(-1, 3)[brightest].max(axis=0), 0.6)

    transmission = 1.0 - omega * cv2.erode((pixels / airlight).min(axis=2), kernel)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    transmission = _guided_filter(gray, transmission.astype(np.float32), kernel_size, 1e-3)
    transmission = np.clip(transmission, t_min, 1.0)

    clipped = (image.min(axis=2) >= clip_threshold).astype(np.float32)
    clipped = cv2.GaussianBlur(cv2.dilate(clipped, np.ones((3, 3), np.uint8)), (0, 0), 1.5)
    transmission = transmission * (1 - clipped) + clipped

    restored = (pixels - airlight) / transmission[..., None] + airlight
    return np.clip(restored * 255, 0, 255).astype(np.uint8)


def enhance_led_roi(roi_img: np.ndarray, clahe_clip: float = CLAHE_CLIP,
                    dehaze: bool | None = False,
                    color: bool = True) -> tuple[np.ndarray, dict]:
    """長曝光 LED 看板增強；可直接取代 V13 的 enhance_roi。

    dehaze：False 不去霧（預設）、True 一律去霧、None 依 glare_level 自動判斷。
    color：True 保留原本的 a、b 色彩（黃字壓暗後會偏橄欖色）；False 輸出灰階。
    回傳 (增強後 BGR 影像, 統計資訊)。
    """
    if roi_img is None or roi_img.size == 0:
        raise ValueError("ROI 不可為空")

    glare = glare_level(roi_img) if dehaze is None else None
    dehaze_used = glare >= GLARE_THRESHOLD if dehaze is None else bool(dehaze)
    source = dehaze_glare(roi_img) if dehaze_used else roi_img

    luma = channel_weighted_luma(source)
    luma = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=CLAHE_GRID).apply(luma)
    if color:
        lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        enhanced = cv2.cvtColor(cv2.merge((luma, lab[..., 1], lab[..., 2])), cv2.COLOR_LAB2BGR)
    else:
        enhanced = cv2.cvtColor(luma, cv2.COLOR_GRAY2BGR)

    info = {
        "clipped": float(np.count_nonzero(_channel_max_min(roi_img)[0] >= 250) / luma.size),
        "glare": glare,
        "dehaze": dehaze_used,
    }
    return enhanced, info


def roi_quality(roi_img: np.ndarray) -> float:
    """ROI 清晰度分數，用於同一台車在多個影格中挑最清楚的一張送 OCR。

    在通道加權亮度上計算 Laplacian 變異數（先縮放到固定高度，讓不同大小的 ROI
    可以比較），再乘上尺寸因子，偏好較大的 ROI。
    """
    if roi_img is None or roi_img.size == 0:
        return 0.0
    height, width = roi_img.shape[:2]
    luma = channel_weighted_luma(roi_img)
    scale = QUALITY_HEIGHT / height
    luma = cv2.resize(luma, (max(1, round(width * scale)), QUALITY_HEIGHT),
                      interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(luma, cv2.CV_32F).var())
    return sharpness * min(1.0, height / QUALITY_HEIGHT)


def main():
    parser = argparse.ArgumentParser(description="長曝光 LED 看板 ROI 前處理（輸出前後對照圖）")
    parser.add_argument("input", help="輸入影像或資料夾")
    parser.add_argument("-o", "--output", default=None,
                        help="輸出資料夾；預設為輸入旁的 led_preprocess_out")
    parser.add_argument("--dehaze", choices=("off", "on", "auto"), default="off",
                        help="玻璃反光去霧：off（預設）、on、auto（依 glare_level 判斷）")
    args = parser.parse_args()

    source = Path(args.input)
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")) \
        if source.is_dir() else [source]
    output_dir = Path(args.output) if args.output else \
        (source if source.is_dir() else source.parent) / "led_preprocess_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    dehaze = {"off": False, "on": True, "auto": None}[args.dehaze]

    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            print(f"略過（讀不到）: {path}")
            continue
        enhanced, info = enhance_led_roi(image, dehaze=dehaze)
        cv2.imwrite(str(output_dir / f"{path.stem}_led.png"), np.hstack((image, enhanced)))
        print(
            f"{path.name}: clipped={info['clipped']:.1%}, glare={glare_level(image):.2f}, "
            f"dehaze={info['dehaze']}, quality={roi_quality(image):.0f}"
        )
    print(f"已輸出到: {output_dir}")


if __name__ == "__main__":
    main()
