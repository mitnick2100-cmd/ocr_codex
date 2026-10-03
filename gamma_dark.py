# -*- coding: utf-8 -*-
"""公車 LED 看板影像增強工具。

此工具保留 gamma、CLAHE 與低強度銳化：

1. 僅處理 LAB 色彩空間的 L（亮度）通道，避免改變 LED 顏色。
2. 依過曝、偏暗與低對比狀態，選擇 gamma 校正或 CLAHE。
3. gamma > 1 變暗，gamma < 1 變亮；與公式及命令列參數方向一致。
4. 不使用雙邊去噪、高光壓縮或獨立高斯平滑。Unsharp 內部的高斯
   模糊只用來計算銳化差值。
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_gamma_lut(gamma: float) -> np.ndarray:
    """建立 gamma LUT；gamma > 1 變暗，gamma < 1 變亮。"""
    if gamma <= 0:
        raise ValueError("gamma 必須大於 0")
    return np.array(
        [((value / 255.0) ** gamma) * 255 for value in range(256)],
        dtype=np.uint8,
    )


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """套用 gamma 校正。"""
    return cv2.LUT(image, build_gamma_lut(gamma))


def overexposed_ratio(image: np.ndarray, threshold: int = 250) -> float:
    """回傳近乎死白的亮度像素比例。"""
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(np.count_nonzero(luma >= threshold) / luma.size)


def auto_gamma(image: np.ndarray, target_mean: float = 128.0,
               min_gamma: float = 0.55, max_gamma: float = 2.5) -> float:
    """估算把平均亮度調向 target_mean 的 gamma。"""
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mean_brightness = float(np.clip(luma.mean(), 1.0, 254.0))
    target = float(np.clip(target_mean, 1.0, 254.0))
    gamma = np.log(target / 255.0) / np.log(mean_brightness / 255.0)
    return float(np.clip(gamma, min_gamma, max_gamma))


def unsharp_mask(image: np.ndarray, sigma: float = 0.8,
                 strength: float = 0.65) -> np.ndarray:
    """低強度銳化；內部模糊是銳化參考，不是獨立平滑步驟。"""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)


def merge_luma(luma: np.ndarray, channel_a: np.ndarray,
               channel_b: np.ndarray) -> np.ndarray:
    """將處理後的 L 通道與原始色彩通道合併回 BGR。"""
    lab = cv2.merge((luma, channel_a, channel_b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def fix_overexposure(image: np.ndarray, gamma: float | None = None,
                      target_mean: float = 120.0) -> np.ndarray:
    """只在亮度通道進行 gamma 校正。

    未指定 gamma 時會自動估算，使結果趨近 target_mean。
    """
    if image is None or image.size == 0:
        raise ValueError("影像不可為空")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luma, channel_a, channel_b = cv2.split(lab)
    if gamma is None:
        gamma = auto_gamma(luma, target_mean=target_mean)
    return merge_luma(apply_gamma(luma, gamma), channel_a, channel_b)


def enhance_image(image: np.ndarray, gamma: float | None = None,
                  use_clahe: bool = True,
                  use_sharpen: bool = True,
                  target_mean: float = 128.0) -> tuple[np.ndarray, dict]:
    """依影像狀態選擇合適的 LED 看板增強策略。

    若明確指定 gamma，會直接使用該值，方便人工調校與比較。未指定時，
    依過曝比例、亮度與局部對比在 overexposed、dark、low-contrast、normal
    四種策略間選擇。
    """
    if image is None or image.size == 0:
        raise ValueError("影像不可為空")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luma, channel_a, channel_b = cv2.split(lab)
    mean_luma = float(luma.mean())
    contrast = float(luma.std())
    white_ratio = overexposed_ratio(luma)
    applied_gamma = gamma
    clahe_used = False

    if gamma is not None:
        strategy = "manual-gamma"
        luma = apply_gamma(luma, gamma)
    elif white_ratio >= 0.08 or mean_luma >= 165:
        strategy = "overexposed"
        applied_gamma = auto_gamma(
            luma,
            target_mean=min(target_mean, 120.0),
            max_gamma=3.0,
        )
        luma = apply_gamma(luma, applied_gamma)
    elif mean_luma <= 115:
        strategy = "dark"
        if use_clahe:
            luma = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luma)
            clahe_used = True
        applied_gamma = auto_gamma(
            luma,
            target_mean=max(target_mean, 142.0),
            min_gamma=0.65,
        )
        luma = apply_gamma(luma, applied_gamma)
    elif contrast <= 35 and use_clahe:
        strategy = "low-contrast"
        luma = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(luma)
        clahe_used = True
    else:
        strategy = "normal"
        applied_gamma = 1.08
        luma = apply_gamma(luma, applied_gamma)

    if use_sharpen:
        luma = unsharp_mask(luma)

    metadata = {
        "strategy": strategy,
        "mean_luma": mean_luma,
        "contrast": contrast,
        "overexposed_ratio": white_ratio,
        "gamma": applied_gamma,
        "clahe": clahe_used,
        "sharpen": use_sharpen,
    }
    return merge_luma(luma, channel_a, channel_b), metadata


def main():
    parser = argparse.ArgumentParser(description="公車 LED 看板影像增強工具")
    parser.add_argument("input", help="輸入影像路徑")
    parser.add_argument("-o", "--output", default=None, help="輸出影像路徑；未指定時自動存成 *_gamma_dark.png")
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="手動 gamma 值；>1 變暗、<1 變亮。不指定則自動選擇策略。",
    )
    parser.add_argument("--target-mean", type=float, default=128.0,
                        help="自動 gamma 的目標平均亮度（預設 128）")
    parser.add_argument("--no-clahe", action="store_true", help="關閉局部對比增強")
    parser.add_argument("--no-sharpen", action="store_true", help="關閉最後的輕度銳化")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}_gamma_dark.png"
    )

    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"讀不到影像: {input_path}")

    result, metadata = enhance_image(
        image,
        gamma=args.gamma,
        use_clahe=not args.no_clahe,
        use_sharpen=not args.no_sharpen,
        target_mean=args.target_mean,
    )
    if not cv2.imwrite(str(output_path), result):
        raise OSError(f"無法寫入輸出影像: {output_path}")

    print(
        f"已輸出: {output_path}\n"
        f"策略: {metadata['strategy']} | 平均亮度: {metadata['mean_luma']:.1f} | "
        f"對比: {metadata['contrast']:.1f} | 死白比例: {metadata['overexposed_ratio']:.1%} | "
        f"gamma: {metadata['gamma']} | CLAHE: {metadata['clahe']} | "
        f"Sharpen: {metadata['sharpen']}"
    )


if __name__ == "__main__":
    main()



