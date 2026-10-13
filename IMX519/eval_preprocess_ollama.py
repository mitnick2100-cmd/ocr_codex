# -*- coding: utf-8 -*-
"""用 qwen3-vl 實測 ROI 前處理：V13 enhance_roi vs. led_roi_preprocess。

使用 bus_image/num_*.png 與下方人工標註的路線號碼，對每種前處理送一次 OCR，
統計正確率。提示詞與 payload 與 ollama_camera_V13.process_ocr 相同，另外固定
temperature=0 以便重現。

  python eval_preprocess_ollama.py                  # 原始 ROI
  python eval_preprocess_ollama.py --degrade        # 再加上過曝、失焦、縮小、雜訊、JPEG
  python eval_preprocess_ollama.py --dry-run        # 不連 Ollama，只輸出送出的影像
"""

import argparse
import base64
import re
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from led_roi_preprocess import enhance_led_roi

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = SCRIPT_DIR.parent / "bus_image"

OLLAMA_URL = "http://192.168.0.110:11434/api/chat"
MODEL_NAME = "qwen3-vl:4B-instruct"
OLLAMA_NUM_CTX = 1024
OLLAMA_MIN_IMAGE_SIDE = 32

# 人工標註（只取數字部分，例如「橘19」記為 19）。num_35 數字被反光截斷，不列入。
LABELS = {
    1: "310", 2: "266", 3: "590", 4: "716", 5: "616", 6: "21", 7: "650", 8: "75",
    9: "333", 10: "309", 11: "325", 12: "932", 13: "226", 14: "39", 15: "39",
    16: "79", 17: "79", 18: "54", 19: "54", 20: "33", 21: "33", 22: "54", 23: "54",
    24: "54", 25: "63", 26: "54", 27: "54", 28: "880", 29: "37", 30: "265",
    31: "508", 32: "226", 33: "21", 34: "21", 36: "19", 37: "286", 38: "303",
    39: "226", 40: "617", 41: "265", 42: "265", 43: "265", 44: "265", 45: "265",
    46: "508", 47: "508", 48: "19", 49: "639", 50: "9", 51: "62", 52: "286",
}


# ---- 以下兩個函式複製自 ollama_camera_V13.py（V13 會載入 Jetson 專用模組，無法直接 import）
def build_gamma_lut(gamma: float) -> np.ndarray:
    return np.array([((value / 255.0) ** gamma) * 255 for value in range(256)], dtype=np.uint8)


def v13_enhance_roi(roi_img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)
    luma, channel_a, channel_b = cv2.split(lab)
    mean_luma, contrast = float(luma.mean()), float(luma.std())
    white_ratio = float(np.count_nonzero(luma >= 250) / luma.size)

    def auto_gamma(image, target_mean):
        mean_brightness = float(np.clip(image.mean(), 1.0, 254.0))
        gamma = np.log(target_mean / 255.0) / np.log(mean_brightness / 255.0)
        return float(np.clip(gamma, 0.55, 2.5))

    if white_ratio >= 0.08 or mean_luma >= 165:
        luma = cv2.LUT(luma, build_gamma_lut(auto_gamma(luma, 120.0)))
    elif mean_luma <= 115:
        luma = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luma)
        luma = cv2.LUT(luma, build_gamma_lut(auto_gamma(luma, 142.0)))
    elif contrast <= 35:
        luma = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(luma)
    blurred = cv2.GaussianBlur(luma, (0, 0), 0.8)
    luma = cv2.addWeighted(luma, 1.65, blurred, -0.65, 0)
    return cv2.cvtColor(cv2.merge((luma, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
# ---- 複製結束


def upscale_for_ocr(roi_img: np.ndarray, target_min_side: int, max_side: int) -> np.ndarray:
    """與 V13 upscale_for_ocr 相同，但目標尺寸可調。"""
    height, width = roi_img.shape[:2]
    scale = max(1.0, OLLAMA_MIN_IMAGE_SIDE / width, OLLAMA_MIN_IMAGE_SIDE / height,
                target_min_side / width, target_min_side / height)
    if max(width, height) * scale > max_side:
        scale = max_side / max(width, height)
    if scale <= 1.0:
        return roi_img
    size = (max(OLLAMA_MIN_IMAGE_SIDE, round(width * scale)),
            max(OLLAMA_MIN_IMAGE_SIDE, round(height * scale)))
    return cv2.resize(roi_img, size, interpolation=cv2.INTER_CUBIC)


PIPELINES = {
    # 名稱: (目標最短邊, 最長邊上限, 增強函式)
    "V13": (96, 512, v13_enhance_roi),
    "LED": (96, 512, lambda img: enhance_led_roi(img)[0]),
    "LED_gray": (96, 512, lambda img: enhance_led_roi(img, color=False)[0]),
    "LED_up192": (192, 1024, lambda img: enhance_led_roi(img)[0]),
}


def degrade(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """模擬實際情況：再過曝 2 倍、失焦、縮小到 36~60 px 高、雜訊、JPEG 70。"""
    height, width = image.shape[:2]
    linear = (image.astype(np.float32) / 255.0) ** 2.2
    image = np.clip(linear * 2.0, 0, 1) ** (1 / 2.2) * 255
    image = cv2.GaussianBlur(image, (0, 0), max(1.0, height / 45))
    new_height = int(rng.integers(36, 61))
    image = cv2.resize(image, (int(width * new_height / height), new_height),
                       interpolation=cv2.INTER_AREA)
    image = np.clip(image + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def ask_ollama(session: requests.Session, image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("無法將 ROI 編碼為 JPEG")
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system",
             "content": "你是公車 LED 號碼辨識系統。只能回傳 1 至 4 位數字；無法辨識時只回傳 NONE。"},
            {"role": "user", "content": "辨識圖片中的公車路線號碼。",
             "images": [base64.b64encode(buffer).decode("utf-8")]},
        ],
        "stream": False,
        "keep_alive": -1,
        "options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0},
    }
    response = session.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def normalize(result: str) -> str | None:
    compact = re.sub(r"\s+", "", result)
    return compact if re.fullmatch(r"\d{1,4}", compact) else None


def main():
    parser = argparse.ArgumentParser(description="以 qwen3-vl 比較 ROI 前處理的辨識正確率")
    parser.add_argument("--degrade", action="store_true", help="套用實際情況劣化（固定亂數種子）")
    parser.add_argument("--dry-run", action="store_true", help="不呼叫 Ollama，只輸出送出的影像")
    parser.add_argument("--pipelines", default=",".join(PIPELINES),
                        help=f"要比較的前處理，逗號分隔（可選 {','.join(PIPELINES)}）")
    parser.add_argument("--out", default=str(SCRIPT_DIR / "eval_out"), help="輸出資料夾")
    args = parser.parse_args()

    names = [name.strip() for name in args.pipelines.split(",") if name.strip()]
    unknown = [name for name in names if name not in PIPELINES]
    if unknown:
        parser.error(f"未知的前處理: {unknown}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    session = requests.Session()
    session.trust_env = False
    correct = {name: 0 for name in names}
    elapsed = {name: 0.0 for name in names}
    rows = []

    for index, label in LABELS.items():
        path = IMAGE_DIR / f"num_{index}.png"
        roi = cv2.imread(str(path))
        if roi is None:
            print(f"略過（讀不到）: {path}")
            continue
        if args.degrade:
            roi = degrade(roi, rng)
        answers = []
        for name in names:
            target, max_side, enhance = PIPELINES[name]
            prepared = enhance(upscale_for_ocr(roi, target, max_side))
            if args.dry_run:
                cv2.imwrite(str(out_dir / f"num_{index}_{name}.png"), prepared)
                answers.append("-")
                continue
            start = time.perf_counter()
            try:
                raw = ask_ollama(session, prepared)
            except (requests.RequestException, ValueError) as error:
                raw = f"ERROR: {error}"
            elapsed[name] += time.perf_counter() - start
            result = normalize(raw)
            correct[name] += result == label
            answers.append(f"{raw!r}{' ✓' if result == label else ''}")
        rows.append(f"num_{index:<3d} {label:>4s} | " + " | ".join(answers))
        print(rows[-1])

    if args.dry_run:
        print(f"已輸出送出的影像到: {out_dir}")
        return
    total = len(rows)
    summary = [f"\n=== {'degraded' if args.degrade else 'original'} ROI, n={total} ==="]
    for name in names:
        summary.append(f"{name:10s} 正確 {correct[name]:3d}/{total} = {correct[name] / max(total, 1):6.1%}"
                       f"   平均 {elapsed[name] / max(total, 1):.2f} s/張")
    print("\n".join(summary))
    report = out_dir / f"result_{'degraded' if args.degrade else 'original'}.txt"
    report.write_text("\n".join(rows + summary), encoding="utf-8")
    print(f"結果已存到: {report}")


if __name__ == "__main__":
    main()
