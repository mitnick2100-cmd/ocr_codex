import base64
import re
import threading
import time
from queue import Full, Queue

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO

# =========================
# Ollama 設定
# =========================
OLLAMA_URL = "http://192.168.0.110:11434/api/chat"
MODEL_NAME = "qwen3-vl:4B-instruct"

# =========================
# GPU / YOLO
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = YOLO("yolov8n.pt")
half = device == "cuda"
VIDEO_PATH = "C:/Users/Mitnick/Desktop/FCU/VS_Code/ocr_codex/bus_image/bus_5.mp4"

# =========================
# 執行參數
# =========================
GRACE_FRAMES = 15
OCR_FRAME_INTERVAL = 5
OCR_INTERVAL_SECONDS = 1.0
OCR_QUEUE_SIZE = 2
CONFIRMATIONS_REQUIRED = 2

# 小於 32 px 的圖片會讓 qwen3-vl runner 回 500；V12 先保留小 ROI，再放大後送出。
OLLAMA_MIN_IMAGE_SIDE = 32
TARGET_OCR_MIN_SIDE = 96
MAX_OCR_SIDE = 512
MIN_SOURCE_ROI_WIDTH = 4
MIN_SOURCE_ROI_HEIGHT = 4
MIN_SOURCE_ROI_AREA = 48

# =========================
# 共用狀態
# =========================
confirmed_buses = {}
last_seen_frame = {}
last_ocr_time = {}
ocr_history = {}
ocr_pending_ids = set()
ocr_queue = Queue(maxsize=OCR_QUEUE_SIZE)
lock = threading.Lock()
ollama_session = requests.Session()
ollama_session.trust_env = False


def encode_frame(frame):
    """將 OpenCV BGR 影像編碼成 Ollama 可接受的 JPEG Base64。"""
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("無法將 ROI 編碼為 JPEG")
    return base64.b64encode(buffer).decode("utf-8")


def warmup_ollama():
    """預載入模型並保持在 VRAM，減少第一個 OCR 任務的等待時間。"""
    print("Ollama warming up...")
    # 暖機只走小尺寸全黑圖片，不使用 OCR 放大或影像增強。
    dummy = np.zeros((OLLAMA_MIN_IMAGE_SIDE, OLLAMA_MIN_IMAGE_SIDE, 3), dtype=np.uint8)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "warmup"},
            {"role": "user", "content": "OK", "images": [encode_frame(dummy)]},
        ],
        "stream": False,
        "keep_alive": -1,
        "options": {"num_predict": 1, "temperature": 0},
    }

    try:
        response = ollama_session.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        print("Ollama ready (kept alive in VRAM)")
    except requests.RequestException as error:
        print("warmup failed:", error)


# =========================
# ROI 影像增強（僅處理亮度通道）
# =========================
def build_gamma_lut(gamma: float) -> np.ndarray:
    """建立 gamma LUT。gamma > 1 變暗，gamma < 1 變亮。"""
    if gamma <= 0:
        raise ValueError("gamma 必須大於 0")
    table = np.array(
        [((value / 255.0) ** gamma) * 255 for value in range(256)],
        dtype=np.uint8,
    )
    return table


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """依 gamma 套用亮度校正；方向與 build_gamma_lut 的說明一致。"""
    return cv2.LUT(image, build_gamma_lut(gamma))


def overexposed_ratio(image: np.ndarray, threshold: int = 250) -> float:
    """計算近乎死白的亮度像素比例。"""
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(np.count_nonzero(luma >= threshold) / luma.size)


def auto_gamma(image: np.ndarray, target_mean: float = 128.0) -> float:
    """估算把平均亮度調向 target_mean 的 gamma（>1 變暗，<1 變亮）。"""
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mean_brightness = float(np.clip(luma.mean(), 1.0, 254.0))
    target = float(np.clip(target_mean, 1.0, 254.0))
    gamma = np.log(target / 255.0) / np.log(mean_brightness / 255.0)
    return float(np.clip(gamma, 0.55, 2.5))


def unsharp_mask(image: np.ndarray, sigma: float = 0.8,
                 strength: float = 0.65) -> np.ndarray:
    """低強度銳化；內部高斯模糊只用來計算銳化差值。"""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)


def enhance_roi(roi_img: np.ndarray) -> np.ndarray:
    """針對 LED 看板做條件式增強，不改變 BGR 色彩通道。

    流程：依 LAB-L 曝光狀態選擇 gamma 校正或 CLAHE -> 輕度銳化。
    不使用雙邊濾波或高光壓縮。正常曝光且對比足夠的影像不會被強制
    套用 gamma，避免損失數字與背景之間的細節。
    """
    if roi_img is None or roi_img.size == 0:
        raise ValueError("ROI 不可為空")

    lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)
    luma, channel_a, channel_b = cv2.split(lab)

    mean_luma = float(luma.mean())
    contrast = float(luma.std())
    white_ratio = overexposed_ratio(luma)
    clahe_used = False
    gamma_used = None

    if white_ratio >= 0.08 or mean_luma >= 165:
        strategy = "overexposed"
        gamma_used = auto_gamma(luma, target_mean=120.0)
        luma = apply_gamma(luma, gamma_used)
    elif mean_luma <= 115:
        strategy = "dark"
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        luma = clahe.apply(luma)
        gamma_used = auto_gamma(luma, target_mean=142.0)
        luma = apply_gamma(luma, gamma_used)
        clahe_used = True
    elif contrast <= 35:
        strategy = "low-contrast"
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        luma = clahe.apply(luma)
        clahe_used = True
    else:
        strategy = "normal"

    luma = unsharp_mask(luma)
    enhanced = cv2.cvtColor(cv2.merge((luma, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    gamma_text = f"{gamma_used:.2f}" if gamma_used is not None else "None"
    print(
        f"[enhance_roi] {strategy}, mean={mean_luma:.1f}, "
        f"contrast={contrast:.1f}, white={white_ratio:.1%}, "
        f"gamma={gamma_text}, CLAHE={clahe_used}"
    )
    return enhanced


def upscale_for_ocr(roi_img: np.ndarray) -> np.ndarray:
    """等比例放大 ROI，避免小圖觸發 500，並提升小看板 OCR 辨識率。"""
    height, width = roi_img.shape[:2]
    scale = max(
        1.0,
        OLLAMA_MIN_IMAGE_SIDE / width,
        OLLAMA_MIN_IMAGE_SIDE / height,
        TARGET_OCR_MIN_SIDE / width,
        TARGET_OCR_MIN_SIDE / height,
    )

    max_scaled_side = max(width, height) * scale
    if max_scaled_side > MAX_OCR_SIDE:
        scale = MAX_OCR_SIDE / max(width, height)

    if scale <= 1.0:
        return roi_img

    output_width = max(OLLAMA_MIN_IMAGE_SIDE, round(width * scale))
    output_height = max(OLLAMA_MIN_IMAGE_SIDE, round(height * scale))
    print(
        f"[upscale_for_ocr] {width}x{height} -> "
        f"{output_width}x{output_height}, scale={scale:.2f}"
    )
    return cv2.resize(
        roi_img,
        (output_width, output_height),
        interpolation=cv2.INTER_CUBIC,
    )


# =========================
# 看板 ROI 擷取與驗證
# =========================
def extract_sign_roi(frame: np.ndarray, x1: int, y1: int,
                     x2: int, y2: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """使用固定比例擷取車體上方 LED 看板區域。"""
    frame_height, frame_width = frame.shape[:2]
    x1, x2 = max(0, x1), min(frame_width, x2)
    y1, y2 = max(0, y1), min(frame_height, y2)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return frame[0:0, 0:0], (x1, y1, x1, y1)

    roi_box = (
        x1 + int(width * 0.06),
        y1,
        x2 - int(width * 0.06),
        y1 + int(height * 0.3),
    )
    rx1, ry1, rx2, ry2 = roi_box
    return frame[ry1:ry2, rx1:rx2], roi_box


def is_valid_roi(roi_img: np.ndarray) -> bool:
    """只過濾空圖或極端小圖；其餘小 ROI 交給 upscale_for_ocr 放大。"""
    if roi_img is None or roi_img.size == 0:
        return False
    height, width = roi_img.shape[:2]
    return (
        width >= MIN_SOURCE_ROI_WIDTH
        and height >= MIN_SOURCE_ROI_HEIGHT
        and width * height >= MIN_SOURCE_ROI_AREA
    )


def normalize_ocr_result(result: str) -> str | None:
    """只接受 1 至 4 位數字，避免模型說明文字進入投票確認。"""
    compact = re.sub(r"\s+", "", result)
    if compact.upper() == "NONE":
        return None
    return compact if re.fullmatch(r"\d{1,4}", compact) else None


# =========================
# OCR worker（受限佇列，避免多台車同時建立大量執行緒）
# =========================
def process_ocr(track_id: int, roi_img: np.ndarray):
    prepared_roi = enhance_roi(upscale_for_ocr(roi_img))
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "你是公車 LED 號碼辨識系統。只能回傳 1 至 4 位數字；無法辨識時只回傳 NONE。",
            },
            {
                "role": "user",
                "content": "辨識圖片中的公車路線號碼。",
                "images": [encode_frame(prepared_roi)],
            },
        ],
        "stream": False,
        "keep_alive": -1,
    }
    response = ollama_session.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    raw_result = response.json()["message"]["content"].strip()
    result = normalize_ocr_result(raw_result)
    print(f"OCR [ID:{track_id}] = {raw_result!r}")

    if result is None:
        return

    with lock:
        history = ocr_history.get(track_id)
        if history and history["last"] == result:
            history["count"] += 1
        else:
            history = {"last": result, "count": 1}
            ocr_history[track_id] = history

        print(f"ID:{track_id} {result} ({history['count']}/{CONFIRMATIONS_REQUIRED})")
        if history["count"] >= CONFIRMATIONS_REQUIRED:
            confirmed_buses[track_id] = result
            print(f"CONFIRMED [ID:{track_id}] {result}")


def ocr_worker():
    while True:
        track_id, roi_img = ocr_queue.get()
        try:
            process_ocr(track_id, roi_img)
        except (ValueError, cv2.error, requests.RequestException) as error:
            print(f"OCR error [ID:{track_id}]:", error)
        except Exception as error:
            print(f"Unexpected OCR error [ID:{track_id}]:", error)
        finally:
            with lock:
                ocr_pending_ids.discard(track_id)
            ocr_queue.task_done()


# =========================
# YOLO detection
# =========================
def detect_bus(frame):
    results = model.track(
        frame,
        imgsz=(768, 1280),
        device=device,
        classes=[5],
        conf=0.5,
        half=half,
        persist=True,
        verbose=False,
        tracker="bytetrack.yaml",
    )

    buses = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            if box.id is None:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            track_id = int(box.id[0])
            buses.append((x1, y1, x2, y2, confidence, track_id))
    return buses


def schedule_ocr(track_id: int, roi_img: np.ndarray, now: float) -> bool:
    """在冷卻時間、重複任務與佇列容量都允許時排入 OCR。"""
    with lock:
        if track_id in ocr_pending_ids:
            return False
        if now - last_ocr_time.get(track_id, 0.0) < OCR_INTERVAL_SECONDS:
            return False
        try:
            ocr_queue.put_nowait((track_id, roi_img.copy()))
        except Full:
            return False
        ocr_pending_ids.add(track_id)
        last_ocr_time[track_id] = now
        return True


def get_visible_results(frame_id: int) -> list[str]:
    """取得目前出現或仍在寬限期內、已確認的公車號碼。"""
    with lock:
        visible_ids = [
            track_id
            for track_id, last_frame in last_seen_frame.items()
            if frame_id - last_frame <= GRACE_FRAMES and track_id in confirmed_buses
        ]
        return [confirmed_buses[track_id] for track_id in sorted(visible_ids)]


def main():
    warmup_ollama()
    threading.Thread(target=ocr_worker, daemon=True, name="ollama-ocr-worker").start()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片: {VIDEO_PATH}")

    frame_id = 0
    aspect_ratio_min = 0.0
    aspect_ratio_max = 1.8

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame = cv2.resize(frame, (720, 1280))
        ##frame = cv2.resize(frame, (1280, 720))
        buses = detect_bus(frame)
        current_ids = {bus[5] for bus in buses}

        with lock:
            for track_id in current_ids:
                if track_id in confirmed_buses:
                    last_seen_frame[track_id] = frame_id

        for x1, y1, x2, y2, confidence, track_id in buses:
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue

            roi, (rx1, ry1, rx2, ry2) = extract_sign_roi(frame, x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

            with lock:
                confirmed_result = confirmed_buses.get(track_id)

            if confirmed_result:
                cv2.putText(
                    frame,
                    f"BUS ID:{track_id} -> {confirmed_result}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                continue

            label = f"BUS ID:{track_id} [OCR]"
            valid_roi = is_valid_roi(roi)
            aspect_ratio = width / height
            if not valid_roi:
                label = f"BUS ID:{track_id} [ROI TOO SMALL]"
            elif not aspect_ratio_min <= aspect_ratio <= aspect_ratio_max:
                label = f"BUS ID:{track_id} [ANGLE FILTER]"
            elif frame_id % OCR_FRAME_INTERVAL == 0:
                roi_height, roi_width = roi.shape[:2]
                if roi_width < TARGET_OCR_MIN_SIDE or roi_height < TARGET_OCR_MIN_SIDE:
                    label = f"BUS ID:{track_id} [ROI UPSCALE]"
                schedule_ocr(track_id, roi, time.monotonic())

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        final_results = get_visible_results(frame_id)
        display_text = ",".join(final_results) if final_results else "NONE"
        color = (255, 0, 0) if final_results else (0, 165, 255)
        cv2.putText(
            frame,
            f"BUS NUMBER: {display_text}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        cv2.imshow("Bus Detection", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
