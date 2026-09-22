import cv2
import base64
import requests
import torch
import threading
import numpy as np
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

VIDEO_PATH = "C:/Users/Mitnick/Desktop/FCU/VS_Code/ocr_codex/bus_image/bus_11.mp4"

# =========================
# 寬限幀數（公車消失後幾幀才清除顯示）
# =========================
GRACE_FRAMES = 15  # 約 1 秒，依影片 fps 調整

# =========================
# 全域變數（多執行緒）
# =========================
confirmed_buses = {}
last_seen_frame = {}
ocr_history = {}
ollama_running_ids = set()
final_results = []
lock = threading.Lock()


# =========================
# Base64 encoding
# =========================
def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


# =========================
# Ollama warmup（預載入模型）
# =========================
def warmup_ollama():
    print("Ollama warming up...")

    dummy = np.zeros((40, 40, 3), dtype=np.uint8)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "warmup"},
            {"role": "user", "content": "OK", "images": [encode_frame(dummy)]}
        ],
        "stream": False,
        "keep_alive": -1
    }

    try:
        requests.post(OLLAMA_URL, json=payload, timeout=60)
        print("Ollama ready (kept alive in VRAM)")
    except Exception as e:
        print("warmup failed:", e)

def gamma_correction(img, gamma=0.5):
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(img, table)


def unsharp_mask(img, sigma=1.0, strength=1.2):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)


def gamma_dark_unsharp(img, gamma=0.5, sigma=1.0, strength=1.2):
    img = gamma_correction(img, gamma)
    img = unsharp_mask(img, sigma, strength)
    return img


# =========================
# ⭐ 整合自 gamma_dark.py：過曝修正
# =========================
def build_gamma_lut(gamma: float) -> np.ndarray:
    """建立 0~255 的 gamma 對照表(LUT)。"""
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return table


def gamma_dark(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """gamma darkening，壓暗過曝的亮部。gamma 越大，壓暗效果越強。"""
    if gamma <= 0:
        raise ValueError("gamma 必須大於 0")
    lut = build_gamma_lut(gamma)
    return cv2.LUT(image, lut)


def highlight_rolloff(image: np.ndarray, knee: float = 0.8, strength: float = 0.6) -> np.ndarray:
    """
    高光壓縮：只針對亮度超過 knee(0~1)的區域做非線性壓縮，
    比全域gamma更能保留死白區域(接近255的大片區塊)的漸層細節。
    """
    img_f = image.astype(np.float32) / 255.0
    knee_val = knee

    mask = img_f > knee_val
    compressed = img_f.copy()
    over = (img_f - knee_val) / (1.0 - knee_val + 1e-6)
    compressed[mask] = knee_val + (1.0 - knee_val) * (over[mask] ** (1.0 + strength))

    return np.clip(compressed * 255.0, 0, 255).astype(np.uint8)


def overexposed_ratio(image: np.ndarray, threshold: int = 250) -> float:
    """回傳影像中『幾乎死白』(灰階值 >= threshold) 的像素佔比。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    overexposed = np.count_nonzero(gray >= threshold)
    return overexposed / gray.size


def auto_gamma(image: np.ndarray, target_mean: float = 128.0) -> float:
    """依照目前畫面的平均亮度，自動估算需要的gamma值，讓壓暗後的平均亮度接近 target_mean。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mean_brightness = max(gray.mean(), 1e-3)
    gamma = np.log(mean_brightness / 255.0) / np.log(target_mean / 255.0)
    return float(np.clip(gamma, 0.5, 5.0))


def fix_overexposure(image: np.ndarray, gamma: float = None,
                      use_highlight_rolloff: bool = True) -> np.ndarray:
    """
    1. 若沒有指定gamma，依照畫面平均亮度自動估算
    2. 先做高光壓縮，保留死白區域的漸層
    3. 再做gamma darkening，整體壓暗
    """
    if gamma is None:
        gamma = auto_gamma(image)

    result = image
    if use_highlight_rolloff:
        result = highlight_rolloff(result)
    result = gamma_dark(result, gamma)
    return result


# =========================
# ⭐ 自動判斷過曝 / 過暗，選擇對應的增強方式
# =========================
def adaptive_enhance(roi_img, dark_gamma=0.25, sigma=1.0, strength=1.35,
                      overexpose_threshold=0.12, target_mean=128.0):
    """
    先偵測 ROI 是否過曝(例如夜間LED反光、車頭燈直射造成死白)，
    再決定要用哪一種增強策略：
      - 過曝 -> 使用 gamma_dark.py 的高光壓縮 + gamma darkening 壓暗畫面
      - 一般夜間偏暗畫面 -> 沿用原本的 gamma brighten + unsharp mask
    最後統一做一次銳化，方便OCR辨識。
    """
    ratio = overexposed_ratio(roi_img)

    if ratio > overexpose_threshold:
        gamma = auto_gamma(roi_img, target_mean=target_mean)
        print(f"[adaptive_enhance] 偵測過曝 (死白比例 {ratio:.1%})，改用 gamma_dark，gamma={gamma:.2f}")
        result = fix_overexposure(roi_img, gamma=gamma, use_highlight_rolloff=True)
        result = unsharp_mask(result, sigma=sigma, strength=strength)
        return result
    else:
        return gamma_dark_unsharp(roi_img, gamma=dark_gamma, sigma=sigma, strength=strength)


# =========================
# 非同步 OCR worker
# =========================
def async_ollama_worker(track_id, roi_img):
    global final_result

    try:
        # 若 roi 像素總數小於 5000，長寬各放大兩倍
        if roi_img.size < 7500:
            roi_img = cv2.resize(
                roi_img,
                (roi_img.shape[1] * 2, roi_img.shape[0] * 2),
                interpolation=cv2.INTER_LINEAR
            )
            # =========================
        # ⭐ 影像增強前處理（自動判斷過曝/過暗）
        # =========================
        roi_img = adaptive_enhance(
            roi_img,
            dark_gamma=0.25,   # 夜間偏暗時使用，建議 0.4~0.7
            sigma=1.0,
            strength=1.35,
            overexpose_threshold=0.12  # 死白像素佔比超過此值視為過曝
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "只能回傳數字，若無法辨識則回傳NONE"},
                {
                    "role": "user",
                    "content": "這張圖片公車號碼是多少",
                    "images": [encode_frame(roi_img)]
                }
            ],
            "stream": False,
            "keep_alive": -1
        }

        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        result = r.json()["message"]["content"].strip()

        print(f"OCR [ID:{track_id}] = {result}")

        if result.upper() != "NONE":

            with lock:
                if track_id not in ocr_history:
                    ocr_history[track_id] = {"last": result, "count": 1}
                else:
                    if ocr_history[track_id]["last"] == result:
                        ocr_history[track_id]["count"] += 1
                    else:
                        ocr_history[track_id] = {"last": result, "count": 1}

                count = ocr_history[track_id]["count"]

                print(f"ID:{track_id} {result} ({count}/2)")

                if count >= 2:
                    confirmed_buses[track_id] = result
                    print(f"CONFIRMED [ID:{track_id}] {result}")

    except Exception as e:
        print("API錯誤:", e)

    finally:
        with lock:
            ollama_running_ids.discard(track_id)


# =========================
# YOLO detection
# =========================
def detect_bus(frame):
    results = model.track(
        frame,
        imgsz=(768, 1280),
        device=device,
        classes=[5],
        conf=0.3,
        half=half,
        persist=True,
        verbose=False,
        tracker="bytetrack.yaml"
    )

    buses = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            buses.append((x1, y1, x2, y2, conf, track_id))

    return buses


# =========================
# MAIN
# =========================
def main():

    global final_result

    warmup_ollama()

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_id = 0

    ASPECT_RATIO_MIN = 0
    ASPECT_RATIO_MAX = 1.4

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        ##frame = cv2.resize(frame, (1280, 720))                ##輸出格式##
        frame = cv2.resize(frame, (720, 1280))

        buses = detect_bus(frame)

        # 目前畫面中的 track_id
        current_ids = {b[5] for b in buses}

        # =========================
        # 更新 FINAL 結果
        # =========================
        with lock:

            for tid in current_ids:
                if tid in confirmed_buses:
                    last_seen_frame[tid] = frame_id

            visible_or_grace = {
                tid
                for tid, last_f in last_seen_frame.items()
                if (frame_id - last_f) <= GRACE_FRAMES
                and tid in confirmed_buses
            }

            if not visible_or_grace:

                final_results = []

            else:

                final_results = [
                    confirmed_buses[tid]
                    for tid in visible_or_grace
                ]

        # =========================
        # 多公車獨立處理
        # =========================
        for (x1, y1, x2, y2, conf, track_id) in buses:

            w = x2 - x1
            h = y2 - y1

            roi_x1 = x1 + int(w * 0.06)
            roi_x2 = x2 - int(w * 0.06)

            roi_y1 = y1 + int(h * 0.00)
            roi_y2 = y1 + int(h * 0.33)

            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.rectangle(
                frame,
                (roi_x1, roi_y1),
                (roi_x2, roi_y2),
                (0, 0, 255),
                2
            )

            # =========================
            # 已確認
            # =========================
            if track_id in confirmed_buses:

                result = confirmed_buses[track_id]

                cv2.putText(
                    frame,
                    f"BUS ID:{track_id} -> {result}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # =========================
            # 尚未確認
            # =========================
            else:

                cv2.putText(
                    frame,
                    f"BUS ID:{track_id} [OCR]",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                aspect_ratio = w / (h + 1e-6)

                if (
                    frame_id % 5 == 0
                    and roi.size > 5000
                    and ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX
                ):

                    with lock:

                        if track_id not in ollama_running_ids:

                            ollama_running_ids.add(track_id)

                            t = threading.Thread(
                                target=async_ollama_worker,
                                args=(track_id, roi.copy()),
                                daemon=True
                            )

                            t.start()

        # =========================
        # 顯示 FINAL
        # =========================
        if len(final_results) == 0:

            display_text = "NONE"
            color = (0, 165, 255)

        else:

            display_text = ",".join(map(str, final_results))
            color = (255, 0, 0)

        cv2.putText(
            frame,
            f"BUS NUMBER: {display_text}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )
        cv2.imshow("Bus Detection", frame)

        if cv2.waitKey(15) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
