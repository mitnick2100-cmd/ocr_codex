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

VIDEO_PATH = "bus_7.mp4"

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
        # ⭐ 新增：影像增強前處理
        # =========================
        roi_img = gamma_dark_unsharp(
            roi_img,
            gamma=0.25,      # 夜間 LED 建議 0.4~0.7
            sigma=1.0,
            strength=1.35
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

            roi_y1 = y1 + int(h * 0.05)
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
