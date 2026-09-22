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

VIDEO_PATH = "bus_11.mp4"

# =========================
# 全域變數（多執行緒）
# =========================
confirmed_buses = {}
ocr_history = {}
ollama_running_ids = set()
final_result = "NONE"
lock = threading.Lock()


# =========================
# Base64 encoding
# =========================
def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


# =========================
# 🔥 Ollama warmup（預載入模型）
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


# =========================
# 🔥 非同步 OCR worker
# =========================
def async_ollama_worker(track_id, roi_img):
    global final_result

    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "你是一個公車LED看板辨識系統。"},
                {
                    "role": "user",
                    "content": "公車號碼是多少，只回傳數字，若無辨識則回傳NONE",
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
                # OCR history（保留第一版邏輯）
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
                    final_result = result
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
        conf=0.2,
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

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame = cv2.resize(frame, (1280, 720))

        buses = detect_bus(frame)

        selected = None
        if buses:
            selected = max(buses, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

        if selected:

            x1, y1, x2, y2, conf, track_id = selected

            w = x2 - x1
            h = y2 - y1

            roi_x1 = x1 + int(w * 0.1)
            roi_x2 = x2 - int(w * 0.1)

            roi_y1 = y1
            roi_y2 = y1 + int(h * 0.3)

            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)

            # =========================
            # 已確認
            # =========================
            if track_id in confirmed_buses:

                result = confirmed_buses[track_id]
                final_result = result

                cv2.putText(frame,
                            f"BUS ID:{track_id} -> {result}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            # =========================
            # 未確認 → 非同步 OCR
            # =========================
            else:

                cv2.putText(frame,
                            f"BUS ID:{track_id} [OCR]",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                if frame_id % 5 == 0 and roi.size > 6000:

                    with lock:
                        if track_id in ollama_running_ids:
                            pass
                        else:
                            ollama_running_ids.add(track_id)

                            roi_copy = roi.copy()

                            t = threading.Thread(
                                target=async_ollama_worker,
                                args=(track_id, roi_copy),
                                daemon=True
                            )
                            t.start()

        cv2.putText(frame,
                    f"FINAL: {final_result}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2)

        cv2.imshow("Bus Detection", frame)

        if cv2.waitKey(15) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
