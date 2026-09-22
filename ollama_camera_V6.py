import cv2
import base64
import requests
import torch
import threading
import numpy as np
import time
from queue import Queue
from ultralytics import YOLO

# =========================
# Ollama 設定
# =========================
OLLAMA_URL = "http://192.168.0.110:11434/api/chat"
MODEL_NAME = "qwen3-vl:4B-instruct"

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
half = device == "cuda"

print("Device:", device)

# =========================
# YOLO
# =========================
model = YOLO("yolov8n.pt")

VIDEO_PATH = "bus_1.mp4"

# =========================
# OCR 控制參數
# =========================
OCR_INTERVAL = 1   # 每台車至少間隔幾秒才允許再OCR
CONF_THRES = 0.3

# =========================
# tracking / memory
# =========================
confirmed_buses = {}
last_seen_frame = {}
last_ocr_time = {}
ocr_history = {}

GRACE_FRAMES = 90

lock = threading.Lock()

# OCR queue（避免 thread 爆炸）
ocr_queue = Queue()

# =========================
# Base64
# =========================
def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

# =========================
# Ollama worker（單線程處理）
# =========================
def ocr_worker():
    while True:
        track_id, roi = ocr_queue.get()

        try:
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "你是一個公車LED看板辨識系統"},
                    {
                        "role": "user",
                        "content": "公車號碼是多少，只回傳數字，若無法辨識則回傳NONE",
                        "images": [encode_frame(roi)]
                    }
                ],
                "stream": False,
                "keep_alive": -1
            }

            r = requests.post(OLLAMA_URL, json=payload, timeout=60)
            result = r.json()["message"]["content"].strip()

            print(f"[OCR] ID:{track_id} -> {result}")

            if result.upper() != "NONE":
                with lock:
                    if track_id not in ocr_history:
                        ocr_history[track_id] = {"last": result, "count": 1}
                    else:
                        if ocr_history[track_id]["last"] == result:
                            ocr_history[track_id]["count"] += 1
                        else:
                            ocr_history[track_id] = {"last": result, "count": 1}

                    if ocr_history[track_id]["count"] >= 2:
                        confirmed_buses[track_id] = result
                        print(f"✅ CONFIRMED ID:{track_id} -> {result}")

        except Exception as e:
            print("OCR error:", e)

        finally:
            ocr_queue.task_done()

# =========================
# YOLO detection + tracking
# =========================
def detect_buses(frame):
    results = model.track(
        frame,
        imgsz=(640, 640),
        device=device,
        classes=[5],  # bus only
        conf=CONF_THRES,
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
            if box.id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0])

            buses.append((x1, y1, x2, y2, track_id))

    return buses

# =========================
# MAIN
# =========================
def main():

    # 啟動 OCR worker（單線程）
    threading.Thread(target=ocr_worker, daemon=True).start()

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_id = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame = cv2.resize(frame, (1280, 720))

        buses = detect_buses(frame)

        current_ids = set()

        # =========================
        # 更新 tracking + FINAL logic
        # =========================
        with lock:
            for b in buses:
                x1, y1, x2, y2, track_id = b
                current_ids.add(track_id)

                if track_id in confirmed_buses:
                    last_seen_frame[track_id] = frame_id

            visible = {
                tid for tid, last_f in last_seen_frame.items()
                if (frame_id - last_f) <= GRACE_FRAMES and tid in confirmed_buses
            }

            final_result = "NONE"
            if visible:
                latest = max(visible, key=lambda tid: last_seen_frame[tid])
                final_result = confirmed_buses[latest]

        # =========================
        # multi-bus processing
        # =========================
        for b in buses:

            x1, y1, x2, y2, track_id = b

            w = x2 - x1
            h = y2 - y1

            roi_x1 = x1 + int(w * 0.1)
            roi_x2 = x2 - int(w * 0.1)
            roi_y1 = y1
            roi_y2 = y1 + int(h * 0.3)

            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # =========================
            # 已確認
            # =========================
            if track_id in confirmed_buses:

                cv2.putText(
                    frame,
                    f"BUS {track_id} -> {confirmed_buses[track_id]}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # =========================
            # 未確認 → OCR queue
            # =========================
            else:

                cv2.putText(
                    frame,
                    f"BUS {track_id} [OCR]",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                now = time.time()

                # 時間限制（避免重複OCR）
                if roi.size > 0:
                    if track_id not in last_ocr_time:
                        last_ocr_time[track_id] = 0

                    if now - last_ocr_time[track_id] > OCR_INTERVAL:

                        last_ocr_time[track_id] = now

                        ocr_queue.put((track_id, roi.copy()))

        # =========================
        # FINAL display
        # =========================
        color = (0, 165, 255) if final_result == "NONE" else (255, 0, 0)

        cv2.putText(
            frame,
            f"FINAL: {final_result}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.imshow("Bus System (Stable)", frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()