import cv2
import time
import base64
import requests
import torch
from ultralytics import YOLO
from collections import deque

# =========================
# Ollama 設定
# =========================
OLLAMA_URL = "http://192.168.0.110:11434/api/chat"
MODEL_NAME = "qwen3-vl:4B-instruct"

# =========================
# GPU / YOLO
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🔥 Device:", device)

model = YOLO("yolov8n.pt")

half = device == "cuda"

VIDEO_PATH = "bus_3.mp4"


def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def call_ollama(frame, prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "只回傳數字或NONE"},
            {"role": "user", "content": prompt, "images": [encode_frame(frame)]}
        ],
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def detect_bus(frame):
    results = model.track(
        frame,
        imgsz=(736, 1280),
        device=device,
        classes=[5],
        conf=0.5,
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


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_id = 0

    # ✅ 每輛公車獨立的 OCR buffer：{track_id: deque}
    bus_ocr_buffers = {}

    # ✅ 已確認號碼的公車：{track_id: "號碼"}
    confirmed_buses = {}

    # 顯示用：最後確認的結果
    final_result = "NONE"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame = cv2.resize(frame, (720, 1280))
        buses = detect_bus(frame)

        selected = None
        if buses:
            selected = max(buses, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

        if selected:
            x1, y1, x2, y2, conf, track_id = selected

            # 畫框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 已確認過的車 → 直接顯示結果，不再呼叫 API
            if track_id in confirmed_buses:
                label = f"BUS ID:{track_id} → {confirmed_buses[track_id]}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                final_result = confirmed_buses[track_id]

            else:
                cv2.putText(frame, f"BUS ID:{track_id} [讀取中]",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                # ✅ 每10幀才打一次 API
                if frame_id % 25 == 0:
                    w, h = x2 - x1, y2 - y1
                    roi = frame[
                        y1:y1 + int(h * 0.4),
                        x1 + int(w * 0.1):x2 - int(w * 0.1)
                    ]

                    if roi.size > 0:
                        try:
                            result = call_ollama(roi, "公車LED號碼？只回傳數字或NONE")
                            print(f"🧠 OCR [ID:{track_id}]:", result)

                            # 初始化這輛車的 buffer
                            if track_id not in bus_ocr_buffers:
                                bus_ocr_buffers[track_id] = deque(maxlen=2)

                            bus_ocr_buffers[track_id].append(result)

                            # ✅ 2次一致 → 確認，之後不再打 API
                            buf = bus_ocr_buffers[track_id]
                            if len(buf) == 1 and len(set(buf)) == 1:
                                confirmed_buses[track_id] = result
                                final_result = result
                                print(f"🎯 CONFIRMED [ID:{track_id}]:", result)

                        except Exception as e:
                            print("API錯誤:", e)

        # UI
        cv2.putText(frame, f"FINAL: {final_result}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Bus Detection", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()