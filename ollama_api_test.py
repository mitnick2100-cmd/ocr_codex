import requests
import base64
import os
import time

# Ollama API
OLLAMA_URL = "http://192.168.0.110:11434/api/chat"

# 模型名稱
MODEL_NAME = "qwen3-vl:4B-instruct"


def encode_image(image_path):
    """讀取圖片並轉 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image(image_path, prompt):
    image_data = encode_image(image_path)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "你是公車 LED 號碼辨識系統。只能回傳 1 至 4 位數字；無法辨識時只回傳 NONE。"
            },
            {
                "role": "user",
                "content": prompt,
                "images": [image_data],
                "keep_alive": "-1m"
            }
        ],
        "stream": False
    }

    # 🔥 只計 API 時間（含推理）
    start = time.time()
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    end = time.time()

    print(f"\n⏱️ API時間（含推理）：{end - start:.2f} 秒")

    return response.json()["message"]["content"]


def main():
    image_path = input("圖片路徑: ")
    prompt = "辨識圖片中的公車路線號碼。"

    if not os.path.exists(image_path):
        print("找不到圖片")
        return

    try:
        result = analyze_image(image_path, prompt)
        print("\n=== 分析結果 ===")
        print(result)

    except Exception as e:
        print(f"錯誤: {e}")


if __name__ == "__main__":
    main()