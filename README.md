# Bus LED Route Number Recognition

A Python prototype for detecting buses and recognizing route numbers displayed on their LED signs. It combines YOLOv8 bus detection with image-based OCR using an Ollama vision-language model.

## Overview

The project explores two workflows:

- **Video-based testing:** Detect buses in a video, track them across frames, crop the LED sign area, and send it to Ollama for recognition.
- **Live camera recognition:** Run the pipeline with an NVIDIA Jetson and an IMX519 camera. The live version also supports camera controls and autofocus.

The OCR pipeline enhances the sign crop and accepts route numbers containing one to four digits. A result is confirmed after repeated matching OCR results to help reduce unstable predictions.

## Features

- Bus detection and tracking with YOLOv8 and ByteTrack
- LED sign region cropping and image enhancement
- OCR requests through the Ollama chat API
- Video-file and live-camera workflows
- Headless operation for SSH sessions
- Automatic and manual image enhancement with gamma correction, CLAHE, and sharpening

## Project Structure

| File | Description |
| --- | --- |
| `main_video_test.py` | Detects and recognizes bus numbers in a video file |
| `IMX519/main.py` | Live Jetson IMX519 recognition with a display window |
| `IMX519/nohead_main.py` | Headless live recognition with terminal controls and optional recording |
| `IMX519/nohead_onetime_main.py` | Headless variant that confirms a result after one OCR response |
| `IMX519/yolo.py` | YOLO detection and camera, exposure, and focus control example |
| `gamma_dark.py` | Command-line LED sign image enhancement tool |
| `ollama_api_test.py` | Sends one image to Ollama for a recognition test |
| `IMX519/` | Camera, focuser, and autofocus support modules |
| `yolov8n.pt` | YOLOv8n model weights |

## Requirements

- Python 3.10 or later
- OpenCV, NumPy, Requests, PyTorch, and Ultralytics
- A running Ollama server with the configured vision-language model
- For live camera scripts: an NVIDIA Jetson with a compatible IMX519 camera and focuser setup

Install PyTorch and the camera dependencies using packages compatible with your operating system and JetPack version.

## Setup

1. Clone this repository.
2. Install the Python dependencies listed above.
3. Start Ollama and make sure the configured model is available:

   ```bash
   ollama pull qwen3-vl:4B-instruct
