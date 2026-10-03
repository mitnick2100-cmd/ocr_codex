"""V12 OCR test with live original-ROI / actual API-JPEG comparison.

Run from this directory:
  python ollama_camera_V12_preview.py --video bus_image/0819_4.mp4
  python ollama_camera_V12_preview.py --image bus_image/1_256.png
  python ollama_camera_V12_preview.py --preprocess gamma_dark

V12 mode uses V12's current gamma / CLAHE / unsharp processing, without
bilateral filtering or highlight rolloff.
gamma_dark mode uses the cleaned gamma_dark.enhance_image instead.
Both modes exclude bilateral filtering and highlight compression.
Gamma, CLAHE, sharpening and color negative each have independent CLI switches.
All are enabled by default; adaptive Gamma / CLAHE still depend on exposure.
--no-processing bypasses all four; V12 API resizing and JPEG encoding remain.
No images or results are automatically saved. Press Q or Escape to exit.
"""
import argparse
import base64
import importlib
from pathlib import Path
from queue import Empty, Queue
import threading
import time

import cv2
import numpy as np


def fit_panel(image, width=600, height=260):
    """Letterbox for display only; this never changes the API input."""
    canvas = np.zeros((height, width, 3), np.uint8)
    scale = min(width/image.shape[1], height/image.shape[0])
    w, h = max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
    x, y = (width-w)//2, (height-h)//2
    canvas[y:y+h, x:x+w] = resized
    return canvas


def preview_panel(event, mode, processing_summary=None):
    panel = np.zeros((395, 1220, 3), np.uint8)
    original, api_image = event['original'], event['api_image']
    panel[50:310, 5:605] = fit_panel(original)
    panel[50:310, 615:1215] = fit_panel(api_image)
    lines = [
        (5, 25, f"Original ROI {original.shape[1]}x{original.shape[0]}"),
        (615, 25, f"Actual API JPEG {api_image.shape[1]}x{api_image.shape[0]} | {mode}"),
        (5, 335, f"Request #{event['request_id']} | Bus ID {event['track_id']} | {event['status']}"),
        (5, 375, processing_summary or 'Preprocess: Gamma / CLAHE + Unsharp -> Color negative'),
    ]
    for x, y, text in lines:
        cv2.putText(panel, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, .6, (240,240,240), 1, cv2.LINE_AA)
    return panel


def configurable_enhance(image, engine, mode, args):
    """Independent operators; preserve each engine's default adaptive recipe."""
    if not args.processing:
        return image.copy()
    if not (args.gamma or args.clahe or args.sharpen):
        return cv2.bitwise_not(image) if args.negative else image.copy()
    luma, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    mean, contrast = float(luma.mean()), float(luma.std())
    white = engine.overexposed_ratio(luma)
    if args.gamma and args.gamma_value is not None:
        strategy = 'manual-gamma'
    elif white >= .08 or mean >= 165:
        strategy = 'overexposed'
    elif mean <= 115:
        strategy = 'dark'
    elif contrast <= 35:
        strategy = 'low-contrast'
    else:
        strategy = 'normal'
    applied = []
    if args.clahe and strategy in ('dark', 'low-contrast'):
        clip = 2.0 if strategy == 'dark' else 1.5
        luma = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8)).apply(luma)
        applied.append('CLAHE')
    gamma = None
    if args.gamma:
        if strategy == 'manual-gamma':
            gamma = args.gamma_value
        elif strategy == 'overexposed':
            options = {'max_gamma':3.0} if mode == 'gamma_dark' else {}
            gamma = engine.auto_gamma(luma, target_mean=120.0, **options)
        elif strategy == 'dark':
            options = {'min_gamma':.65} if mode == 'gamma_dark' else {}
            gamma = engine.auto_gamma(luma, target_mean=142.0, **options)
        elif strategy == 'normal' and mode == 'gamma_dark':
            gamma = 1.08
        if gamma is not None:
            luma = engine.apply_gamma(luma, gamma)
            applied.append(f'Gamma={gamma:.3f}')
    if args.sharpen:
        luma = engine.unsharp_mask(luma)
        applied.append('Unsharp')
    result = cv2.cvtColor(cv2.merge((luma,a,b)), cv2.COLOR_LAB2BGR)
    if args.negative:
        result = cv2.bitwise_not(result)
        applied.append('Negative')
    print(f'[preprocess] {strategy}: '+(' -> '.join(applied) or 'No operators applied'))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--video', help='Video path; default is current V12 VIDEO_PATH')
    inputs.add_argument('--image', help='Single already-cropped LED image; sent as a whole ROI')
    parser.add_argument('--preprocess', choices=['v12','gamma_dark'], default='v12',
                        help='Choose adaptive exposure parameters (default: v12)')
    for name, help_text in [('processing','Master switch; API resizing remains'),
                            ('gamma','Adaptive Gamma correction'),
                            ('clahe','Adaptive CLAHE contrast enhancement'),
                            ('sharpen','Unsharp sharpening'),
                            ('negative','Final color negative')]:
        parser.add_argument('--'+name, action=argparse.BooleanOptionalAction,
                            default=True, help=help_text+' (default: on)')
    parser.add_argument('--gamma-value', type=float, help='Use a fixed positive Gamma instead of adaptive Gamma')
    parser.add_argument('--url', help='Override V12 Ollama chat URL')
    parser.add_argument('--model', help='Override V12 Ollama model')
    parser.add_argument('--skip-warmup', action='store_true')
    return parser


def install_preview(v12, events):
    """Observe the actual transport, preserving V12 payload and confirmation logic."""
    local = threading.local()
    original_process = v12.process_ocr
    original_post = v12.ollama_session.post
    request_count = 0

    def process(track_id, roi_img):
        local.original = roi_img.copy()
        local.track_id = track_id
        try:
            return original_process(track_id, roi_img)
        finally:
            local.original = None

    def post(url, **kwargs):
        nonlocal request_count
        event = None
        if getattr(local, 'original', None) is not None:
            encoded = kwargs['json']['messages'][1]['images'][0]
            decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError('Could not decode actual API JPEG')
            request_count += 1
            event = dict(request_id=request_count, track_id=local.track_id,
                         original=local.original, api_image=decoded, status='Sending / awaiting API')
            events.put(event)
        began = time.monotonic()
        try:
            response = original_post(url, **kwargs)
            if event is not None:
                response.raise_for_status()
                raw = response.json()['message']['content'].strip()
                normalized = v12.normalize_ocr_result(raw)
                events.put(dict(event, status=f"Result: {normalized or 'NONE / invalid'} | {time.monotonic()-began:.2f}s"))
            return response
        except Exception as error:
            if event is not None:
                events.put(dict(event, status=f'API error: {type(error).__name__}'))
            raise

    v12.process_ocr = process
    v12.ollama_session.post = post


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.gamma_value is not None:
        if not np.isfinite(args.gamma_value) or args.gamma_value <= 0:
            parser.error('--gamma-value must be finite and greater than zero')
        if not args.gamma or not args.processing:
            parser.error('--gamma-value requires Gamma and processing to be enabled')
    # Import after argparse so --help does not load YOLO or initialize CUDA.
    v12 = importlib.import_module('ollama_camera_V12')
    if args.url:
        v12.OLLAMA_URL = args.url
    if args.model:
        v12.MODEL_NAME = args.model
    engine = importlib.import_module('gamma_dark') if args.preprocess == 'gamma_dark' else v12
    v12.enhance_roi = lambda image: configurable_enhance(image, engine, args.preprocess, args)
    enabled = [name for name in ('gamma','clahe','sharpen','negative')
               if args.processing and getattr(args,name)]
    summary = 'Enabled: '+(', '.join(enabled) if enabled else 'NONE (API resize only)')
    if args.gamma_value is not None:
        summary += f' | Gamma={args.gamma_value:g}'
    print(f'Preprocess: {args.preprocess}; URL: {v12.OLLAMA_URL}; model: {v12.MODEL_NAME}')
    print(summary)
    print('Left = original ROI; right = actual transmitted JPEG. Q / Escape exits.')
    events = Queue()
    install_preview(v12, events)
    cap = None
    try:
        if args.image:
            single = cv2.imdecode(np.frombuffer(Path(args.image).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if single is None or not v12.is_valid_roi(single):
                raise ValueError(f'Invalid LED image: {args.image}')
        else:
            cap = cv2.VideoCapture(args.video or v12.VIDEO_PATH)
            if not cap.isOpened():
                raise FileNotFoundError(f'Cannot open video: {args.video or v12.VIDEO_PATH}')
        cv2.namedWindow('V12 API Preview', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('V12 API Preview', 1220, 395)
        placeholder = np.zeros((395,1220,3), np.uint8)
        cv2.putText(placeholder, 'Waiting for an OCR request... Q / Escape to exit', (20,60),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (240,240,240), 2)
        cv2.imshow('V12 API Preview', placeholder)
        # Warmup stays off the UI thread so windows remain responsive.
        def worker():
            if not args.skip_warmup:
                v12.warmup_ollama()
            v12.ocr_worker()
        threading.Thread(target=worker, daemon=True, name='v12-preview-ocr').start()
        frame_id = 0
        finished = bool(args.image)
        if args.image:
            v12.schedule_ocr(1, single, time.monotonic())
        while True:
            if cap is not None and not finished:
                ok, frame = cap.read()
                if not ok:
                    finished = True
                    print('Video ended. Preview remains open; Q / Escape exits.')
                else:
                    frame_id += 1
                    frame = cv2.resize(frame, (1280,720))
                    ##frame = cv2.resize(frame, (720,1280))
                    buses = v12.detect_bus(frame)
                    clean_frame = frame.copy()
                    current_ids = {bus[5] for bus in buses}
                    with v12.lock:
                        for track_id in current_ids:
                            if track_id in v12.confirmed_buses:
                                v12.last_seen_frame[track_id] = frame_id
                    for x1,y1,x2,y2,confidence,track_id in buses:
                        width, height = x2-x1, y2-y1
                        if width <= 0 or height <= 0:
                            continue
                        roi, (rx1,ry1,rx2,ry2) = v12.extract_sign_roi(clean_frame,x1,y1,x2,y2)
                        # Copy before display annotations, keeping ROI free of drawn boxes.
                        roi = roi.copy()
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.rectangle(frame,(rx1,ry1),(rx2,ry2),(0,0,255),2)
                        with v12.lock:
                            confirmed = v12.confirmed_buses.get(track_id)
                        label = f'BUS ID:{track_id} -> {confirmed}' if confirmed else f'BUS ID:{track_id} [OCR]'
                        if not confirmed:
                            if not v12.is_valid_roi(roi):
                                label = f'BUS ID:{track_id} [ROI TOO SMALL]'
                            elif not 0.0 <= width/height <= 1.8:
                                label = f'BUS ID:{track_id} [ANGLE FILTER]'
                            elif frame_id % v12.OCR_FRAME_INTERVAL == 0:
                                v12.schedule_ocr(track_id, roi, time.monotonic())
                        cv2.putText(frame,label,(x1,max(25,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,.7,
                                    (0,255,0) if confirmed else (0,255,255),2)
                    routes = v12.get_visible_results(frame_id)
                    cv2.putText(frame,'BUS NUMBER: '+(','.join(routes) if routes else 'NONE'),(20,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0) if routes else (0,165,255),2)
                    cv2.imshow('Bus Detection',frame)
            while True:
                try:
                    event = events.get_nowait()
                except Empty:
                    break
                cv2.imshow('V12 API Preview',preview_panel(event,args.preprocess,summary))
            key = cv2.waitKey(10) & 0xff
            if key in (ord('q'),27):
                break
            if cv2.getWindowProperty('V12 API Preview',cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
