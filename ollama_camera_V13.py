"""V13: native-resolution crops, duplicate filtering before ByteTrack, audited OCR.

Run in conda environment `ocr`: python ollama_camera_V13.py --source bus_image/bus_20.mp4
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace
import unicodedata

import cv2
import numpy as np
import requests


ROOT = Path(__file__).resolve().parent
OLLAMA_URL = "http://192.168.0.110:11434/api/chat"
MODEL_NAME = "qwen3-vl:4B-instruct"


def merge_bus_detections(boxes: np.ndarray, enabled: bool = True) -> np.ndarray:
    """Merge highly overlapping boxes or aligned fragments of a side-view bus.

    Input/output: Nx6 xyxy, confidence, class. Small, distant buses inside a
    foreground box are retained unless their top/bottom and height agree.
    Geometry cannot distinguish every pair of occluding buses; --no-merge
    provides an explicit fallback for crowded scenes.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
    if not enabled or len(boxes) < 2:
        return boxes.copy()
    groups = list(range(len(boxes)))

    def root(index):
        while groups[index] != index:
            index = groups[index]
        return index

    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            aw, ah = a[2] - a[0], a[3] - a[1]
            bw, bh = b[2] - b[0], b[3] - b[1]
            if min(aw, ah, bw, bh) <= 0 or a[5] != b[5]:
                continue
            iw = max(0., min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0., min(a[3], b[3]) - max(a[1], b[1]))
            intersection = iw * ih
            iou = intersection / max(1., aw * ah + bw * bh - intersection)
            union_w = max(a[2], b[2]) - min(a[0], b[0])
            union_h = max(a[3], b[3]) - min(a[1], b[1])
            aligned = (
                min(ah, bh) / max(ah, bh) >= .70
                and abs(a[1] - b[1]) / max(ah, bh) <= .20
                and abs(a[3] - b[3]) / max(ah, bh) <= .15
                and ih / min(ah, bh) >= .85
            )
            aligned_side = (
                aligned and union_w / union_h >= 1.6
                and iw / min(aw, bw) >= .50
                and iou >= .28
            )
            if iou >= .65 or (aligned and iou >= .50) or aligned_side:
                groups[root(j)] = root(i)
    merged = []
    for key in sorted({root(i) for i in range(len(boxes))}):
        members = boxes[[i for i in range(len(boxes)) if root(i) == key]]
        merged.append([*members[:, :2].min(axis=0), *members[:, 2:4].max(axis=0),
                       members[:, 4].max(), members[0, 5]])
    return np.asarray(merged, dtype=np.float32).reshape(-1, 6)


def extract_sign_roi(frame: np.ndarray, box) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Copy from the clean original. Side signs sit lower than front signs."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box[:4])
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return frame[0:0, 0:0].copy(), (0, 0, 0, 0)
    side = bw / bh >= 1.5
    coords = (max(0, min(width, math.floor(x1 + bw * .02))),
              max(0, min(height, math.floor(y1 - bh * .02))),
              max(0, min(width, math.ceil(x2 - bw * .02))),
              max(0, min(height, math.ceil(y1 + bh * (.45 if side else .33)))))
    rx1, ry1, rx2, ry2 = coords
    return frame[ry1:ry2, rx1:rx2].copy(), coords


def roi_quality(roi: np.ndarray) -> dict:
    if roi.size == 0:
        return dict(valid=False, width=0, height=0, sharpness=0., contrast=0., score=0.)
    height, width = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Score at a comparable scale, before enhancement or upscaling.
    scale = min(1., 320 / max(width, height))
    gray = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    valid = width >= 48 and height >= 16 and width * height >= 1000 and contrast >= 8
    score = math.log1p(width * height) * math.log1p(sharpness) * min(contrast / 40, 1.5)
    return dict(valid=valid, width=width, height=height, sharpness=round(sharpness, 2),
                contrast=round(contrast, 2), score=round(score, 3))


def prepare_ocr_image(roi: np.ndarray, mode: str = "original") -> np.ndarray:
    """Preserve aspect ratio; enforce both dimension limits using padding.

    Default keeps the original exposure. Gamma/CLAHE cannot reconstruct clipped
    LED strokes. Optional mild enhancement is available for comparison.
    """
    if roi is None or roi.size == 0:
        raise ValueError("Empty ROI")
    height, width = roi.shape[:2]
    scale = min(max(1., 96 / min(width, height)), 1024 / max(width, height), 4.)
    resized = cv2.resize(roi, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    if mode == "mild":
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        luma = lab[:, :, 0]
        if luma.std() < 35 and np.percentile(luma, 95) < 235:
            adjusted = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4)).apply(luma)
            lab[:, :, 0] = cv2.addWeighted(luma, .6, adjusted, .4, 0)
            resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif mode != "original":
        raise ValueError(f"Unknown preparation mode: {mode}")
    pad_y, pad_x = max(0, 32 - resized.shape[0]), max(0, 32 - resized.shape[1])
    return cv2.copyMakeBorder(resized, pad_y // 2, pad_y - pad_y // 2,
                              pad_x // 2, pad_x - pad_x // 2, cv2.BORDER_REPLICATE)


def normalize_ocr_result(raw: str, route_format: str = "digits") -> str | None:
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", raw))
    pattern = r"[0-9]{1,4}" if route_format == "digits" else r"(?:紅|橘|黃|綠|藍|棕|幹線)?[0-9]{1,4}[A-Za-z]?(?:區間|區|副|延[0-9]{0,2})?"
    return compact if re.fullmatch(pattern, compact) else None


def display_route(route: str) -> str:
    """Readable ASCII labels for OpenCV; UTF-8 logs keep the original route."""
    for original, label in [('區間', '(short)'), ('幹線', 'Main'), ('紅', 'Red'), ('橘', 'Orange'),
                            ('黃', 'Yellow'), ('綠', 'Green'), ('藍', 'Blue'), ('棕', 'Brown'),
                            ('區', '(short)'), ('副', '(branch)'), ('延', '(extended)')]:
        route = route.replace(original, label)
    return route


def encode_jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return encoded.tobytes()


class OllamaClient:
    def __init__(self, url=OLLAMA_URL, model=MODEL_NAME, timeout=60., route_format="digits"):
        self.url, self.model, self.timeout, self.route_format = url, model, timeout, route_format
        self.session = requests.Session()
        self.session.trust_env = False

    def recognize(self, jpeg: bytes) -> tuple[str, str | None]:
        output = ("只回傳主線號碼的 1 至 4 位數字。忽略顏色前綴及區間、副線、延駛標記；"
                  "例如橘19回傳19，21延2回傳21，不可把延駛編號併入主線。" if self.route_format == "digits" else
                  "保留路線前綴與後綴，例如橘19、紅5、265區、21延2。只回傳完整路線號碼。"
                  "延駛標記可能上下排列，不可把21延2寫成212。")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是公車 LED 路線號碼辨識系統。" + output +
                 "無法辨識或筆畫不完整時只回傳 NONE。不可猜測；不要讀取車牌、廣告、目的地或影片字幕。"},
                {"role": "user", "content": "辨識 LED 看板上的公車路線號碼。", "images": [base64.b64encode(jpeg).decode("ascii")]},
            ],
            "stream": False, "keep_alive": "10m",
            "options": {"temperature": 0, "num_predict": 32},
        }
        response = self.session.post(self.url, json=payload, timeout=(5, self.timeout))
        response.raise_for_status()
        data = response.json()
        raw = data.get("message", {}).get("content")
        if not isinstance(raw, str):
            raise ValueError("Ollama response has no message.content string")
        return raw.strip(), normalize_ocr_result(raw, self.route_format)

    def close(self):
        self.session.close()


class BusDetector:
    def __init__(self, weights: str, fps: float, device: str = "auto", merge=True):
        import torch
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model = YOLO(weights)
        self.merge = merge
        self.tracker = BYTETracker(SimpleNamespace(track_high_thresh=.3, track_low_thresh=.1,
                                   new_track_thresh=.35, track_buffer=30,
                                   match_thresh=.8, fuse_score=True), frame_rate=round(fps))
        print(f"Device: {self.device}", flush=True)

    def detect(self, frame):
        from ultralytics.engine.results import Boxes
        result = self.model.predict(frame, imgsz=(768, 1280), device=self.device,
                                    half=self.device != "cpu", classes=[5], conf=.1,
                                    iou=.7, verbose=False)[0]
        raw = result.boxes.data.cpu().numpy()
        # Low-confidence boxes remain available to ByteTrack for recovery, but
        # must not expand the crop of a confident detection.
        high = raw[raw[:, 4] >= .3]
        filtered = merge_bus_detections(high, self.merge)
        low = raw[raw[:, 4] < .3]
        # Drop low-score detections substantially overlapping a retained box.
        kept_low = []
        for b in low:
            if len(filtered):
                overlap = np.maximum(0, np.minimum(filtered[:, 2:4], b[2:4]) - np.maximum(filtered[:, :2], b[:2]))
                if np.any(overlap.prod(axis=1) / max(1., (b[2] - b[0]) * (b[3] - b[1])) > .5):
                    continue
            kept_low.append(b)
        detections = np.concatenate((filtered, np.asarray(kept_low, dtype=np.float32).reshape(-1, 6)))
        tracks = self.tracker.update(Boxes(detections, frame.shape[:2]), frame)
        # x1,y1,x2,y2,track_id,score,class,detection_index
        return tracks, dict(raw=len(high), merged=len(filtered))


@dataclass
class Candidate:
    frame_id: int
    media_time: float
    roi: np.ndarray
    roi_box: tuple
    quality: dict
    captured_at: float = field(default_factory=time.monotonic)


@dataclass
class TrackState:
    last_seen: float
    last_frame: int
    best: Candidate | None = None
    window_start: float = 0.
    last_attempt: float = -math.inf
    last_submitted_frame: int = -1
    pending: bool = False
    votes: deque = field(default_factory=lambda: deque(maxlen=4))
    confirmed: str | None = None
    failures: int = 0
    last_sent_score: float = 0.

    def accept(self, result: str | None, confirmations: int):
        self.votes.append(result)
        self.failures = self.failures + 1 if result is None else 0
        if (result is not None and len(self.votes) >= confirmations and
                all(x == result for x in list(self.votes)[-confirmations:])):
            self.confirmed = result


class OCRPipeline:
    """One API worker, bounded replaceable waiting jobs, best clean frame per track."""
    def __init__(self, client, output: Path, interval=1., ttl=2., sample_window=.25,
                 queue_size=2, confirmations=2, mode="original", enabled=True, recheck=5.):
        self.client, self.output, self.interval, self.ttl = client, output, interval, ttl
        self.sample_window, self.queue_size, self.confirmations = sample_window, queue_size, confirmations
        self.mode, self.enabled = mode, enabled
        self.recheck = recheck
        self.states: dict[int, TrackState] = {}
        self.waiting: dict[int, tuple[TrackState, Candidate]] = {}
        self.condition = threading.Condition()
        self.media_time = 0.
        self.stopping = False
        self.thread = None
        self.sequence = 0
        self.events = []
        self.worker_error = None
        self.output.mkdir(parents=True, exist_ok=True)

    def start(self):
        if self.enabled:
            self.thread = threading.Thread(target=self._worker, name="ollama-ocr")
            self.thread.start()

    def advance(self, media_time: float):
        with self.condition:
            if self.worker_error:
                raise RuntimeError('OCR audit writer failed') from self.worker_error
            self.media_time = media_time
            for tid, state in list(self.states.items()):
                if media_time - state.last_seen > self.ttl:
                    self.states.pop(tid)
                    self.waiting.pop(tid, None)

    def observe(self, tid: int, candidate: Candidate):
        with self.condition:
            state = self.states.setdefault(tid, TrackState(candidate.media_time, candidate.frame_id,
                                                           window_start=candidate.media_time))
            state.last_seen, state.last_frame = candidate.media_time, candidate.frame_id
            if not candidate.quality['valid'] or not self.enabled:
                return
            if state.best is None or candidate.media_time - state.best.media_time > 1. or candidate.quality['score'] >= state.best.quality['score']:
                state.best = candidate
            # Waiting work can be replaced by a newer/better frame without
            # accumulating stale FIFO tasks. In-flight work is left alone.
            if tid in self.waiting:
                self.waiting[tid] = (state, state.best)
            retry_interval = self.interval * 2 ** min(state.failures, 2)
            if state.best.quality['score'] >= state.last_sent_score * 1.2:
                retry_interval = self.interval
            if state.confirmed and state.votes and state.votes[-1] == state.confirmed:
                retry_interval = self.recheck
            if state.pending or candidate.media_time - state.last_attempt < retry_interval:
                return
            if candidate.media_time - state.window_start < self.sample_window:
                return
            if len(self.waiting) >= self.queue_size:
                return
            if state.best.frame_id <= state.last_submitted_frame:
                return
            self.waiting[tid] = (state, state.best)
            state.pending = True
            state.last_attempt = candidate.media_time
            self.condition.notify()

    def result(self, tid):
        with self.condition:
            state = self.states.get(tid)
            return state.confirmed if state else None

    def visible_results(self):
        with self.condition:
            return sorted({s.confirmed for s in self.states.values()
                           if s.confirmed and self.media_time - s.last_seen <= .5})

    def _record(self, event):
        # Called only by the worker; keeps the on-disk audit even on API failure.
        self.events.append(event)
        with (self.output / 'ocr.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(event, ensure_ascii=False) + '\n')
        print(json.dumps(event, ensure_ascii=False), flush=True)

    def _worker(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.waiting or self.stopping)
                if not self.waiting:
                    return
                tid = next(iter(self.waiting))
                state, candidate = self.waiting.pop(tid)
                current = self.states.get(tid) is state
                stale = (self.media_time - candidate.media_time > self.ttl or
                         time.monotonic() - candidate.captured_at > self.ttl)
                if not current or stale:
                    state.pending = False
                    state.best = None
                    continue
                state.last_submitted_frame = candidate.frame_id
                state.last_sent_score = candidate.quality['score']
                state.best = None
            self.sequence += 1
            stem = f'{self.sequence:04d}_id{tid}_f{candidate.frame_id:06d}'
            event = dict(track_id=tid, frame=candidate.frame_id,
                         media_time=round(candidate.media_time, 3), roi_box=candidate.roi_box,
                         quality=candidate.quality, source=f'{stem}_source.png', sent=f'{stem}_sent.jpg')
            start = time.monotonic()
            try:
                prepared = prepare_ocr_image(candidate.roi, self.mode)
                jpeg = encode_jpeg(prepared)
                if not cv2.imwrite(str(self.output / event['source']), candidate.roi):
                    raise OSError("Failed to save source ROI")
                (self.output / event['sent']).write_bytes(jpeg)
                event['sent_size'] = [prepared.shape[1], prepared.shape[0]]
                raw, result = self.client.recognize(jpeg)
                event.update(raw=raw, result=result)
                with self.condition:
                    if self.states.get(tid) is state and self.media_time - state.last_seen <= self.ttl:
                        # Require agreeing distinct frames; an intervening NONE
                        # or conflicting reading breaks the confirmation streak.
                        state.accept(result, self.confirmations)
                        event['confirmed'] = state.confirmed
                    else:
                        event['discarded'] = 'track expired'
            except Exception as error:
                event['error'] = f'{type(error).__name__}: {error}'
                with self.condition:
                    state.accept(None, self.confirmations)
            finally:
                event['elapsed_seconds'] = round(time.monotonic() - start, 3)
                with self.condition:
                    state.pending = False
                try:
                    self._record(event)
                except OSError as error:
                    with self.condition:
                        self.worker_error = error
                        self.waiting.clear()
                        self.stopping = True
                    return

    def close(self):
        with self.condition:
            # No new requests after EOF. Finish and log the already active one.
            self.waiting.clear()
            self.stopping = True
            self.condition.notify_all()
        if self.thread:
            self.thread.join()
        self.client.close()
        if self.worker_error:
            raise RuntimeError('OCR audit writer failed') from self.worker_error


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(ROOT / 'bus_image' / 'bus_24.mp4'), help='Video path or camera index')
    parser.add_argument('--weights', default=str(ROOT / 'yolov8n.pt'))
    parser.add_argument('--ollama-url', default=OLLAMA_URL)
    parser.add_argument('--model', default=MODEL_NAME)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no-ocr', action='store_true')
    parser.add_argument('--no-merge', action='store_true')
    parser.add_argument('--fast', action='store_true', help='Process files without pacing; OCR still uses wall-clock network time')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--ocr-interval', type=float, default=1.)
    parser.add_argument('--recheck-seconds', type=float, default=5., help='Recheck a confirmed route after this media-time interval')
    parser.add_argument('--timeout', type=float, default=60.)
    parser.add_argument('--route-format', choices=['digits', 'full'], default='digits')
    parser.add_argument('--enhance', choices=['original', 'mild'], default='original')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.ocr_interval <= 0 or args.recheck_seconds <= 0 or args.timeout <= 0 or args.max_frames < 0:
        parser.error('interval/timeout must be positive; max-frames must be non-negative')
    return args


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open source: {args.source}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if math.isfinite(fps) and fps > 0 else 30.
    output = args.output or ROOT / 'runs_v13' / time.strftime('%Y%m%d_%H%M%S')
    if any((output / name).exists() for name in ['config.json', 'ocr.jsonl', 'detections.jsonl', 'summary.json']):
        cap.release()
        raise FileExistsError(f'Choose a fresh --output directory to preserve previous evidence: {output}')
    pipeline = None
    frame_id, removed, raw_total = 0, 0, 0
    started = time.monotonic()
    try:
        detector = BusDetector(args.weights, fps, args.device, not args.no_merge)
        client = OllamaClient(args.ollama_url, args.model, args.timeout, args.route_format)
        pipeline = OCRPipeline(client, output, interval=args.ocr_interval, mode=args.enhance,
                               enabled=not args.no_ocr, recheck=args.recheck_seconds)
        pipeline.start()
        (output / 'config.json').write_text(json.dumps(vars(args) | {'output': str(output), 'fps': fps},
                                                       default=str, indent=2), encoding='utf-8')
        playback_start = time.monotonic()
        with (output / 'detections.jsonl').open('w', encoding='utf-8') as detection_log:
            while not args.max_frames or frame_id < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id += 1
                media_time = (frame_id - 1) / fps if isinstance(source, str) else time.monotonic() - playback_start
                pipeline.advance(media_time)
                tracks, counts = detector.detect(frame)
                raw_total += counts['raw']
                removed += counts['raw'] - counts['merged']
                display = frame.copy() if not args.headless else None
                for track in tracks:
                    tid = int(track[4])
                    roi, roi_box = extract_sign_roi(frame, track)
                    quality = roi_quality(roi)
                    pipeline.observe(tid, Candidate(frame_id, media_time, roi, roi_box, quality))
                    if display is not None:
                        x1, y1, x2, y2 = map(int, track[:4])
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(display, roi_box[:2], roi_box[2:], (0, 0, 255), 2)
                        result = pipeline.result(tid)
                        label = display_route(result) if result else ('OCR' if quality['valid'] else 'WAIT DETAIL')
                        cv2.putText(display, f'ID:{tid} {label}', (max(0, x1), max(25, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
                detection_log.write(json.dumps(dict(frame=frame_id, **counts, tracks=tracks.tolist())) + '\n')
                if display is not None:
                    results = ','.join(pipeline.visible_results()) or 'NONE'
                    cv2.putText(display, 'BUS: ' + display_route(results), (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
                    scale = min(1., 1280 / display.shape[1], 800 / display.shape[0])
                    display = cv2.resize(display, (round(display.shape[1] * scale), round(display.shape[0] * scale)))
                    cv2.imshow('Bus Detection V13', display)
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break
                if isinstance(source, str) and not args.fast:
                    delay = playback_start + frame_id / fps - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                if frame_id % 120 == 0:
                    print(f'frame={frame_id} duplicate_boxes_removed={removed} routes={pipeline.visible_results()}', flush=True)
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if pipeline:
            pipeline.close()
            summary = dict(frames=frame_id, raw_boxes=raw_total, duplicate_boxes_removed=removed,
                           ocr_requests=len(pipeline.events), elapsed_seconds=round(time.monotonic() - started, 2),
                           confirmed=sorted({e['confirmed'] for e in pipeline.events if e.get('confirmed')}))
            (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
