# 학습된 모델로 동영상 객체 탐지 + 추적(ByteTrack), 객체별 이동 궤적(trajectory)도 함께 표시
import cv2, sys, numpy as np
from collections import defaultdict
from ultralytics import YOLO
weights, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
model = YOLO(weights)
cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(3)), int(cap.get(4))
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
track_hist = defaultdict(list); ids_seen = defaultdict(set)
while True:
    ok, frame = cap.read()
    if not ok: break
    r = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.3, imgsz=416, verbose=False)[0]
    vis = r.plot()
    if r.boxes.id is not None:
        for (x, y, w, h), tid, c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()):
            track_hist[tid].append((int(x), int(y))); track_hist[tid] = track_hist[tid][-30:]
            ids_seen[model.names[c]].add(tid)
            cv2.polylines(vis, [np.array(track_hist[tid], np.int32).reshape(-1, 1, 2)], False, (255, 255, 255), 3)
    vw.write(vis)
vw.release(); cap.release()
print({k: len(v) for k, v in ids_seen.items()})
