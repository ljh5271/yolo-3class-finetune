# test 이미지(학습에 쓰지 않은 데이터)로 카메라 패닝/줌 효과를 준 5초 영상 생성 (30fps x 150프레임)
import cv2, os, sys, numpy as np
root, out = sys.argv[1], sys.argv[2]
L, I = f"{root}/labels/test", f"{root}/images/test"
best = {}
for lf in os.listdir(L):
    rows = [list(map(float, l.split())) for l in open(f"{L}/{lf}") if l.strip()]
    cls = {int(r[0]) for r in rows}
    if len(cls) != 1: continue
    c = cls.pop(); area = max(r[3] * r[4] for r in rows)
    if 0.15 < area < 0.6 and 1 <= len(rows) <= 4:
        best.setdefault(c, []).append((len(rows), area, lf[:-4]))
W, H, FPS, N = 960, 540, 30, 50
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for c in (0, 1, 2):
    n, area, stem = sorted(best[c], key=lambda t: (-t[0], -t[1]))[0]
    img = cv2.imread(f"{I}/{stem}.jpg"); print(c, stem, n, round(area, 2))
    h, w = img.shape[:2]; s = max(W / w, H / h) * 1.25; img = cv2.resize(img, None, fx=s, fy=s); h, w = img.shape[:2]
    for i in range(N):
        t = i / (N - 1); z = 1.0 - 0.15 * t                      # 서서히 줌인
        cw, ch = int(W / 1.25 * 1.25 * z), int(H * z)
        x = int((w - cw) * (0.1 + 0.8 * t)); y = int((h - ch) * 0.5)  # 좌→우 패닝
        vw.write(cv2.resize(img[y:y + ch, x:x + cw], (W, H)))
vw.release()
