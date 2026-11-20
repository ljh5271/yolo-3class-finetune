from ultralytics import YOLO
model = YOLO("yolov8n.pt")          # COCO 사전학습 가중치 (전이학습)
model.train(data="data.yaml",  # 3개 부류: elephant, rhino, zebra
            epochs=30, imgsz=416, batch=16, device="cpu", workers=2,
            project="runs", name="wild3", exist_ok=True,
            seed=0, plots=True, amp=False)
