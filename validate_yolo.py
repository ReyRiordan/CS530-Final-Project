from ultralytics import YOLO

MODEL = "YOLO26n_best.pt"
DATA_YAML = "synthetic_dataset/data.yaml"

model = YOLO(MODEL)
metrics = model.val(data=DATA_YAML, imgsz=1280, device="mps")

print(f"\nmAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")