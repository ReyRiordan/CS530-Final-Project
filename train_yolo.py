from ultralytics import YOLO


# ----- Config -----

# Pretrained base model from https://docs.ultralytics.com/models/yolo26/
MODEL = "yolo26n.pt"

# Dataset config
DATA_YAML = "synthetic_dataset/data.yaml"

EPOCHS = 100
IMAGE_SIZE = 1280
BATCH_SIZE = 16

# Praying Apple silicon thing works (otherwise 0 for GPU and cpu for CPU)
DEVICE = "mps"

# Run name (appears under runs/detect/)
NAME = f"troop_detect_{MODEL.replace('.pt', '')}"


# True removes NMS: https://docs.ultralytics.com/guides/end2end-detection/#how-end-to-end-detection-works
END2END = False


# ----------


def main() -> None:
    model = YOLO(MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        name=NAME,
        end2end=END2END,
        # Data augmentation: https://docs.ultralytics.com/guides/yolo-data-augmentation/
    )

    print(f"\n\nTraining complete. Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
