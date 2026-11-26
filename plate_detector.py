"""License plate detection with YOLOv8.

Usage:
    python plate_detector.py prepare --export export
    python plate_detector.py train --epochs 50
    python plate_detector.py evaluate
    python plate_detector.py predict --source path/to/image_or_video
"""

import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

DATA_DIR = Path("data")
DATA_YAML = DATA_DIR / "data.yaml"
WEIGHTS = "runs/plate_detector/weights/best.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def check_label(label_path):
    """Return a list of problems in one YOLO label file."""
    problems = []
    for i, line in enumerate(label_path.read_text().strip().splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            problems.append(f"line {i}: expected 5 values")
        elif values[0] != "0":
            problems.append(f"line {i}: class must be 0")
        elif any(not 0 <= float(v) <= 1 for v in values[1:]):
            problems.append(f"line {i}: coordinates must be between 0 and 1")
    return problems


def prepare(export_dir, val_ratio=0.2):
    """Check Label Studio labels, split into train/val and write data.yaml."""
    export_dir = Path(export_dir)
    pairs = []
    for image in sorted((export_dir / "images").iterdir()):
        label = export_dir / "labels" / (image.stem + ".txt")
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not label.exists():
            print(f"Skipping {image.name}: no label")
            continue
        problems = check_label(label)
        if problems:
            print(f"Skipping {image.name}: {problems}")
            continue
        pairs.append((image, label))

    random.Random(42).shuffle(pairs)
    val_count = int(len(pairs) * val_ratio)
    splits = {"val": pairs[:val_count], "train": pairs[val_count:]}

    for split, split_pairs in splits.items():
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image, label in split_pairs:
            shutil.copy(image, DATA_DIR / "images" / split / image.name)
            shutil.copy(label, DATA_DIR / "labels" / split / label.name)

    DATA_YAML.write_text(
        f"path: {DATA_DIR.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: license_plate\n"
    )
    print(f"{len(splits['train'])} train, {len(splits['val'])} val images")


def train(epochs, device):
    """Fine-tune a pretrained YOLOv8n model on the plate dataset."""
    model = YOLO("yolov8n.pt")
    model.train(data=str(DATA_YAML), epochs=epochs, imgsz=640,
                device=device, project="runs", name="plate_detector")


def evaluate(conf):
    """Print precision, recall and mAP on the validation set."""
    metrics = YOLO(WEIGHTS).val(data=str(DATA_YAML), conf=conf)
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")
    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")


def predict(source, conf):
    """Detect plates in an image, folder or video and save the result."""
    results = YOLO(WEIGHTS).predict(source=source, conf=conf, save=True,
                                    project="runs", name="predict")
    for result in results:
        print(f"{Path(result.path).name}: {len(result.boxes)} plate(s)")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 license plate detector")
    parser.add_argument("mode", choices=["prepare", "train", "evaluate", "predict"])
    parser.add_argument("--export", default="export", help="Label Studio YOLO export folder")
    parser.add_argument("--source", help="Image, folder or video for predict")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu", help="cpu or 0 for GPU")
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare(args.export)
    elif args.mode == "train":
        train(args.epochs, args.device)
    elif args.mode == "evaluate":
        evaluate(args.conf)
    else:
        predict(args.source, args.conf)


if __name__ == "__main__":
    main()
