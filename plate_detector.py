"""License plate detection with YOLOv8.

Dataset: Kaggle "Car License Plate Detection" (433 images, Pascal VOC XML boxes).

Usage:
    python plate_detector.py prepare --raw car_plate_raw
    python plate_detector.py train --epochs 30
    python plate_detector.py evaluate
    python plate_detector.py predict --source path/to/image_or_video
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from ultralytics import YOLO

DATA_DIR = Path("data")
DATA_YAML = DATA_DIR / "data.yaml"
RUNS_DIR = Path("runs").resolve()
WEIGHTS = RUNS_DIR / "plate_detector" / "weights" / "best.pt"


def voc_to_yolo(xml_path):
    """Convert one Pascal VOC XML file into YOLO label lines.

    YOLO format: class x_center y_center width height (all between 0 and 1).
    """
    root = ET.parse(xml_path).getroot()
    width = float(root.find("size/width").text)
    height = float(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(box.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        if xmax <= xmin or ymax <= ymin:
            print(f"Skipping broken box in {xml_path.name}")
            continue
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        box_w = (xmax - xmin) / width
        box_h = (ymax - ymin) / height
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
    return lines


def prepare(raw_dir, val_ratio=0.2):
    """Convert labels to YOLO, split 80/20 into train/val and write data.yaml."""
    raw_dir = Path(raw_dir)
    pairs = []
    for xml_path in sorted((raw_dir / "annotations").glob("*.xml")):
        image_path = raw_dir / "images" / (xml_path.stem + ".png")
        lines = voc_to_yolo(xml_path)
        if image_path.exists() and lines:
            pairs.append((image_path, lines))

    random.Random(42).shuffle(pairs)
    val_count = int(len(pairs) * val_ratio)
    splits = {"val": pairs[:val_count], "train": pairs[val_count:]}

    for split, split_pairs in splits.items():
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_path, lines in split_pairs:
            shutil.copy(image_path, DATA_DIR / "images" / split / image_path.name)
            label_path = DATA_DIR / "labels" / split / (image_path.stem + ".txt")
            label_path.write_text("\n".join(lines) + "\n")

    DATA_YAML.write_text(
        f"path: {DATA_DIR.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: license_plate\n"
    )
    boxes = sum(len(lines) for _, lines in pairs)
    print(f"{len(pairs)} images, {boxes} plates")
    print(f"{len(splits['train'])} train, {len(splits['val'])} val images")


def train(epochs, device, imgsz):
    """Fine-tune a COCO-pretrained YOLOv8n model on the plate dataset."""
    model = YOLO("yolov8n.pt")
    model.train(data=str(DATA_YAML), epochs=epochs, imgsz=imgsz, batch=16,
                device=device, project=str(RUNS_DIR), name="plate_detector", exist_ok=True)


def evaluate(conf, imgsz):
    """Print precision, recall and mAP on the validation set."""
    metrics = YOLO(WEIGHTS).val(data=str(DATA_YAML), conf=conf, imgsz=imgsz)
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")
    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")


def predict(source, conf):
    """Detect plates in an image, folder or video and save the result."""
    results = YOLO(WEIGHTS).predict(source=source, conf=conf, save=True,
                                    project=str(RUNS_DIR), name="predict", exist_ok=True)
    for result in results:
        print(f"{Path(result.path).name}: {len(result.boxes)} plate(s)")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 license plate detector")
    parser.add_argument("mode", choices=["prepare", "train", "evaluate", "predict"])
    parser.add_argument("--raw", default="car_plate_raw", help="Folder with images/ and annotations/")
    parser.add_argument("--source", help="Image, folder or video for predict")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu", help="cpu or 0 for GPU")
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare(args.raw)
    elif args.mode == "train":
        train(args.epochs, args.device, args.imgsz)
    elif args.mode == "evaluate":
        evaluate(args.conf, args.imgsz)
    else:
        predict(args.source, args.conf)


if __name__ == "__main__":
    main()
