# License Plate Detection with YOLOv8

A custom detector that finds license plates in images and videos.
I labeled the images in Label Studio, trained a YOLOv8 model and evaluated it.

## Steps

1. Label car images in Label Studio with one class: `license_plate`.
2. Export in YOLO format and unzip it into `export/`.
3. Check the labels, split into train and val, and train a pretrained YOLOv8n model.
4. Evaluate with precision, recall and mAP.
5. Run the model on new images or videos.

The dataset is not included in this repo.

## Usage

```bash
pip install ultralytics
python plate_detector.py prepare --export export
python plate_detector.py train --epochs 50
python plate_detector.py evaluate
python plate_detector.py predict --source my_car.jpg
```

Use `--device 0` to train on a GPU. Results are saved in `runs/`.

## Results

| Metric    | Value |
|-----------|-------|
| Precision | TODO  |
| Recall    | TODO  |
| mAP50     | TODO  |

## Tools

Python, Ultralytics YOLOv8, PyTorch, OpenCV, Label Studio
