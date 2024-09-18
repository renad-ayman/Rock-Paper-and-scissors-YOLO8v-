# Rock-Paper-Scissors Detection using YOLOv8

This repository contains code to train a YOLOv8 model for detecting hand gestures in the Rock-Paper-Scissors game.

## Features
- Utilizes Roboflow API to fetch Rock-Paper-Scissors dataset.
- Trains YOLOv8 model for detection.
- Provides evaluation and prediction scripts for real-time detection.

## Installation

1. Install required packages:
    ```bash
    pip install roboflow ultralytics
    ```

2. Download dataset and initialize Roboflow:
    ```python
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("workspace-name").project("rock-paper-scissors-sxsw")
    version = project.version(14)
    dataset = version.download("yolov8")
    ```

## Model Training

- Train the model using the downloaded dataset:
    ```bash
    yolo task=detect mode=train model=yolov8n.pt data='data.yaml' epochs=10 imgsz=640
    ```

## Evaluation

- Evaluate the model:
    ```bash
    yolo task=detect mode=val model=/content/runs/detect/train2/weights/best.pt data=/content/data.yaml
    ```

## Prediction

- Predict on new images:
    ```bash
    yolo task=detect mode=predict model=/content/runs/detect/train2/weights/best.pt conf=0.5 source=/content/test/images
    ```

## Visualize Results

- Display the confusion matrix and results:
    ```python
    from IPython.display import Image
    Image(filename='/content/runs/detect/train2/confusion_matrix.png')
    Image(filename='/content/runs/detect/train2/results.png')
    ```

## Inference Results

- Display prediction results on images:
    ```python
    import glob
    from IPython.display import display, Image

    for image_path in glob.glob('/content/runs/detect/predict/*.jpg')[:20]:
        display(Image(filename=image_path, height=600))
        print('\n')
    ```

## License

This project is licensed under the MIT License.

---
