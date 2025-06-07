'''
module: detect.py

This module defines PCBDetector, which loads a YOLOv8 model to detect defects
in PCB images and outputs annotated images along with defect metadata.
'''
from ultralytics import YOLO
import numpy as np
from PIL import Image


class PCBDetector:
    """
    Performs defect detection on PCB images using a YOLOv8 model.
    """
    def __init__(self, model_path: str = "models/best.onnx", device: str = "cpu"):
        """
        Initialize the PCBDetector with model path and device settings.

        :param model_path: Filesystem path to the ONNX model weights.
        :param device: Compute device identifier (e.g., 'cpu' or 'cuda').
        """
        self.model_path = model_path
        self.device = device
        self.model = None

    def load_model(self) -> None:
        """
        Load the YOLOv8 model into memory if not already loaded.
        """
        if self.model is None:
            self.model = YOLO(self.model_path)
            # Optionally enforce device: self.model.to(self.device)

    def detect_defects(self, image: Image.Image):
        """
        Detect defects in a PCB image and produce an annotated image and summary.

        :param image: A PIL Image of the PCB to analyze.
        :return: Tuple of (annotated PIL Image, list of detections, summary string).
                 - annotated image contains bounding boxes overlaid.
                 - detections is a list of dicts with keys: class, confidence, bbox.
                 - summary describes counts of each defect type.
        """
        self.load_model()
        # Convert PIL Image to numpy array for model input
        if isinstance(image, Image.Image):
            np_img = np.array(image.convert("RGB"))
        else:
            np_img = np.array(image)

        results = self.model(np_img)
        detections = []
        summary = ""

        if results and len(results) > 0:
            result = results[0]
            names = result.names  # Mapping from class IDs to names

            # Extract bounding box data
            if hasattr(result, "boxes"):
                for det in result.boxes:
                    cls_id = int(det.cls[0])
                    cls_name = names[cls_id] if names else str(cls_id)
                    cls_name = cls_name.replace("_", " ")
                    conf = float(det.conf[0])
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    detections.append({
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })

            # Summarize defect counts
            if detections:
                counts = {}
                for d in detections:
                    counts[d['class']] = counts.get(d['class'], 0) + 1
                parts = [f"{c}x {cls}" for cls, c in counts.items()]
                summary = "Detected defects: " + ", ".join(parts)
            else:
                summary = "No defects detected."

            # Generate annotated image
            try:
                annotated_array = result.plot()
                annotated_image = Image.fromarray(annotated_array)
            except Exception:
                annotated_image = image
        else:
            # No detections at all
            annotated_image = image
            summary = "No defects detected."

        return annotated_image, detections, summary