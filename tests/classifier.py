from ultralytics import YOLO
from ultralytics.utils import LOGGER
import torch

LOGGER.setLevel(50)

class ObjectClassifier:
    def __init__(self, weights_path="yolov8n.pt", conf_threshold=0.5 , allowed_classes={"person", "cat"}):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(weights_path)
        self.model.to(self.device)         

        self.conf_threshold = conf_threshold
        self.allowed_classes = set(allowed_classes) if allowed_classes else None

    def detect(self, frame):
        results = self.model(frame, device=self.device)[0]
        chosen_detection = None
        for box in results.boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            label = self.model.names[cls_id]

            if score >= self.conf_threshold and label in self.allowed_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                chosen_detection = {
                    "label": label,
                    "score": score,
                    "bbox": (int(x1), int(y1), int(x2), int(y2))
                }
                break
        return chosen_detection
