from ultralytics import YOLOWorld, YOLOE
from PIL import Image
import numpy as np
import cv2

class DesktopElementDetector:
    def __init__(self, model_path="yolov8x-worldv2.pt", prompt=None):
        self.model = YOLOE("yoloe-11s-seg.pt")

        # پرامپت پیش‌فرض برای شناسایی آیکون‌ها و عناصر دسکتاپ
        if prompt is None:
            prompt = [
                "icon"
            ]
        self.set_prompt(prompt)

    def set_prompt(self, prompt):
        """
        تنظیم پرامپت (لیست کلاس‌های دلخواه برای شناسایی)
        """
        self.model.set_classes(prompt, self.model.get_text_pe(['icon']))

    def detect_elements(self, image_path, conf_threshold=0.00005):
        # بارگذاری تصویر
        image = Image.open(image_path).convert("RGB")
        # اجرای مدل با پرامپت دلخواه
        results = self.model.predict(image_path)
        elements = {}
        idx = 1
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = results[0].names[int(box.cls[0])]
            if conf < conf_threshold:
                continue
            elements[f"yoloworld_{idx}"] = [x1, y1, x2, y2, label, conf]
            idx += 1
        print(f"تعداد المان‌های شناسایی شده (YOLO-World): {len(elements)}")
        return elements