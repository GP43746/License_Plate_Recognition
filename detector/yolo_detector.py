import torch
from ultralytics import YOLO

class detector:
    def __init__(self, model_path, device="cuda", half=False):
        self.model=YOLO(model_path)
        self.device=device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        if half and self.device=="cuda":
            self.model.model.half()
            
    def detect(self, frame, confidence=0.5):
        
        results=self.model(
            frame,
            conf=confidence,
            device=self.device,
            verbose=False
        )[0]
        
        detection=[]
        
        for box in results.boxes:
            x1, y1, x2, y2=map(int, box.xyxy[0])
            conf=float(box.conf[0])
            detection.append([x1,y1,x2,y2,conf])
            
        return detection
                
        
        
        
        