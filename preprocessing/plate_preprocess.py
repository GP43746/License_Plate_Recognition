import cv2
import numpy as np

def plate_preprocess(plate_img):
    
    if plate_img is None or plate_img.size == 0:
        return None
    
    gray=cv2.cvtColor(plate_img,cv2.COLOR_BGR2GRAY)
    
    blur=cv2.GaussianBlur(gray,(5,5),0)
    
    thresh=cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
        
    )
    
    kernel=np.ones((3,3), np.uint8)
    morph=cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    resized=cv2.resize(
        morph,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )
    
    return resized
    
    
    