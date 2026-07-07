import cv2
import numpy as np
from ultralytics import YOLO
import os

model = YOLO('yolov8m.pt')

def apply_creepy_effect(frame, boxes):
    # 1. Background desaturation or fog effects
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.4  # Desaturate
    hsv[:, :, 2] = hsv[:, :, 2] * 0.7  # Darken
    bg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Fog tint
    tint = np.zeros_like(bg)
    tint[:, :, 0] = 50  # Blue
    tint[:, :, 1] = 30  # Green
    bg = cv2.addWeighted(bg, 1, tint, 0.4, 0)
    
    # Glitch effect (Chromatic Aberration) - Keep this as it looks cinematic
    if np.random.rand() > 0.85:
        b, g, r = cv2.split(bg)
        shift = np.random.randint(5, 10)
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)
        bg = cv2.merge((b, g, r))

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        
        # Only apply extreme creepy effect to persons
        if cls_id == 0: 
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
                
            # Darken the person significantly (shadow effect)
            dark_roi = cv2.convertScaleAbs(roi, alpha=0.5, beta=0)
            
            # Subtle red tint aura (blood/glowing effect approximation)
            red_tint = np.zeros_like(dark_roi)
            red_tint[:, :, 2] = 120 # Red channel
            
            creepy_person = cv2.addWeighted(dark_roi, 1, red_tint, 0.3, 0)
            
            # Insert person back into the desaturated background
            bg[y1:y2, x1:x2] = creepy_person
            
            # Add spatial glitch/flicker to the person randomly
            if np.random.rand() > 0.8:
                shift = np.random.randint(-10, 10)
                if x1+shift > 0 and x2+shift < bg.shape[1]:
                    bg[y1:y2, x1+shift:x2+shift] = creepy_person
            
    return bg

def main():
    input_video = r"d:\Semester 4\AI LAB\OEL\trailer_raw.mp4"
    output_video = r"d:\Semester 4\AI LAB\OEL\trailer_creepy.mp4"
    
    if not os.path.exists(input_video):
        print(f"Error: {input_video} not found. Run task2_3 first.")
        return
        
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print("Applying creepy transformations...")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}...")
            
        # Detect objects
        results = model(frame, conf=0.15, verbose=False)
        
        boxes = []
        for r in results:
            boxes.extend(r.boxes)
            
        creepy_frame = apply_creepy_effect(frame, boxes)
        out.write(creepy_frame)
        
    cap.release()
    out.release()
    print(f"Creepy trailer generated successfully: {output_video}")

if __name__ == '__main__':
    main()
