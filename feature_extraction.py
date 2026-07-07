import os
import cv2
import numpy as np
import torch
import pandas as pd
from ultralytics import YOLO
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# 1. Initialize Models
print("Loading Models...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# YOLO for object detection
yolo_model = YOLO('yolov8n.pt')

# ResNet18 for visual embeddings
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Remove the final classification layer to get embeddings
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
resnet.to(device)
resnet.eval()

# Transform for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Audio extraction removed since videos have no audio

def process_clip(video_path):
    """Processes a single video clip to extract visual and object features."""
    cap = cv2.VideoCapture(video_path)
    
    total_motion = 0.0
    total_objects = 0
    frame_count = 0
    
    prev_frame_gray = None
    embeddings = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Sample every 10th frame to speed up processing
        if frame_count % 10 != 0:
            continue
            
        # 1. Visual Intensity (Motion / Frame Diff)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame_gray is not None:
            diff = cv2.absdiff(gray, prev_frame_gray)
            motion = np.mean(diff)
            total_motion += motion
        prev_frame_gray = gray
        
        # 2. Object Presence (YOLO)
        results = yolo_model(frame, verbose=False)
        # Count number of objects detected (persons, cars, weapons etc.)
        for r in results:
            total_objects += len(r.boxes)
            
        # 3. Visual Embeddings (ResNet)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = resnet(img_tensor).cpu().numpy().flatten()
            embeddings.append(emb)
            
    cap.release()
    
    sampled_frames = max(1, len(embeddings))
    avg_motion = total_motion / sampled_frames
    avg_objects = total_objects / sampled_frames
    
    if len(embeddings) > 0:
        avg_embedding = np.mean(embeddings, axis=0)
    else:
        avg_embedding = np.zeros(512) # ResNet18 output size
        
    return avg_motion, avg_objects, avg_embedding

def main():
    data_dir = r"d:\Semester 4\AI LAB\OEL\data"
    output_csv = r"d:\Semester 4\AI LAB\OEL\features.csv"
    
    video_files = [f for f in os.listdir(data_dir) if f.endswith('.mp4')]
    
    print(f"Found {len(video_files)} clips. Starting feature extraction...")
    
    features_list = []
    
    for idx, video_name in enumerate(video_files):
        print(f"[{idx+1}/{len(video_files)}] Processing {video_name}...")
        video_path = os.path.join(data_dir, video_name)
        
        # Extract visual and object features
        avg_motion, avg_objects, avg_emb = process_clip(video_path)
        
        # Compile features
        feature_dict = {
            'clip_name': video_name,
            'motion_intensity': avg_motion,
            'object_density': avg_objects
        }
        
        # Add embedding dimensions as separate columns
        for i, val in enumerate(avg_emb[:10]): # Storing top 10 embedding dims to save space, normally you'd save all or use PCA
            feature_dict[f'emb_{i}'] = val
            
        features_list.append(feature_dict)
        
    # Save to CSV
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv, index=False)
    print(f"Feature extraction complete! Saved to {output_csv}")

if __name__ == '__main__':
    main()
