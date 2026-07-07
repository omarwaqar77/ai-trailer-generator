import cv2
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

def extract_frame_features(frame, yolo_model, resnet, transform, device, prev_gray):
    # 1. Visual Intensity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion = 0.0
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        motion = np.mean(diff)
        
    # 2. Object Presence
    results = yolo_model(frame, verbose=False)
    objects = sum([len(r.boxes) for r in results])
    
    # 3. ResNet Embedding
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb = resnet(img_tensor).cpu().numpy().flatten()
        
    return motion, objects, emb, gray

def evaluate_trailer(video_path, model_path, scaler_path, output_graph):
    print("Loading ML models for evaluation...")
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model = YOLO('yolov8n.pt')
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    scores = []
    times = []
    
    frame_count = 0
    prev_gray = None
    
    print("Evaluating trailer frame-by-frame...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Sample every 15 frames (0.5 seconds)
        if frame_count % 15 != 0:
            continue
            
        motion, objects, emb, prev_gray = extract_frame_features(frame, yolo_model, resnet, transform, device, prev_gray)
        
        # Compile feature vector. No audio features anymore.
        feature_vector = [motion, objects] + list(emb[:10])
        
        # Scale and Predict
        X = scaler.transform([feature_vector])
        
        # Get probability of class 1 (High Impact)
        prob = classifier.predict_proba(X)[0][1] 
        score = prob * 2 - 1 # Map [0, 1] -> [-1, 1]
        
        scores.append(score)
        times.append(frame_count / fps)
        
    cap.release()
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(times, scores, label="Impact Score (-1 to 1)", color='red', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--')
    plt.fill_between(times, 0, scores, where=(np.array(scores) >= 0), color='red', alpha=0.3)
    plt.fill_between(times, 0, scores, where=(np.array(scores) < 0), color='blue', alpha=0.3)
    
    plt.title("Trailer Evaluation: Impact Score Timeline")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Impact Score")
    plt.ylim(-1.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(output_graph, dpi=300)
    print(f"Evaluation complete! Graph saved to {output_graph}")
    
    final_score = np.mean(scores)
    print(f"Overall Trailer Impact Score: {final_score:.2f} / 1.00")

if __name__ == '__main__':
    video_path = r"d:\Semester 4\AI LAB\OEL\trailer_final.mp4"
    model_path = r"d:\Semester 4\AI LAB\OEL\impact_model.pkl"
    scaler_path = r"d:\Semester 4\AI LAB\OEL\scaler.pkl"
    output_graph = r"d:\Semester 4\AI LAB\OEL\impact_timeline.png"
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
    elif not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train model first.")
    else:
        evaluate_trailer(video_path, model_path, scaler_path, output_graph)
