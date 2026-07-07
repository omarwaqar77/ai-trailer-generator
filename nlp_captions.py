import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
import torch
import os
import textwrap

def get_creepy_synonyms(text):
    """Simple rule-based NLP to make text creepy."""
    creepy_dict = {
        "man": "shadow",
        "woman": "figure",
        "person": "entity",
        "walking": "creeping",
        "running": "fleeing",
        "looking": "staring blankly",
        "standing": "lurking",
        "sitting": "waiting in the dark",
        "room": "abyss",
        "car": "metal coffin",
        "street": "abandoned path",
        "day": "eternal night",
        "sun": "blood moon",
        "smiling": "grinning maliciously",
        "talking": "whispering secrets",
        "holding": "grasping desperately",
        "boy": "lost child",
        "girl": "lost child",
        "dog": "hellhound",
        "cat": "watcher"
    }
    
    words = text.lower().split()
    creepy_words = [creepy_dict.get(w, w) for w in words]
    creepy_text = " ".join(creepy_words)
    
    # Add creepy suffix
    suffixes = [
        "... and they are never coming back.",
        "... it's right behind you.",
        "... don't look away.",
        "... the shadows are moving.",
        "... no one can hear you."
    ]
    import random
    creepy_text += " " + random.choice(suffixes)
    
    return creepy_text.capitalize()

def generate_captions(video_path, output_path):
    print("Loading NLP Model (BLIP)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # We will generate one caption per scene (every 60 frames to save time)
    current_caption = "..."
    
    frame_count = 0
    print("Applying NLP captions...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Generate new caption every 2 seconds (assuming 30fps)
        if frame_count % 60 == 1 or frame_count == 1:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            inputs = processor(pil_img, return_tensors="pt").to(device)
            out_ids = model.generate(**inputs, max_new_tokens=20)
            normal_caption = processor.decode(out_ids[0], skip_special_tokens=True)
            
            current_caption = get_creepy_synonyms(normal_caption)
            print(f"Frame {frame_count}: Normal: '{normal_caption}' -> Creepy: '{current_caption}'")
            
        # Draw text on frame
        # Wrap text to fit screen
        wrapped_text = textwrap.wrap(current_caption, width=50)
        
        y_offset = height - 100
        for i, line in enumerate(wrapped_text):
            # Draw black background for text
            (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
            cv2.rectangle(frame, (50, y_offset - h - 10), (50 + w + 10, y_offset + 10), (0,0,0), -1)
            # Draw red text
            cv2.putText(frame, line, (55, y_offset), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
            y_offset += h + 15
            
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"Captioned trailer saved to {output_path}")

if __name__ == '__main__':
    input_video = r"d:\Semester 4\AI LAB\OEL\trailer_creepy.mp4"
    output_video = r"d:\Semester 4\AI LAB\OEL\trailer_final.mp4"
    
    if not os.path.exists(input_video):
        print(f"Error: {input_video} not found.")
    else:
        generate_captions(input_video, output_video)
