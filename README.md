# Computer Vision and Machine Learning-Based Movie Trailer Generation Pipeline

An intelligent movie trailer generation pipeline that automatically analyzes video clips, identifies the most impactful scenes using computer vision and machine learning techniques, and generates a cinematic trailer with AI-generated captions and visual enhancements.

The project demonstrates how artificial intelligence can automate the traditionally manual process of trailer creation by combining feature extraction, scene ranking, trailer composition, visual transformations, and natural language processing into a single end-to-end workflow.

---

## Features

- Automatic analysis of movie clips
- Computer vision-based feature extraction
- Object detection using YOLOv8
- Deep feature extraction using ResNet18
- Machine learning-based scene ranking
- Automatic trailer generation
- AI-generated captions using BLIP
- Cinematic visual transformations
- Trailer quality evaluation and visualization

---

## Pipeline Overview

```
Input Video Clips
        │
        ▼
Feature Extraction
(Motion + Objects + Deep Features)
        │
        ▼
Machine Learning Model
(Scene Ranking)
        │
        ▼
Top Scene Selection
        │
        ▼
Trailer Generation
        │
        ▼
Visual Enhancements
        │
        ▼
AI Caption Generation
        │
        ▼
Final Movie Trailer
        │
        ▼
Evaluation
```

---

## Technologies Used

- Python
- OpenCV
- PyTorch
- YOLOv8
- ResNet18
- BLIP Image Captioning
- Scikit-learn
- MoviePy
- NumPy
- Pandas
- Matplotlib

---

## Project Structure

```
.
├── feature_extraction.py
├── model_training.py
├── trailer_generation.py
├── creepy_transformation.py
├── nlp_captions.py
├── evaluation.py
├── features.csv
├── labeled_features.csv
├── trained_model.pkl
└── README.md
```

---

## Workflow

### 1. Feature Extraction

Each video clip is analyzed to extract meaningful visual features including:

- Motion intensity
- Object density using YOLOv8
- Deep visual embeddings using ResNet18

These features provide quantitative information about the content of every scene.

---

### 2. Scene Ranking

The extracted features are used to train a Logistic Regression model that predicts the importance of each clip.

Higher-ranked clips are selected for inclusion in the final trailer.

---

### 3. Trailer Generation

The selected clips are arranged into a coherent sequence to create a concise movie trailer while preserving visual continuity.

---

### 4. Visual Enhancement

The generated trailer is enhanced using cinematic visual effects including:

- Color grading
- Contrast enhancement
- Atmospheric effects
- Scene stylization

These enhancements improve the overall presentation of the trailer.

---

### 5. Caption Generation

Representative trailer frames are processed using the BLIP image captioning model to generate descriptive captions automatically.

The captions are then formatted as trailer text overlays.

---

### 6. Evaluation

The generated trailer is evaluated by analyzing the selected scenes and visualizing the predicted impact scores to assess trailer quality.

---

## Installation

Clone the repository

```bash
git clone https://github.com/omarwaqar77/ai-trailer-generator.git
```

Install the required packages

```bash
pip install opencv-python torch torchvision ultralytics transformers moviepy scikit-learn pandas numpy matplotlib pillow
```

---

## Running the Project

Execute the scripts in the following order:

1. Feature Extraction

```bash
python feature_extraction.py
```

2. Model Training

```bash
python model_training.py
```

3. Trailer Generation

```bash
python trailer_generation.py
```

4. Visual Enhancement

```bash
python creepy_transformation.py
```

5. Caption Generation

```bash
python nlp_captions.py
```

6. Evaluation

```bash
python evaluation.py
```

---

## Future Improvements

- Support for multiple movie genres
- Transformer-based scene ranking
- Audio sentiment analysis
- Automatic soundtrack generation
- Large Language Model (LLM) based trailer narration
- Web-based user interface
- Real-time trailer generation

---

## Authors

- Mohammad Omar Waqar
- Hassan Tariq
- Najam Ul Saqib
- Khadija Usman
- Diya Hurmat

FAST National University of Computer and Emerging Sciences

Bachelor of Science in Artificial Intelligence

---

## License

This project was developed for educational and research purposes.
