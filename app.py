# app.py
import sqlite3
from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Database setup
def init_db():
    conn = sqlite3.connect('pest_detection.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pest_detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image BLOB NOT NULL,
        pest_id INTEGER,
        pest_name TEXT,
        detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

# Class names for pests
class_names = {
    0: "corn pests (superclass)",
    1: "army worm",
    2: "black cut worm",
    3: "grub",
    4: "mole cricket",
    5: "peach borer",
    6: "red spider mite"
}

# Load the trained model
model = fasterrcnn_resnet50_fpn(weights=None)
num_classes = 7
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model_path = "D:/pest_detection_app/C4pest_detector_best.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to detect pests in an image
def detect_pests(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image_tensor)
    
    pred_labels = predictions[0]['labels'].numpy()
    pred_scores = predictions[0]['scores'].numpy()
    
    # Filter predictions with a confidence threshold
    mask = pred_scores > 0.5
    detected_pests = pred_labels[mask]
    
    return detected_pests

# Function to get pest counts from the database
def get_pest_counts():
    conn = sqlite3.connect('pest_detection.db')
    c = conn.cursor()
    c.execute('SELECT pest_id, COUNT(*) FROM pest_detections GROUP BY pest_id')
    counts = {}
    for pest_id, count in c.fetchall():
        if isinstance(pest_id, bytes):
            # Convert bytes to integer (little-endian)
            pest_id = int.from_bytes(pest_id, byteorder='little')
        counts[pest_id] = count
    conn.close()
    return counts

# Function to get recent alerts from the database
def get_alerts():
    conn = sqlite3.connect('pest_detection.db')
    c = conn.cursor()
    c.execute('SELECT id, pest_name FROM pest_detections ORDER BY detection_time DESC LIMIT 5')
    alerts = [{'id': row[0], 'message': f"Your field is infested by {row[1]}"} for row in c.fetchall()]
    conn.close()
    return alerts

# Route for the main page
@app.route('/')
def index():
    pest_counts = get_pest_counts()
    alerts = get_alerts()
    return render_template('index.html', alerts=alerts, pest_counts=pest_counts)

# Route to handle image uploads and pest detection
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image']
    image_pil = Image.open(image).convert("RGB")
    
    # Convert image to binary for database storage
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    image_binary = buffered.getvalue()
    
    # Detect pests
    detected_pests = detect_pests(image_pil)
    
    # Store results in the database
    conn = sqlite3.connect('pest_detection.db')
    c = conn.cursor()
    for pest_id in detected_pests:
        # Ensure pest_id is a Python integer
        pest_id = int(pest_id)
        print(f"Inserting pest_id: {pest_id}, type: {type(pest_id)}")  # Debug print
        pest_name = class_names.get(pest_id, "Unknown Pest")
        c.execute('INSERT INTO pest_detections (image, pest_id, pest_name) VALUES (?, ?, ?)',
                  (image_binary, pest_id, pest_name))
    conn.commit()
    conn.close()
    
    # Fetch updated data
    pest_counts = get_pest_counts()
    alerts = get_alerts()
    
    return jsonify({'alerts': alerts, 'pest_counts': pest_counts})

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)