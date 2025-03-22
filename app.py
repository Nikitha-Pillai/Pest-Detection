import sqlite3
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
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
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24).hex())  # Secure secret key

# Database setup for pest detections and users
def init_db():
    conn = sqlite3.connect('pest_detection.db')
    c = conn.cursor()
    # Table for pest detections
    c.execute('''CREATE TABLE IF NOT EXISTS pest_detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        image BLOB NOT NULL,
        pest_id INTEGER,
        pest_name TEXT,
        detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    # Table for users
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        mobile TEXT NOT NULL,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
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
try:
    model = fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 7
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model_path = "D:/pest_detection_app/C4pest_detector_best.pth"
    print("Current working directory:", os.getcwd())
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the file exists.")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

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

# Function to get pest counts from the database for a specific user
def get_pest_counts(user_id):
    conn = sqlite3.connect('pest_detection.db')
    c = conn.cursor()
    c.execute('SELECT pest_id, COUNT(*) FROM pest_detections WHERE user_id = ? GROUP BY pest_id', (user_id,))
    counts = {}
    for pest_id, count in c.fetchall():
        if isinstance(pest_id, bytes):
            pest_id = int.from_bytes(pest_id, byteorder='little')
        counts[pest_id] = count
    conn.close()
    return counts

# Function to get recent alerts from the database for a specific user
def get_alerts(user_id):
    conn = sqlite3.connect('pest_detection.db')
    c = conn.cursor()
    c.execute('SELECT id, pest_name FROM pest_detections WHERE user_id = ? ORDER BY detection_time DESC LIMIT 5', (user_id,))
    alerts = [{'id': row[0], 'message': f"Your field is infested by {row[1]}"} for row in c.fetchall()]
    conn.close()
    return alerts

# Route for the main page (protected)
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get the user_id of the logged-in user
    username = session['username']
    conn = sqlite3.connect('pest_detection.db')
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    
    if not user:
        session.pop('username', None)
        return redirect(url_for('login'))
    
    user_id = user[0]
    pest_counts = get_pest_counts(user_id)
    alerts = get_alerts(user_id)
    return render_template('index.html', alerts=alerts, pest_counts=pest_counts)

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('pest_detection.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = c.fetchone()
        conn.close()
        
        if user:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid username or password")
    
    return render_template('login.html', error=None)

# Route for sign-up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        mobile = request.form['mobile']
        username = request.form['username']
        password = request.form['password']
        
        try:
            conn = sqlite3.connect('pest_detection.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (name, mobile, username, password) VALUES (?, ?, ?, ?)',
                      (name, mobile, username, password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('signup.html', error="Username already exists")
    
    return render_template('signup.html', error=None)

# Route for logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Route to handle image uploads and pest detection
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'username' not in session:
        return jsonify({'error': 'Please log in to upload images'}), 401
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image']
    image_pil = Image.open(image).convert("RGB")
    
    # Convert image to binary for database storage
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    image_binary = buffered.getvalue()
    
    # Get the user_id of the logged-in user
    username = session['username']
    conn = sqlite3.connect('pest_detection.db')
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    if not user:
        conn.close()
        return jsonify({'error': 'User not found'}), 400
    user_id = user[0]
    
    # Detect pests
    detected_pests = detect_pests(image_pil)
    
    # Store results in the database with user_id
    for pest_id in detected_pests:
        pest_id = int(pest_id)
        print(f"Inserting pest_id: {pest_id}, type: {type(pest_id)}")  # Debug print
        pest_name = class_names.get(pest_id, "Unknown Pest")
        c.execute('INSERT INTO pest_detections (user_id, image, pest_id, pest_name) VALUES (?, ?, ?, ?)',
                  (user_id, image_binary, pest_id, pest_name))
    conn.commit()
    conn.close()
    
    # Fetch updated data for the current user
    pest_counts = get_pest_counts(user_id)
    alerts = get_alerts(user_id)
    
    return jsonify({'alerts': alerts, 'pest_counts': pest_counts})

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)