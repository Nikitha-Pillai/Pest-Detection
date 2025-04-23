import firebase_admin
from firebase_admin import credentials, firestore
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
from dotenv import load_dotenv
import hashlib

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24).hex())  # Secure secret key from .env

# Initialize Firebase
cred = credentials.Certificate(os.getenv('FIREBASE_CREDENTIALS', 'D:/pest_detection_app/serviceAccountKey.json'))
firebase_admin.initialize_app(cred)
db = firestore.client()

# Class names for pests
class_names = {
    0: "corn pests (superclass)",
    1: "army worm",
    2: "black cut worm",
    3: "grub",
    4: "mole cricket",
    5: "yellow peach moth",
    6: "red spider mite"
}

# Load the trained model
try:
    model = fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 7
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model_path = "D:/pest_detection_app/C4pest_detector_best.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
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
    
    mask = pred_scores > 0.5
    detected_pests = pred_labels[mask]
    
    return detected_pests

# Function to get pest counts from Firestore for a specific user
def get_pest_counts(user_id):
    try:
        pest_ref = db.collection('pest_detections').where('user_id', '==', user_id).stream()
        counts = {}
        for doc in pest_ref:
            pest_id = doc.to_dict().get('pest_id')
            if pest_id in counts:
                counts[pest_id] += 1
            else:
                counts[pest_id] = 1
        print(f"Pest counts for user_id {user_id}: {counts}")
        return counts
    except Exception as e:
        print(f"Error fetching pest counts: {e}")
        return {}

# Function to get recent alerts from Firestore for a specific user
def get_alerts(user_id):
    try:
        pest_ref = db.collection('pest_detections').where('user_id', '==', user_id).order_by('detection_time', direction=firestore.Query.DESCENDING).limit(5).stream()
        alerts = [{'id': doc.id, 'message': f"Your field is infested by {doc.to_dict().get('pest_name')}"} for doc in pest_ref]
        print(f"Alerts for user_id {user_id}: {alerts}")
        return alerts
    except Exception as e:
        print(f"Error fetching alerts: {e}")
        return []

# Route for the main page (protected)
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    try:
        user_ref = db.collection('users').where('username', '==', username).limit(1).stream()
        user = next(user_ref, None)
        if not user:
            session.pop('username', None)
            return redirect(url_for('login'))
        user_id = user.id
        pest_counts = get_pest_counts(user_id)
        alerts = get_alerts(user_id)
        return render_template('index.html', alerts=alerts, pest_counts=pest_counts, class_names=class_names)
    except Exception as e:
        print(f"Error in index route: {e}")
        return render_template('index.html', alerts=[], pest_counts={}, error="Database error")

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        try:
            user_ref = db.collection('users').where('username', '==', username).where('password', '==', password_hash).limit(1).stream()
            user = next(user_ref, None)
            if user:
                session['username'] = username
                return redirect(url_for('index'))
            else:
                return render_template('login.html', error="Invalid username or password")
        except Exception as e:
            print(f"Error in login route: {e}")
            return render_template('login.html', error="Database error")
    
    return render_template('login.html', error=None)

# Route for sign-up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        mobile = request.form['mobile']
        username = request.form['username']
        password = request.form['password']
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        try:
            user_ref = db.collection('users').where('username', '==', username).limit(1).stream()
            if next(user_ref, None):
                return render_template('signup.html', error="Username already exists")
            
            db.collection('users').add({
                'name': name,
                'mobile': mobile,
                'username': username,
                'password': password_hash
            })
            return redirect(url_for('login'))
        except Exception as e:
            print(f"Error in signup route: {e}")
            return render_template('signup.html', error="Database error")
    
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
    
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    image_binary = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    username = session['username']
    try:
        user_ref = db.collection('users').where('username', '==', username).limit(1).stream()
        user = next(user_ref, None)
        if not user:
            return jsonify({'error': 'User not found'}), 400
        user_id = user.id
        
        detected_pests = detect_pests(image_pil)
        
        for pest_id in detected_pests:
            pest_id = int(pest_id)
            pest_name = class_names.get(pest_id, "Unknown Pest")
            db.collection('pest_detections').add({
                'user_id': user_id,
                'image': image_binary,
                'pest_id': pest_id,
                'pest_name': pest_name,
                'detection_time': firestore.SERVER_TIMESTAMP
            })
        
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Error in upload_image route: {e}")
        return jsonify({'error': 'Database error'}), 500

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
