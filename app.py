import mysql.connector
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

# Database connection configuration
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST', 'localhost'),
            user=os.getenv('MYSQL_USER', 'pest_app_user'),
            password=os.getenv('MYSQL_PASSWORD', 'PestAppPass123!'),
            database=os.getenv('MYSQL_DATABASE', 'pest_detection')
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        raise

# Database setup for users and pest detections
def init_db():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        # Check if tables exist
        c.execute("SHOW TABLES LIKE 'users'")
        users_table_exists = c.fetchone()
        c.execute("SHOW TABLES LIKE 'pest_detections'")
        pest_detections_table_exists = c.fetchone()

        if not users_table_exists:
            try:
                c.execute('''CREATE TABLE users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    mobile VARCHAR(20) NOT NULL,
                    username VARCHAR(50) NOT NULL UNIQUE,
                    password VARCHAR(255) NOT NULL
                )''')
                print("Created 'users' table.")
            except mysql.connector.Error as err:
                print(f"Error creating 'users' table: {err}")
                print("Please create the 'users' table manually using a user with CREATE privileges.")
                raise

        if not pest_detections_table_exists:
            try:
                c.execute('''CREATE TABLE pest_detections (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    image MEDIUMBLOB NOT NULL,
                    pest_id INT,
                    pest_name VARCHAR(100),
                    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )''')
                print("Created 'pest_detections' table.")
            except mysql.connector.Error as err:
                print(f"Error creating 'pest_detections' table: {err}")
                print("Please create the 'pest_detections' table manually using a user with CREATE and REFERENCES privileges.")
                raise

        conn.commit()
        print("Database tables verified/created successfully!")
    except mysql.connector.Error as err:
        print(f"Error initializing database: {err}")
        raise
    finally:
        conn.close()

# Initialize the database
init_db()

# Test the MySQL connection on startup
try:
    conn = get_db_connection()
    print("Connected to MySQL successfully!")
    conn.close()
except Exception as e:
    print(f"Failed to connect to MySQL: {e}")
    exit(1)

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
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT pest_id, COUNT(*) FROM pest_detections WHERE user_id = %s GROUP BY pest_id', (user_id,))
        counts = {}
        for pest_id, count in c.fetchall():
            counts[pest_id] = count
        return counts
    except mysql.connector.Error as err:
        print(f"Error fetching pest counts: {err}")
        return {}
    finally:
        conn.close()

# Function to get recent alerts from the database for a specific user
def get_alerts(user_id):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT id, pest_name FROM pest_detections WHERE user_id = %s ORDER BY detection_time DESC LIMIT 5', (user_id,))
        alerts = [{'id': row[0], 'message': f"Your field is infested by {row[1]}"} for row in c.fetchall()]
        return alerts
    except mysql.connector.Error as err:
        print(f"Error fetching alerts: {err}")
        return []
    finally:
        conn.close()

# Route for the main page (protected)
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get the user_id of the logged-in user
    username = session['username']
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE username = %s', (username,))
        user = c.fetchone()
        if not user:
            session.pop('username', None)
            return redirect(url_for('login'))
        user_id = user[0]
        pest_counts = get_pest_counts(user_id)
        alerts = get_alerts(user_id)
        return render_template('index.html', alerts=alerts, pest_counts=pest_counts)
    except mysql.connector.Error as err:
        print(f"Error in index route: {err}")
        return render_template('index.html', alerts=[], pest_counts={}, error="Database error")
    finally:
        conn.close()

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Hash the entered password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password_hash))
            user = c.fetchone()
            if user:
                session['username'] = username
                return redirect(url_for('index'))
            else:
                return render_template('login.html', error="Invalid username or password")
        except mysql.connector.Error as err:
            print(f"Error in login route: {err}")
            return render_template('login.html', error="Database error")
        finally:
            conn.close()
    
    return render_template('login.html', error=None)

# Route for sign-up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        mobile = request.form['mobile']
        username = request.form['username']
        password = request.form['password']
        # Hash the password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('INSERT INTO users (name, mobile, username, password) VALUES (%s, %s, %s, %s)',
                      (name, mobile, username, password_hash))
            conn.commit()
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            return render_template('signup.html', error="Username already exists")
        except mysql.connector.Error as err:
            print(f"Error in signup route: {err}")
            return render_template('signup.html', error="Database error")
        finally:
            conn.close()
    
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
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE username = %s', (username,))
        user = c.fetchone()
        if not user:
            return jsonify({'error': 'User not found'}), 400
        user_id = user[0]
        
        # Detect pests
        detected_pests = detect_pests(image_pil)
        
        # Store results in the database with user_id
        for pest_id in detected_pests:
            pest_id = int(pest_id)
            print(f"Inserting pest_id: {pest_id}, type: {type(pest_id)}")  # Debug print
            pest_name = class_names.get(pest_id, "Unknown Pest")
            c.execute('INSERT INTO pest_detections (user_id, image, pest_id, pest_name) VALUES (%s, %s, %s, %s)',
                      (user_id, image_binary, pest_id, pest_name))
        conn.commit()
        
        # Fetch updated data for the current user
        pest_counts = get_pest_counts(user_id)
        alerts = get_alerts(user_id)
        
        return jsonify({'alerts': alerts, 'pest_counts': pest_counts})
    except mysql.connector.Error as err:
        print(f"Error in upload_image route: {err}")
        return jsonify({'error': 'Database error'}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)