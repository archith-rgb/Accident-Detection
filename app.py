# app.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, unset_jwt_cookies
from flask_cors import CORS # Import CORS
from datetime import timedelta
import os
import datetime

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all origins (for development)
CORS(app)

# --- Configuration ---
# Database configuration: SQLite database file
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///accident_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# JWT Configuration
# !!! IMPORTANT: Change this secret key in a production environment !!!
app.config['JWT_SECRET_KEY'] = 'your_super_secret_jwt_key_please_change_this'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1) # Token expires in 1 hour

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# --- Database Models ---
# User Model for authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

# Camera Model
class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(200))
    stream_url = db.Column(db.String(500), nullable=False)
    username = db.Column(db.String(100)) # Credentials for camera stream
    password = db.Column(db.String(100)) # Credentials for camera stream
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # Link to user who added it

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'location': self.location,
            'stream_url': self.stream_url,
            'username': self.username,
            'password': self.password, # Be cautious about exposing passwords in API responses
            'createdAt': self.created_at.isoformat(),
            'userId': self.user_id
        }

# Accident Model
class Accident(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)
    camera_name = db.Column(db.String(100)) # Denormalized for easier display
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    video_url = db.Column(db.String(500))
    alert_details = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'camera_name': self.camera_name,
            'timestamp': self.timestamp.isoformat(),
            'video_url': self.video_url,
            'alert_details': self.alert_details,
            'userId': self.user_id
        }

# Alert Model
class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)
    camera_name = db.Column(db.String(100)) # Denormalized
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    alert_type = db.Column(db.String(100))
    description = db.Column(db.Text)
    status = db.Column(db.String(50), default='New') # e.g., 'New', 'Resolved', 'Acknowledged'
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'camera_name': self.camera_name,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'description': self.description,
            'status': self.status,
            'userId': self.user_id
        }

# --- Database Initialization (Run once to create tables) ---
@app.before_first_request
def create_tables():
    db.create_all()

# --- User Authentication Endpoints ---

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"msg": "Username and password are required"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"msg": "Username already exists"}), 409

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, password_hash=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"msg": "User registered successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    if not user or not bcrypt.check_password_hash(user.password_hash, password):
        return jsonify({"msg": "Bad username or password"}), 401

    access_token = create_access_token(identity=user.id)
    return jsonify(access_token=access_token, username=user.username, userId=user.id), 200

@app.route('/logout', methods=['POST'])
@jwt_required() # Requires a valid JWT token
def logout():
    # In a real app, you might blacklist tokens. For simplicity, we just unset cookies.
    response = jsonify({"msg": "Logout successful"})
    unset_jwt_cookies(response)
    return response, 200

# --- Camera Management Endpoints ---

@app.route('/cameras', methods=['POST'])
@jwt_required()
def add_camera():
    current_user_id = get_jwt_identity()
    data = request.get_json()
    name = data.get('name')
    location = data.get('location')
    stream_url = data.get('stream_url')
    username = data.get('username')
    password = data.get('password')

    if not name or not stream_url:
        return jsonify({"msg": "Camera name and stream URL are required"}), 400

    new_camera = Camera(
        name=name,
        location=location,
        stream_url=stream_url,
        username=username,
        password=password,
        user_id=current_user_id
    )
    db.session.add(new_camera)
    db.session.commit()
    return jsonify(new_camera.to_dict()), 201

@app.route('/cameras', methods=['GET'])
@jwt_required()
def get_cameras():
    current_user_id = get_jwt_identity()
    # Fetch only cameras added by the current user
    cameras = Camera.query.filter_by(user_id=current_user_id).all()
    return jsonify([camera.to_dict() for camera in cameras]), 200

@app.route('/cameras/<int:camera_id>', methods=['PUT'])
@jwt_required()
def update_camera(camera_id):
    current_user_id = get_jwt_identity()
    camera = Camera.query.filter_by(id=camera_id, user_id=current_user_id).first()

    if not camera:
        return jsonify({"msg": "Camera not found or not authorized"}), 404

    data = request.get_json()
    camera.name = data.get('name', camera.name)
    camera.location = data.get('location', camera.location)
    camera.stream_url = data.get('stream_url', camera.stream_url)
    camera.username = data.get('username', camera.username)
    camera.password = data.get('password', camera.password) # Update password if provided
    db.session.commit()
    return jsonify(camera.to_dict()), 200

@app.route('/cameras/<int:camera_id>', methods=['DELETE'])
@jwt_required()
def delete_camera(camera_id):
    current_user_id = get_jwt_identity()
    camera = Camera.query.filter_by(id=camera_id, user_id=current_user_id).first()

    if not camera:
        return jsonify({"msg": "Camera not found or not authorized"}), 404

    db.session.delete(camera)
    db.session.commit()
    return jsonify({"msg": "Camera deleted successfully"}), 200

# --- Accident History Endpoints (for your Python detection models to use) ---

@app.route('/accidents', methods=['POST'])
# No jwt_required here if your detection model is a separate service that doesn't log in
# You might use an API key or other secure method for model-to-backend communication
def add_accident():
    data = request.get_json()
    camera_id = data.get('camera_id')
    camera_name = data.get('camera_name')
    video_url = data.get('video_url')
    alert_details = data.get('alert_details')
    # Assuming a default user for detections if not linked to a specific logged-in user
    # Or you could pass a user_id from your detection system if it's user-specific
    user_id = data.get('user_id', 1) # Default to user_id 1 for simplicity if not provided

    if not camera_id or not video_url:
        return jsonify({"msg": "Camera ID and video URL are required for accident record"}), 400

    new_accident = Accident(
        camera_id=camera_id,
        camera_name=camera_name,
        video_url=video_url,
        alert_details=alert_details,
        user_id=user_id
    )
    db.session.add(new_accident)
    db.session.commit()
    return jsonify(new_accident.to_dict()), 201

@app.route('/accidents', methods=['GET'])
@jwt_required()
def get_accidents():
    current_user_id = get_jwt_identity()
    accidents = Accident.query.filter_by(user_id=current_user_id).order_by(Accident.timestamp.desc()).all()
    return jsonify([accident.to_dict() for accident in accidents]), 200

# --- Alerts Endpoints (for your Python detection models to use) ---

@app.route('/alerts', methods=['POST'])
# No jwt_required here if your detection model is a separate service
def add_alert():
    data = request.get_json()
    camera_id = data.get('camera_id')
    camera_name = data.get('camera_name')
    alert_type = data.get('alert_type')
    description = data.get('description')
    status = data.get('status', 'New')
    user_id = data.get('user_id', 1) # Default to user_id 1

    if not camera_id or not alert_type:
        return jsonify({"msg": "Camera ID and alert type are required for alert"}), 400

    new_alert = Alert(
        camera_id=camera_id,
        camera_name=camera_name,
        alert_type=alert_type,
        description=description,
        status=status,
        user_id=user_id
    )
    db.session.add(new_alert)
    db.session.commit()
    return jsonify(new_alert.to_dict()), 201

@app.route('/alerts', methods=['GET'])
@jwt_required()
def get_alerts():
    current_user_id = get_jwt_identity()
    alerts = Alert.query.filter_by(user_id=current_user_id).order_by(Alert.timestamp.desc()).all()
    return jsonify([alert.to_dict() for alert in alerts]), 200

# --- Main entry point for Flask app ---
if __name__ == '__main__':
    # Create the database tables if they don't exist
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001) # Run on port 5001 for development
