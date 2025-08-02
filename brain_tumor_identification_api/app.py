from flask_bcrypt import Bcrypt
from flask import Flask
from flask_cors import CORS
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')

import os
import logging

from src.auth.auth import auth_bp, init_auth
from src.routes.routes import main_bp

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()  # Generate a random secret key
app.permanent_session_lifetime = timedelta(minutes=30)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

logging.basicConfig(level=logging.DEBUG)
# Initialize CORS with support for credentials
CORS(app, resources={
     r"/*": {"origins": "http://localhost:8100", "supports_credentials": True}})

bcrypt = Bcrypt(app)
# Initialize authentication
init_auth(app, bcrypt)

# Application Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['VISUALIZATION_FOLDER'] = 'static/visualizations'
app.config['PATIENT_DATA_PATH'] = 'patient data json.json'

# Register blueprints
app.register_blueprint(main_bp)
app.register_blueprint(auth_bp)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VISUALIZATION_FOLDER'], exist_ok=True)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
