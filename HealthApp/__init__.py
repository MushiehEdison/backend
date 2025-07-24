import os
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.DEBUG)
db = SQLAlchemy()
bcrypt = Bcrypt()

def create_app():
    app = Flask(__name__)
    load_dotenv()
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    # Use PostgreSQL DATABASE_URL from environment variables
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    logging.debug(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
    
    try:
        db.init_app(app)
        bcrypt.init_app(app)
        CORS(auth_bp, origins=["http://localhost:5173", "https://healia.netlify.app"], supports_credentials=True)        
        from .routes import auth_bp
        app.register_blueprint(auth_bp)
        
        # Create database tables if they don't exist
        with app.app_context():
            db.create_all()
            logging.info("Database tables created successfully")
            
    except Exception as e:
        logging.error(f"Database initialization failed: {str(e)}")
        raise
    
    @app.errorhandler(Exception)
    def handle_error(error):
        logging.error(f"Unhandled error: {str(error)}")
        return jsonify({'message': 'Internal server error', 'error': str(error)}), 500
    
    return app