import os
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
db = SQLAlchemy()
bcrypt = Bcrypt()

def create_app():
    app = Flask(__name__)
    load_dotenv()
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    logger.debug(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
    
    try:
        db.init_app(app)
        bcrypt.init_app(app)
        # Configure CORS globally for all routes
        CORS(app, resources={
            r"/api/*": {
                "origins": ["https://healia.netlify.app", "http://localhost:5173"],
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
                "supports_credentials": True
            }
        })
        
        # Register blueprints
        from .routes import auth_bp
        from .admin import admin_bp  # Import admin blueprint
        app.register_blueprint(auth_bp, url_prefix='/api/auth')
        app.register_blueprint(admin_bp, url_prefix='/api/admin')
        
        with app.app_context():
            db.create_all()
            logger.info("Database tables created successfully")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise
    
    # Root endpoint for health checks (for UptimeRobot)
    @app.route('/', methods=['GET', 'HEAD'])
    def root_health_check():
        logger.info("Received root health check request")
        return jsonify({'status': 'alive'}), 200
    
    # Fallback for unhandled OPTIONS requests on /api/admin/*
    @app.route('/api/admin/<path:path>', methods=['OPTIONS'])
    def admin_options(path):
        logger.debug(f"Handling OPTIONS request for /api/admin/{path}")
        return jsonify({}), 200
    
    @app.errorhandler(Exception)
    def handle_error(error):
        logger.error(f"Unhandled error: {str(error)}")
        return jsonify({'message': 'Internal server error', 'error': str(error)}), 500
    
    return app
