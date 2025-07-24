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
    # Use absolute path for site.db
    db_path = os.path.join(os.path.dirname(__file__), 'site.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    logging.debug(f"Database path: {db_path}")
    db.init_app(app)
    bcrypt.init_app(app)
    CORS(app, resources={r"/*": {"origins": "https://healia.netlify.app/"}}, supports_credentials=True)
    from .routes import auth_bp
    app.register_blueprint(auth_bp)
    @app.errorhandler(Exception)
    def handle_error(error):
        logging.error(f"Unhandled error: {str(error)}")
        return jsonify({'message': 'Internal server error', 'error': str(error)}), 500
    return app