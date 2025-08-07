from HealthApp import create_app, db
from flask_migrate import Migrate, upgrade  # Import upgrade for migration
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)  # Use INFO for production to reduce verbosity
logger = logging.getLogger(__name__)

app = create_app()
migrate = Migrate(app, db)  # Initialize Migrate (ensure this is also in create_app)

with app.app_context():
    try:
        # Create tables for new models (if they don't exist)
        db.create_all()
        logger.info("Database tables created successfully")
        
        # Apply migrations to update existing tables (e.g., add created_at column)
        logger.info("Applying database migrations...")
        upgrade()  # Runs flask db upgrade to apply pending migrations
        logger.info("Database migrations applied successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

if __name__ == '__main__':
    # Use debug mode only in development
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
