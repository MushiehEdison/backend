from HealthApp import create_app, db
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)  # Use INFO for production to reduce verbosity

app = create_app()

with app.app_context():
    try:
        db.create_all()
        logging.info("Database tables created successfully")
    except Exception as e:
        logging.error(f"Error creating database tables: {str(e)}")

if __name__ == '__main__':
    # Use debug mode only in development
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
