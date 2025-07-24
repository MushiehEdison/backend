from HealthApp import create_app, db
import logging

logging.basicConfig(level=logging.DEBUG)

app = create_app()

with app.app_context():
    try:
        db.create_all()
        logging.debug("Database tables created successfully")
    except Exception as e:
        logging.error(f"Error creating database tables: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)