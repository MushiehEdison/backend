from . import db
from sqlalchemy.types import JSON
from datetime import datetime, date

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    phone = db.Column(db.String(15), nullable=False, unique=True)
    language = db.Column(db.String(32), nullable=False)
    gender = db.Column(db.String(20), nullable=False)

    # Relationships
    conversations = db.relationship('Conversation', back_populates='user', lazy=True)
    medical_profile = db.relationship('MedicalProfile', back_populates='user', uselist=False, lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'

class Conversation(db.Model):
    __tablename__ = 'conversation'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    messages = db.Column(JSON, nullable=False, default=[])
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = db.relationship('User', back_populates='conversations')

    def __repr__(self):
        return f'<Conversation {self.id} for user {self.user_id}>'

class MedicalProfile(db.Model):
    __tablename__ = 'medical_profiles'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Personal Info
    first_name = db.Column(db.String(100), nullable=True)
    last_name = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    date_of_birth = db.Column(db.Date, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    marital_status = db.Column(db.String(20), nullable=True)
    nationality = db.Column(db.String(50), nullable=True)
    region = db.Column(db.String(50), nullable=True)
    city = db.Column(db.String(50), nullable=True)
    quarter = db.Column(db.String(50), nullable=True)
    address = db.Column(db.String(200), nullable=True)
    profession = db.Column(db.String(100), nullable=True)

    # Emergency Contact
    emergency_contact = db.Column(db.String(100), nullable=True)
    emergency_relation = db.Column(db.String(50), nullable=True)
    emergency_phone = db.Column(db.String(20), nullable=True)

    # Health Info
    blood_type = db.Column(db.String(5), nullable=True)
    genotype = db.Column(db.String(5), nullable=True)
    allergies = db.Column(db.Text, nullable=True)
    chronic_conditions = db.Column(db.Text, nullable=True)
    medications = db.Column(db.Text, nullable=True)

    # Medical Providers
    primary_hospital = db.Column(db.String(100), nullable=True)
    primary_physician = db.Column(db.String(100), nullable=True)

    # Medical History
    medical_history = db.Column(db.Text, nullable=True)
    vaccination_history = db.Column(db.Text, nullable=True)
    last_dental_visit = db.Column(db.Date, nullable=True)
    last_eye_exam = db.Column(db.Date, nullable=True)

    # Lifestyle
    lifestyle = db.Column(
        db.JSON,
        nullable=True,
        default=lambda: {
            'smokes': False,
            'alcohol': 'Never',
            'exercise': 'Never',
            'diet': 'Balanced'
        }
    )

    # Family Medical History
    family_history = db.Column(db.Text, nullable=True)

    user = db.relationship('User', back_populates='medical_profile')

    @property
    def age(self):
        if self.date_of_birth:
            today = date.today()
            return today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
            )
        return None

    def to_dict(self):
            return {
                # ... all existing fields ...
                'dateOfBirth': self.date_of_birth.isoformat() if self.date_of_birth else None,
                'lastDentalVisit': self.last_dental_visit.isoformat() if self.last_dental_visit else None,
                'lastEyeExam': self.last_eye_exam.isoformat() if self.last_eye_exam else None,
                'lifestyle': self.lifestyle or {}
            }
class MessageIndex(db.Model):
    __tablename__ = 'message_index'
    id = db.Column(db.Integer, primary_key=True)
    convo_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    keyword = db.Column(db.String(100), nullable=False)
    summary = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)