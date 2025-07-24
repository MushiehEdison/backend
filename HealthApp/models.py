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
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    date_of_birth = db.Column(db.Date)
    gender = db.Column(db.String(10))
    marital_status = db.Column(db.String(20))
    nationality = db.Column(db.String(50))
    region = db.Column(db.String(50))
    city = db.Column(db.String(50))
    quarter = db.Column(db.String(50))
    address = db.Column(db.String(200))
    profession = db.Column(db.String(100))

    # Emergency Contact
    emergency_contact = db.Column(db.String(100))
    emergency_relation = db.Column(db.String(50))
    emergency_phone = db.Column(db.String(20))

    # Health Info
    blood_type = db.Column(db.String(5))
    genotype = db.Column(db.String(5))
    allergies = db.Column(db.Text)
    chronic_conditions = db.Column(db.Text)
    medications = db.Column(db.Text)

    # Medical Providers
    primary_hospital = db.Column(db.String(100))
    primary_physician = db.Column(db.String(100))

    # Medical History
    medical_history = db.Column(db.Text)
    vaccination_history = db.Column(db.Text)
    last_dental_visit = db.Column(db.Date)
    last_eye_exam = db.Column(db.Date)

    # Lifestyle
    lifestyle = db.Column(JSON)

    # Family Medical History
    family_history = db.Column(db.Text)

    user = db.relationship('User', back_populates='medical_profile')

    @property
    def age(self):
        if self.date_of_birth:
            today = date.today()
            return today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
            )
        return None

class MessageIndex(db.Model):
    __tablename__ = 'message_index'
    id = db.Column(db.Integer, primary_key=True)
    convo_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    keyword = db.Column(db.String(100), nullable=False)
    summary = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)