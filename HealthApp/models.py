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
    sessions = db.relationship('UserSession', back_populates='user', lazy=True)
    treatment_preferences = db.relationship('TreatmentPreference', back_populates='user', lazy=True)
    health_literacy = db.relationship('HealthLiteracy', back_populates='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'

class Conversation(db.Model):
    __tablename__ = 'conversation'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    messages = db.Column(JSON, nullable=False, default=[])
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    sentiment_score = db.Column(db.Float, nullable=True)
    completion_status = db.Column(db.String(20), default='in_progress', nullable=False)

    user = db.relationship('User', back_populates='conversations')
    message_indices = db.relationship('MessageIndex', back_populates='conversation', lazy=True)
    symptom_entries = db.relationship('SymptomEntry', back_populates='conversation', lazy=True)
    diagnoses = db.relationship('Diagnosis', back_populates='conversation', lazy=True)
    sentiment_records = db.relationship('SentimentRecord', back_populates='conversation', lazy=True)

    def __repr__(self):
        return f'<Conversation {self.id} for user {self.user_id}>'

class MedicalProfile(db.Model):
    __tablename__ = 'medical_profiles'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
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
    emergency_contact = db.Column(db.String(100), nullable=True)
    emergency_relation = db.Column(db.String(50), nullable=True)
    emergency_phone = db.Column(db.String(20), nullable=True)
    blood_type = db.Column(db.String(5), nullable=True)
    genotype = db.Column(db.String(5), nullable=True)
    allergies = db.Column(db.Text, nullable=True)
    chronic_conditions = db.Column(db.Text, nullable=True)
    medications = db.Column(db.Text, nullable=True)
    primary_hospital = db.Column(db.String(100), nullable=True)
    primary_physician = db.Column(db.String(100), nullable=True)
    medical_history = db.Column(db.Text, nullable=True)
    vaccination_history = db.Column(db.Text, nullable=True)
    last_dental_visit = db.Column(db.Date, nullable=True)
    last_eye_exam = db.Column(db.Date, nullable=True)
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
            'id': self.id,
            'user_id': self.user_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'phone': self.phone,
            'dateOfBirth': self.date_of_birth.isoformat() if self.date_of_birth else None,
            'gender': self.gender,
            'marital_status': self.marital_status,
            'nationality': self.nationality,
            'region': self.region,
            'city': self.city,
            'quarter': self.quarter,
            'address': self.address,
            'profession': self.profession,
            'emergency_contact': self.emergency_contact,
            'emergency_relation': self.emergency_relation,
            'emergency_phone': self.emergency_phone,
            'blood_type': self.blood_type,
            'genotype': self.genotype,
            'allergies': self.allergies,
            'chronic_conditions': self.chronic_conditions,
            'medications': self.medications,
            'primary_hospital': self.primary_hospital,
            'primary_physician': self.primary_physician,
            'medical_history': self.medical_history,
            'vaccination_history': self.vaccination_history,
            'lastDentalVisit': self.last_dental_visit.isoformat() if self.last_dental_visit else None,
            'lastEyeExam': self.last_eye_exam.isoformat() if self.last_eye_exam else None,
            'lifestyle': self.lifestyle or {},
            'family_history': self.family_history
        }

class MessageIndex(db.Model):
    __tablename__ = 'message_index'
    id = db.Column(db.Integer, primary_key=True)
    convo_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    keyword = db.Column(db.String(100), nullable=False)
    summary = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    conversation = db.relationship('Conversation', back_populates='message_indices')

    def __repr__(self):
        return f'<MessageIndex {self.id} for conversation {self.convo_id}>'

class UserSession(db.Model):
    __tablename__ = 'user_sessions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    end_time = db.Column(db.DateTime, nullable=True)
    duration_seconds = db.Column(db.Integer, nullable=True)
    last_active = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    user = db.relationship('User', back_populates='sessions')

    def __repr__(self):
        return f'<UserSession {self.id} for user {self.user_id}>'

class SymptomEntry(db.Model):
    __tablename__ = 'symptom_entries'
    id = db.Column(db.Integer, primary_key=True)
    convo_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    symptom_name = db.Column(db.String(100), nullable=False)
    severity = db.Column(db.String(20), nullable=True)
    reported_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    location = db.Column(db.String(100), nullable=True)

    conversation = db.relationship('Conversation', back_populates='symptom_entries')

    def __repr__(self):
        return f'<SymptomEntry {self.id} for conversation {self.convo_id}>'

class Diagnosis(db.Model):
    __tablename__ = 'diagnoses'
    id = db.Column(db.Integer, primary_key=True)
    convo_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    condition_name = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    requires_attention = db.Column(db.Boolean, default=False, nullable=False)

    conversation = db.relationship('Conversation', back_populates='diagnoses')

    def __repr__(self):
        return f'<Diagnosis {self.id} for conversation {self.convo_id}>'

class HealthAlert(db.Model):
    __tablename__ = 'health_alerts'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    region = db.Column(db.String(50), nullable=True)

    def __repr__(self):
        return f'<HealthAlert {self.id} - {self.title}>'

class SentimentRecord(db.Model):
    __tablename__ = 'sentiment_records'
    id = db.Column(db.Integer, primary_key=True)
    convo_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    sentiment_category = db.Column(db.String(20), nullable=False)
    percentage = db.Column(db.Float, nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    conversation = db.relationship('Conversation', back_populates='sentiment_records')

    def __repr__(self):
        return f'<SentimentRecord {self.id} for conversation {self.convo_id}>'

class CommunicationMetric(db.Model):
    __tablename__ = 'communication_metrics'
    id = db.Column(db.Integer, primary_key=True)
    metric_name = db.Column(db.String(100), nullable=False)
    current_value = db.Column(db.Float, nullable=False)
    previous_value = db.Column(db.Float, nullable=True)
    trend = db.Column(db.String(20), nullable=True)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    time_range = db.Column(db.String(20), nullable=False)

    def __repr__(self):
        return f'<CommunicationMetric {self.id} - {self.metric_name}>'

class TreatmentPreference(db.Model):
    __tablename__ = 'treatment_preferences'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    treatment_type = db.Column(db.String(100), nullable=False)
    preference_score = db.Column(db.Float, nullable=False)
    trend = db.Column(db.String(20), nullable=True)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    user = db.relationship('User', back_populates='treatment_preferences')

    def __repr__(self):
        return f'<TreatmentPreference {self.id} for user {self.user_id}>'

class HealthLiteracy(db.Model):
    __tablename__ = 'health_literacy'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    age_group = db.Column(db.String(20), nullable=False)
    understanding_rate = db.Column(db.Float, nullable=False)
    engagement_rate = db.Column(db.Float, nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    user = db.relationship('User', back_populates='health_literacy')

    def __repr__(self):
        return f'<HealthLiteracy {self.id} for user {self.user_id}>'

class WorkflowMetric(db.Model):
    __tablename__ = 'workflow_metrics'
    id = db.Column(db.Integer, primary_key=True)
    metric_name = db.Column(db.String(100), nullable=False)
    value = db.Column(db.Float, nullable=False)
    change_percentage = db.Column(db.Float, nullable=True)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f'<WorkflowMetric {self.id} - {metric_name}>'

class AIPerformance(db.Model):
    __tablename__ = 'ai_performance'
    id = db.Column(db.Integer, primary_key=True)
    metric_name = db.Column(db.String(100), nullable=False)
    value = db.Column(db.Float, nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f'<AIPerformance {self.id} - {self.metric_name}>'