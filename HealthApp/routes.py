from flask import Blueprint, request, jsonify, current_app
import jwt
import datetime
from datetime import timezone
import uuid
import logging
from functools import wraps
from statistics import mean
from . import db, bcrypt
from .models import User, Conversation, MedicalProfile, UserSession
from flask_jwt_extended import jwt_required
from .ai_engine import generate_personalized_response
from .analysis import HealthAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')
admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

# Root route for health checks
@auth_bp.route('/', methods=['GET', 'HEAD'])
def health_check():
    """Handle Render health checks."""
    logger.info("Received health check request")
    return jsonify({'status': 'alive'}), 200

# Catch-all OPTIONS route for all /api/auth/* endpoints
@auth_bp.route('/<path:path>', methods=['OPTIONS'])
@auth_bp.route('/', methods=['OPTIONS'])
def handle_options(path=None):
    """Handle preflight OPTIONS requests for all auth routes."""
    logger.debug(f"Handling OPTIONS request for path: {path or '/'}")
    return jsonify({}), 200

def verify_user_from_token(auth_header):
    """Verify user from JWT token in Authorization header."""
    logger.debug(f"Verifying token with auth_header: {auth_header}")
    if not auth_header:
        logger.error("Authorization header is missing")
        return None
    if not auth_header.startswith('Bearer '):
        logger.error(f"Invalid Authorization header format: {auth_header}")
        return None
    token = auth_header.split(' ')[1].strip()
    if not token:
        logger.error("Token is empty after splitting Authorization header")
        return None
    try:
        decoded = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'], options={'verify_exp': True})
        user = User.query.get(decoded['user_id'])
        if not user:
            logger.error(f"User not found for user_id: {decoded['user_id']}")
            return None
        logger.debug(f"User verified: {user.id}")
        return user
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error verifying token: {str(e)}")
        return None

def get_safe_medical_profile_attr(user, attr_name, default_value):
    """Safely get an attribute from user's medical profile."""
    if user.medical_profile is None:
        return default_value
    return getattr(user.medical_profile, attr_name, default_value)

@auth_bp.route('/signup', methods=['POST'])
def signup():
    """Handle user signup."""
    try:
        data = request.get_json()
        logger.debug(f"Signup data received: {data}")
        required_fields = ['name', 'username', 'email', 'password', 'phone', 'language', 'gender']
        if not data or not all(k in data for k in required_fields):
            logger.error("Missing required fields")
            return jsonify({'message': 'Missing required fields'}), 400

        if User.query.filter_by(email=data['email']).first():
            logger.error("Email already exists")
            return jsonify({'message': 'Email already exists'}), 400
        if User.query.filter_by(username=data['username']).first():
            logger.error("Username already exists")
            return jsonify({'message': 'Username already exists'}), 400
        if User.query.filter_by(phone=data['phone']).first():
            logger.error("Phone number already exists")
            return jsonify({'message': 'Phone number already exists'}), 400

        hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
        new_user = User(
            name=data['name'],
            username=data['username'],
            email=data['email'],
            password=hashed_password,
            phone=data['phone'],
            language=data['language'],
            gender=data['gender']
        )
        db.session.add(new_user)
        db.session.commit()
        logger.debug(f"User created: {new_user.id}")

        token = jwt.encode({
            'user_id': new_user.id,
            'exp': datetime.datetime.now(timezone.utc) + datetime.timedelta(days=1)
        }, current_app.config['SECRET_KEY'], algorithm='HS256')

        return jsonify({
            'message': 'User created successfully',
            'token': token,
            'user': {
                'id': new_user.id,
                'name': new_user.name,
                'username': new_user.username,
                'email': new_user.email,
                'phone': new_user.phone,
                'language': new_user.language,
                'gender': new_user.gender
            }
        }), 201

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error during signup: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@auth_bp.route('/signin', methods=['POST'])
def signin():
    """Handle user signin."""
    try:
        data = request.get_json()
        logger.debug(f"Signin data received: {data}")
        if not data or not all(k in data for k in ['email', 'password']):
            logger.error("Missing email or password")
            return jsonify({'message': 'Missing email or password'}), 400

        # Check for admin credentials first
        ADMIN_EMAIL = 'admin@healia.com'
        ADMIN_PASSWORD = 'healia123'
        
        if data['email'] == ADMIN_EMAIL and data['password'] == ADMIN_PASSWORD:
            token = jwt.encode({
                'user_id': 'admin',
                'is_admin': True,
                'exp': datetime.datetime.now(timezone.utc) + datetime.timedelta(days=1)
            }, current_app.config['SECRET_KEY'], algorithm='HS256')
            
            logger.debug("Admin signed in successfully")
            return jsonify({
                'message': 'Signed in successfully',
                'token': token,
                'user': {
                    'id': 'admin',
                    'name': 'Admin User',
                    'email': ADMIN_EMAIL,
                    'language': 'en',
                    'gender': 'unknown',
                    'is_admin': True
                }
            }), 200

        # Regular user authentication
        user = User.query.filter_by(email=data['email']).first()
        if not user or not bcrypt.check_password_hash(user.password, data['password']):
            logger.error("Invalid credentials")
            return jsonify({'message': 'Invalid credentials'}), 401

        token = jwt.encode({
            'user_id': user.id,
            'is_admin': False,
            'exp': datetime.datetime.now(timezone.utc) + datetime.timedelta(days=1)
        }, current_app.config['SECRET_KEY'], algorithm='HS256')

        logger.debug(f"User signed in: {user.id}")
        return jsonify({
            'message': 'Signed in successfully',
            'token': token,
            'user': {
                'id': user.id,
                'name': user.name,
                'email': user.email,
                'language': user.language,
                'gender': user.gender,
                'is_admin': False
            }
        }), 200

    except Exception as e:
        logger.error(f"Error during signin: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@auth_bp.route('/verify', methods=['GET'])
def verify():
    """Verify user token."""
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.error("Invalid or missing token")
            return jsonify({'message': 'Invalid or missing token'}), 401

        token = auth_header.split(' ')[1]
        data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        
        if data['user_id'] == 'admin':
            logger.debug("Token verified for admin")
            return jsonify({
                'user': {
                    'id': 'admin',
                    'name': 'Admin User',
                    'email': 'admin@healia.com',
                    'language': 'en',
                    'gender': 'unknown',
                    'is_admin': True
                }
            }), 200

        user = User.query.get(data['user_id'])
        if not user:
            logger.error("User not found")
            return jsonify({'message': 'User not found'}), 401

        logger.debug(f"Token verified for user: {user.id}")
        return jsonify({
            'user': {
                'id': user.id,
                'name': user.name,
                'email': user.email,
                'language': user.language,
                'gender': user.gender,
                'is_admin': False
            }
        }), 200

    except jwt.ExpiredSignatureError:
        logger.error("Token expired")
        return jsonify({'message': 'Token expired'}), 401
    except jwt.InvalidTokenError:
        logger.error("Invalid token")
        return jsonify({'message': 'Invalid token'}), 401
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@auth_bp.route('/conversations', methods=['GET'])
def conversations():
    """Get all conversations for the authenticated user with pagination."""
    logger.debug("Received conversations request")
    user = verify_user_from_token(request.headers.get('Authorization'))
    if not user:
        logger.error("Invalid or missing token")
        return jsonify({'message': 'Invalid or missing token'}), 401

    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))

        query = Conversation.query.filter_by(user_id=user.id)\
            .filter(Conversation.messages != None)\
            .filter(db.cast(Conversation.messages, db.Text) != '[]')\
            .order_by(Conversation.updated_at.desc().nullslast(), Conversation.created_at.desc())

        paginated_conversations = query.paginate(page=page, per_page=per_page, error_out=False)
        conversations = paginated_conversations.items
        total = paginated_conversations.total
        pages = paginated_conversations.pages

        logger.debug(f"Fetched {len(conversations)} conversations for user: {user.id}, page: {page}, total: {total}")
        return jsonify({
            'conversations': [{
                'id': conv.id,
                'created_at': conv.created_at.isoformat(),
                'updated_at': conv.updated_at.isoformat() if conv.updated_at else conv.created_at.isoformat(),
                'preview': conv.messages[-1]['text'][:100] if conv.messages and len(conv.messages) > 0 else 'No messages yet',
                'title': conv.messages[0]['text'][:30] if conv.messages and len(conv.messages) > 0 else 'Untitled Conversation',
                'message_count': len(conv.messages) if conv.messages else 0
            } for conv in conversations],
            'total': total,
            'pages': pages,
            'current_page': page
        }), 200
    except Exception as e:
        logger.error(f"Error fetching conversations: {str(e)}")
        return jsonify({'message': f'Error fetching conversations: {str(e)}'}), 500

@auth_bp.route('/conversation/<int:conversation_id>', methods=['GET', 'POST'])
def conversation(conversation_id):
    """Handle GET and POST requests for a specific conversation."""
    logger.debug(f"Received conversation request for ID: {conversation_id}")
    user = verify_user_from_token(request.headers.get('Authorization'))
    if not user:
        logger.error("Invalid or missing token")
        return jsonify({'message': 'Invalid or missing token'}), 401

    conversation = Conversation.query.filter_by(id=conversation_id, user_id=user.id).first()
    if request.method == 'GET':
        if not conversation:
            logger.debug(f"Conversation {conversation_id} not found")
            return jsonify({
                'id': conversation_id,
                'created_at': datetime.datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.datetime.now(timezone.utc).isoformat(),
                'messages': [],
                'preview': 'No messages yet'
            }), 200
        try:
            logger.debug(f"Fetched conversation: {conversation.id}")
            return jsonify({
                'id': conversation.id,
                'created_at': conversation.created_at.isoformat(),
                'updated_at': conversation.updated_at.isoformat() if conversation.updated_at else conversation.created_at.isoformat(),
                'messages': conversation.messages or [],
                'preview': conversation.messages[-1]['text'][:100] if conversation.messages and len(conversation.messages) > 0 else 'No messages yet'
            }), 200
        except Exception as e:
            logger.error(f"Error fetching conversation: {str(e)}")
            return jsonify({'message': f'Error fetching conversation: {str(e)}'}), 500

    elif request.method == 'POST':
        logger.debug(f"Processing POST request for conversation: {conversation_id}")
        try:
            data = request.get_json()
            logger.debug(f"Conversation data received: {data}")
            if not data or 'message' not in data:
                logger.error("Message is required")
                return jsonify({'message': 'Message is required'}), 400

            if not conversation:
                conversation = Conversation(
                    id=conversation_id,
                    user_id=user.id,
                    messages=[],
                    created_at=datetime.datetime.now(timezone.utc)
                )
                db.session.add(conversation)
                logger.debug(f"Created new conversation with ID: {conversation_id}")

            if conversation.messages is None:
                conversation.messages = []

            session_id = f"user_{user.id}_conv_{conversation.id}"
            user_message = {
                'id': str(uuid.uuid4()),
                'text': data['message'],
                'isUser': True,
                'timestamp': datetime.datetime.now(timezone.utc).isoformat()
            }
            conversation.messages.append(user_message)
            logger.debug(f"Added user message: {user_message}")

            # Use the safe medical profile access function
            patient_info = {
                'name': user.name,
                'language': user.language,
                'gender': user.gender,
                'age': get_safe_medical_profile_attr(user, 'age', 'N/A'),
                'chronic_conditions': get_safe_medical_profile_attr(user, 'chronic_conditions', 'None'),
                'allergies': get_safe_medical_profile_attr(user, 'allergies', 'None'),
                'region': get_safe_medical_profile_attr(user, 'region', 'N/A'),
                'city': get_safe_medical_profile_attr(user, 'city', 'N/A'),
                'profession': get_safe_medical_profile_attr(user, 'profession', 'N/A'),
                'marital_status': get_safe_medical_profile_attr(user, 'marital_status', 'N/A'),
                'lifestyle': get_safe_medical_profile_attr(user, 'lifestyle', {})
            }

            logger.debug(f"Calling generate_personalized_response with session_id: {session_id}, message: {data['message']}")
            try:
                ai_response_text = generate_personalized_response(data['message'], patient_info, session_id, conversation.messages)
                if not ai_response_text or not isinstance(ai_response_text, str) or ai_response_text.strip() == "":
                    logger.error(f"Invalid AI response text: {ai_response_text}")
                    ai_response_text = "I'm sorry, I couldn't generate a response. Please try again."
            except Exception as ai_error:
                logger.error(f"Error in generate_personalized_response: {str(ai_error)}")
                ai_response_text = "I'm sorry, I couldn't process your request due to an issue with the AI service. Please try again later."

            logger.info(f"AI response received: {ai_response_text[:100]}...")

            ai_message = {
                'id': str(uuid.uuid4()),
                'text': ai_response_text,
                'isUser': False,
                'timestamp': datetime.datetime.now(timezone.utc).isoformat()
            }
            conversation.messages.append(ai_message)
            logger.debug(f"Added AI message: {ai_message}")

            conversation.updated_at = datetime.datetime.now(timezone.utc)
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(conversation, "messages")
            try:
                db.session.commit()
                logger.debug(f"Conversation saved, message count: {len(conversation.messages)}")
            except Exception as db_error:
                logger.error(f"Database commit error: {str(db_error)}")
                db.session.rollback()
                return jsonify({'message': f'Error saving conversation: {str(db_error)}'}), 500

            response = {
                'id': conversation.id,
                'created_at': conversation.created_at.isoformat(),
                'updated_at': conversation.updated_at.isoformat(),
                'messages': conversation.messages,
                'preview': user_message['text'][:100]
            }
            logger.debug(f"Returning response: {response}")
            return jsonify(response), 200

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving conversation: {str(e)}")
            return jsonify({'message': f'Error saving conversation: {str(e)}'}), 500

@auth_bp.route('/conversation', methods=['GET', 'POST'])
def latest_conversation():
    """Handle GET and POST requests for the latest conversation."""
    logger.debug("Received latest conversation request")
    auth_header = request.headers.get('Authorization')
    user = verify_user_from_token(auth_header)
    if not user:
        logger.error(f"Token verification failed for header: {auth_header}")
        return jsonify({'message': 'Invalid or missing token'}), 401

    if request.method == 'GET':
        try:
            conversation = Conversation.query.filter_by(user_id=user.id)\
                .filter(Conversation.messages != None)\
                .filter(db.cast(Conversation.messages, db.Text) != '[]')\
                .order_by(Conversation.updated_at.desc().nullslast(), Conversation.created_at.desc())\
                .first()
            if not conversation:
                logger.debug("No conversation found")
                return jsonify({
                    'id': None,
                    'created_at': None,
                    'updated_at': None,
                    'messages': [],
                    'preview': 'No messages yet'
                }), 200
            logger.debug(f"Fetched conversation: {conversation.id}")
            return jsonify({
                'id': conversation.id,
                'created_at': conversation.created_at.isoformat(),
                'updated_at': conversation.updated_at.isoformat() if conversation.updated_at else conversation.created_at.isoformat(),
                'messages': conversation.messages or [],
                'preview': conversation.messages[-1]['text'][:100] if conversation.messages and len(conversation.messages) > 0 else 'No messages yet'
            }), 200
        except Exception as e:
            logger.error(f"Error fetching conversation: {str(e)}")
            return jsonify({'message': f'Error fetching conversation: {str(e)}'}), 500

    elif request.method == 'POST':
        logger.debug("Processing POST request for latest conversation")
        try:
            data = request.get_json()
            logger.debug(f"Conversation data received: {data}")
            if not data or 'message' not in data:
                logger.error("Message is required")
                return jsonify({'message': 'Message is required'}), 400

            if data['message'] == '':
                conversation = Conversation(
                    user_id=user.id,
                    messages=[],
                    created_at=datetime.datetime.now(timezone.utc)
                )
                db.session.add(conversation)
                db.session.commit()
                logger.debug(f"Created new conversation with ID: {conversation.id}")
                return jsonify({
                    'id': conversation.id,
                    'created_at': conversation.created_at.isoformat(),
                    'updated_at': conversation.updated_at.isoformat() if conversation.updated_at else conversation.created_at.isoformat(),
                    'messages': [],
                    'preview': 'No messages yet'
                }), 200

            conversation = Conversation.query.filter_by(user_id=user.id)\
                .filter(Conversation.messages != None)\
                .filter(db.cast(Conversation.messages, db.Text) != '[]')\
                .order_by(Conversation.updated_at.desc().nullslast(), Conversation.created_at.desc())\
                .first()
            if not conversation:
                conversation = Conversation(
                    user_id=user.id,
                    messages=[],
                    created_at=datetime.datetime.now(timezone.utc)
                )
                db.session.add(conversation)
                db.session.commit()
                logger.debug(f"Created new conversation with ID: {conversation.id}")

            session_id = f"user_{user.id}_conv_{conversation.id}"
            user_message = {
                'id': str(uuid.uuid4()),
                'text': data['message'],
                'isUser': True,
                'timestamp': datetime.datetime.now(timezone.utc).isoformat()
            }
            conversation.messages.append(user_message)
            logger.debug(f"Added user message: {user_message}")

            # Use the safe medical profile access function
            patient_info = {
                'name': user.name,
                'language': user.language,
                'gender': user.gender,
                'age': get_safe_medical_profile_attr(user, 'age', 'N/A'),
                'chronic_conditions': get_safe_medical_profile_attr(user, 'chronic_conditions', 'None'),
                'allergies': get_safe_medical_profile_attr(user, 'allergies', 'None'),
                'region': get_safe_medical_profile_attr(user, 'region', 'N/A'),
                'city': get_safe_medical_profile_attr(user, 'city', 'N/A'),
                'profession': get_safe_medical_profile_attr(user, 'profession', 'N/A'),
                'marital_status': get_safe_medical_profile_attr(user, 'marital_status', 'N/A'),
                'lifestyle': get_safe_medical_profile_attr(user, 'lifestyle', {})
            }

            logger.debug(f"Calling generate_personalized_response with session_id: {session_id}, message: {data['message']}")
            try:
                ai_response_text = generate_personalized_response(data['message'], patient_info, session_id, conversation.messages)
                if not ai_response_text or not isinstance(ai_response_text, str) or ai_response_text.strip() == "":
                    logger.error(f"Invalid AI response text: {ai_response_text}")
                    ai_response_text = "I'm sorry, I couldn't generate a response. Please try again."
            except Exception as ai_error:
                logger.error(f"Error in generate_personalized_response: {str(ai_error)}")
                ai_response_text = "I'm sorry, I couldn't process your request due to an issue with the AI service. Please try again later."

            logger.info(f"AI response received: {ai_response_text[:100]}...")

            ai_message = {
                'id': str(uuid.uuid4()),
                'text': ai_response_text,
                'isUser': False,
                'timestamp': datetime.datetime.now(timezone.utc).isoformat()
            }
            conversation.messages.append(ai_message)
            logger.debug(f"Added AI message: {ai_message}")

            conversation.updated_at = datetime.datetime.now(timezone.utc)
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(conversation, "messages")
            try:
                db.session.commit()
                logger.debug(f"Conversation saved, message count: {len(conversation.messages)}")
            except Exception as db_error:
                logger.error(f"Database commit error: {str(db_error)}")
                db.session.rollback()
                return jsonify({'message': f'Error saving conversation: {str(db_error)}'}), 500

            response = {
                'id': conversation.id,
                'created_at': conversation.created_at.isoformat(),
                'updated_at': conversation.updated_at.isoformat(),
                'messages': conversation.messages,
                'preview': user_message['text'][:100]
            }
            logger.debug(f"Returning response: {response}")
            return jsonify(response), 200

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving conversation: {str(e)}")
            return jsonify({'message': f'Error saving conversation: {str(e)}'}), 500

@auth_bp.route('/profile', methods=['GET'])
def get_profile():
    """Fetch the user's medical profile."""
    try:
        user = verify_user_from_token(request.headers.get('Authorization'))
        if not user:
            logger.error("Invalid or missing token")
            return jsonify({'message': 'Invalid or missing token'}), 401

        logger.debug(f"Fetching profile for user_id: {user.id}")
        profile = MedicalProfile.query.filter_by(user_id=user.id).first()
        
        if not profile:
            logger.debug(f"No profile found for user_id: {user.id}, returning empty profile")
            return jsonify({}), 200
        
        profile_data = {
            'firstName': profile.first_name or '',
            'lastName': profile.last_name or '',
            'phone': profile.phone or '',
            'dateOfBirth': profile.date_of_birth.isoformat() if profile.date_of_birth else '',
            'age': profile.age if profile.age is not None else '',
            'gender': profile.gender or '',
            'maritalStatus': profile.marital_status or '',
            'nationality': profile.nationality or 'Cameroonian',
            'region': profile.region or '',
            'city': profile.city or '',
            'quarter': profile.quarter or '',
            'address': profile.address or '',
            'profession': profile.profession or '',
            'emergencyContact': profile.emergency_contact or '',
            'emergencyRelation': profile.emergency_relation or '',
            'emergencyPhone': profile.emergency_phone or '',
            'bloodType': profile.blood_type or '',
            'genotype': profile.genotype or '',
            'allergies': profile.allergies or '',
            'chronicConditions': profile.chronic_conditions or '',
            'medications': profile.medications or '',
            'primaryHospital': profile.primary_hospital or '',
            'primaryPhysician': profile.primary_physician or '',
            'medicalHistory': profile.medical_history or '',
            'vaccinationHistory': profile.vaccination_history or '',
            'lastDentalVisit': profile.last_dental_visit.isoformat() if profile.last_dental_visit else '',
            'lastEyeExam': profile.last_eye_exam.isoformat() if profile.last_eye_exam else '',
            'lifestyle': profile.lifestyle or {
                'smokes': False,
                'alcohol': 'Never',
                'exercise': 'Never',
                'diet': 'Balanced'
            },
            'familyHistory': profile.family_history or ''
        }
        logger.debug(f"Profile fetched successfully for user_id: {user.id}")
        return jsonify(profile_data), 200
    except Exception as e:
        logger.error(f"Error fetching profile for user_id {user.id if 'user' in locals() else 'unknown'}: {str(e)}")
        return jsonify({'message': f'Error fetching profile: {str(e)}'}), 500

@auth_bp.route('/profile', methods=['POST'])
def save_profile():
    """Save or update the user's medical profile."""
    try:
        user = verify_user_from_token(request.headers.get('Authorization'))
        if not user:
            logger.error("Invalid or missing token")
            return jsonify({'message': 'Invalid or missing token'}), 401

        logger.debug(f"Saving profile for user_id: {user.id}")
        data = request.get_json()
        if not data:
            logger.error("No data provided in request")
            return jsonify({'message': 'No data provided'}), 400
        
        profile = MedicalProfile.query.filter_by(user_id=user.id).first()
        if not profile:
            logger.debug(f"No existing profile for user_id: {user.id}, creating new")
            profile = MedicalProfile(user_id=user.id)
            db.session.add(profile)
        
        # Update fields with validation
        profile.first_name = data.get('firstName', profile.first_name or '')
        profile.last_name = data.get('lastName', profile.last_name or '')
        profile.phone = data.get('phone', profile.phone or '')
        
        # Handle date fields safely
        try:
            profile.date_of_birth = datetime.datetime.strptime(data['dateOfBirth'], '%Y-%m-%d').date() if data.get('dateOfBirth') and data['dateOfBirth'].strip() else profile.date_of_birth
        except (ValueError, TypeError):
            logger.warning(f"Invalid dateOfBirth format: {data.get('dateOfBirth')}")
            profile.date_of_birth = profile.date_of_birth
        
        profile.gender = data.get('gender', profile.gender or '')
        profile.marital_status = data.get('maritalStatus', profile.marital_status or '')
        profile.nationality = data.get('nationality', profile.nationality or 'Cameroonian')
        profile.region = data.get('region', profile.region or '')
        profile.city = data.get('city', profile.city or '')
        profile.quarter = data.get('quarter', profile.quarter or '')
        profile.address = data.get('address', profile.address or '')
        profile.profession = data.get('profession', profile.profession or '')
        profile.emergency_contact = data.get('emergencyContact', profile.emergency_contact or '')
        profile.emergency_relation = data.get('emergencyRelation', profile.emergency_relation or '')
        profile.emergency_phone = data.get('emergencyPhone', profile.emergency_phone or '')
        profile.blood_type = data.get('bloodType', profile.blood_type or '')
        profile.genotype = data.get('genotype', profile.genotype or '')
        profile.allergies = data.get('allergies', profile.allergies or '')
        profile.chronic_conditions = data.get('chronicConditions', profile.chronic_conditions or '')
        profile.medications = data.get('medications', profile.medications or '')
        profile.primary_hospital = data.get('primaryHospital', profile.primary_hospital or '')
        profile.primary_physician = data.get('primaryPhysician', profile.primary_physician or '')
        profile.medical_history = data.get('medicalHistory', profile.medical_history or '')
        profile.vaccination_history = data.get('vaccinationHistory', profile.vaccination_history or '')
        
        # Handle date fields safely
        try:
            profile.last_dental_visit = datetime.datetime.strptime(data['lastDentalVisit'], '%Y-%m-%d').date() if data.get('lastDentalVisit') and data['lastDentalVisit'].strip() else profile.last_dental_visit
        except (ValueError, TypeError):
            logger.warning(f"Invalid lastDentalVisit format: {data.get('lastDentalVisit')}")
            profile.last_dental_visit = profile.last_dental_visit
        
        try:
            profile.last_eye_exam = datetime.datetime.strptime(data['lastEyeExam'], '%Y-%m-%d').date() if data.get('lastEyeExam') and data['lastEyeExam'].strip() else profile.last_eye_exam
        except (ValueError, TypeError):
            logger.warning(f"Invalid lastEyeExam format: {data.get('lastEyeExam')}")
            profile.last_eye_exam = profile.last_eye_exam
        
        profile.lifestyle = data.get('lifestyle', profile.lifestyle or {
            'smokes': False,
            'alcohol': 'Never',
            'exercise': 'Never',
            'diet': 'Balanced'
        })
        profile.family_history = data.get('familyHistory', profile.family_history or '')

        # updated_at will be set automatically by SQLAlchemy onupdate
        
        try:
            db.session.commit()
            logger.debug(f"Profile saved successfully for user_id: {user.id}")
        except Exception as db_error:
            db.session.rollback()
            logger.error(f"Database commit error for user_id {user.id}: {str(db_error)}")
            return jsonify({'message': f'Database error: {str(db_error)}'}), 500
        
        profile_data = {
            'firstName': profile.first_name or '',
            'lastName': profile.last_name or '',
            'phone': profile.phone or '',
            'dateOfBirth': profile.date_of_birth.isoformat() if profile.date_of_birth else '',
            'age': profile.age if profile.age is not None else '',
            'gender': profile.gender or '',
            'maritalStatus': profile.marital_status or '',
            'nationality': profile.nationality or 'Cameroonian',
            'region': profile.region or '',
            'city': profile.city or '',
            'quarter': profile.quarter or '',
            'address': profile.address or '',
            'profession': profile.profession or '',
            'emergencyContact': profile.emergency_contact or '',
            'emergencyRelation': profile.emergency_relation or '',
            'emergencyPhone': profile.emergency_phone or '',
            'bloodType': profile.blood_type or '',
            'genotype': profile.genotype or '',
            'allergies': profile.allergies or '',
            'chronicConditions': profile.chronic_conditions or '',
            'medications': profile.medications or '',
            'primaryHospital': profile.primary_hospital or '',
            'primaryPhysician': profile.primary_physician or '',
            'medicalHistory': profile.medical_history or '',
            'vaccinationHistory': profile.vaccination_history or '',
            'lastDentalVisit': profile.last_dental_visit.isoformat() if profile.last_dental_visit else '',
            'lastEyeExam': profile.last_eye_exam.isoformat() if profile.last_eye_exam else '',
            'lifestyle': profile.lifestyle or {
                'smokes': False,
                'alcohol': 'Never',
                'exercise': 'Never',
                'diet': 'Balanced'
            },
            'familyHistory': profile.family_history or ''
        }
        logger.debug(f"Returning profile data for user_id: {user.id}")
        return jsonify({'profile': profile_data}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving profile for user_id {user.id if 'user' in locals() else 'unknown'}: {str(e)}")
        return jsonify({'message': f'Error saving profile: {str(e)}'}), 500

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        logger.debug(f"Verifying token with auth_header: {auth_header}")
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.error("Invalid or missing Authorization header")
            return jsonify({'message': 'Invalid or missing token'}), 401

        token = auth_header.split(' ')[1].strip()
        if not token:
            logger.error("Token is empty after splitting Authorization header")
            return jsonify({'message': 'Token is missing'}), 401

        try:
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'], options={'verify_exp': True})
            logger.debug(f"Token verified for user_id: {data.get('user_id', 'unknown')}")
            return f(data, *args, **kwargs)
        except jwt.ExpiredSignatureError:
            logger.error("Token has expired")
            return jsonify({'message': 'Token expired'}), 401
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {str(e)}")
            return jsonify({'message': 'Invalid token'}), 401
        except Exception as e:
            logger.error(f"Unexpected error verifying token: {str(e)}")
            return jsonify({'message': f'Error: {str(e)}'}), 500

    return decorated

@admin_bp.route('/users', methods=['GET'])
@token_required
def get_users(token_data):
    """
    Fetch all users for the admin dashboard with pagination.
    Query parameters: page (default 1), per_page (default 10)
    """
    if not token_data.get('is_admin', False):
        logger.warning(f"Non-admin access attempt by user_id: {token_data.get('user_id', 'unknown')}")
        return jsonify({'message': 'Admin access required'}), 403
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))

        # Query users with their session data, sorted by id
        users_query = User.query.order_by(User.id.desc())
        paginated_users = users_query.paginate(page=page, per_page=per_page, error_out=False)
        users = paginated_users.items
        total = paginated_users.total
        pages = paginated_users.pages

        user_data = []
        for user in users:
            # Calculate session metrics
            sessions = UserSession.query.filter_by(user_id=user.id).all()
            total_sessions = len(sessions)
            avg_session_time = 0
            if sessions:
                session_durations = [
                    (s.end_time - s.start_time).total_seconds() / 60
                    for s in sessions if s.end_time
                ]
                avg_session_time = round(mean(session_durations), 1) if session_durations else 0

            # Use created_at from User model to determine last active and status
            last_session = UserSession.query.filter_by(user_id=user.id)\
                .order_by(UserSession.start_time.desc()).first()
            last_active = last_session.start_time if last_session else user.created_at
            status = 'active' if last_active and (datetime.now(timezone.utc) - last_active).days < 30 else 'inactive'

            user_data.append({
                'id': user.id,
                'name': user.name,
                'email': user.email,
                'status': status,
                'last_active': last_active.isoformat() if last_active else None,
                'total_sessions': total_sessions,
                'avg_session_time': avg_session_time
            })

        logger.info(f"Fetched {len(user_data)} users for admin_id: {token_data.get('user_id', 'admin')}, page: {page}")
        return jsonify({
            'users': user_data,
            'total': total,
            'pages': pages,
            'current_page': page
        }), 200
    except Exception as e:
        logger.error(f"Error in get_users: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@admin_bp.route('/analytics/symptom_trends', methods=['GET'])
@token_required
def get_symptom_trends(token_data):
    """
    Fetch symptom trends data for the admin dashboard across all users.
    Query parameter: time_range (24h, 7d, 30d, 90d)
    """
    if not token_data.get('is_admin', False):
        logger.warning(f"Non-admin access attempt by user_id: {token_data.get('user_id', 'unknown')}")
        return jsonify({'message': 'Admin access required'}), 403
    try:
        time_range = request.args.get('time_range', '7d')
        if time_range not in ['24h', '7d', '30d', '90d']:
            logger.warning(f"Invalid time_range: {time_range}")
            return jsonify({'message': 'Invalid time range. Use 24h, 7d, 30d, or 90d'}), 400

        analyzer = HealthAnalyzer()
        data = analyzer.analyze_symptom_trends(time_range)
        logger.info(f"Retrieved symptom trends for time_range: {time_range}, admin_id: {token_data.get('user_id', 'admin')}")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in get_symptom_trends: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@admin_bp.route('/analytics/sentiment', methods=['GET'])
@token_required
def get_sentiment_analysis(token_data):
    """
    Fetch sentiment analysis data for the admin dashboard across specified conversations.
    Query parameter: conversation_ids (comma-separated list of IDs)
    """
    if not token_data.get('is_admin', False):
        logger.warning(f"Non-admin access attempt by user_id: {token_data.get('user_id', 'unknown')}")
        return jsonify({'message': 'Admin access required'}), 403
    try:
        conversation_ids = request.args.get('conversation_ids', '')
        if not conversation_ids:
            logger.warning("No conversation IDs provided for sentiment analysis")
            return jsonify({'message': 'Conversation IDs required'}), 400

        try:
            conversation_ids = [int(cid) for cid in conversation_ids.split(',')]
        except ValueError:
            logger.warning(f"Invalid conversation IDs: {conversation_ids}")
            return jsonify({'message': 'Invalid conversation IDs'}), 400

        analyzer = HealthAnalyzer()
        data = analyzer.analyze_sentiment(conversation_ids)
        logger.info(f"Retrieved sentiment analysis for {len(conversation_ids)} conversations, admin_id: {token_data.get('user_id', 'admin')}")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in get_sentiment_analysis: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@admin_bp.route('/analytics/diagnostic_patterns', methods=['GET'])
@token_required
def get_diagnostic_patterns(token_data):
    """
    Fetch diagnostic patterns data for the admin dashboard across all users.
    Query parameter: time_range (24h, 7d, 30d, 90d)
    """
    if not token_data.get('is_admin', False):
        logger.warning(f"Non-admin access attempt by user_id: {token_data.get('user_id', 'unknown')}")
        return jsonify({'message': 'Admin access required'}), 403
    try:
        time_range = request.args.get('time_range', '7d')
        if time_range not in ['24h', '7d', '30d', '90d']:
            logger.warning(f"Invalid time_range: {time_range}")
            return jsonify({'message': 'Invalid time range. Use 24h, 7d, 30d, or 90d'}), 400

        analyzer = HealthAnalyzer()
        data = analyzer.analyze_diagnostic_patterns(time_range)
        logger.info(f"Retrieved diagnostic patterns for time_range: {time_range}, admin_id: {token_data.get('user_id', 'admin')}")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in get_diagnostic_patterns: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@admin_bp.route('/analytics/communication_metrics', methods=['GET'])
@token_required
def get_communication_metrics(token_data):
    """
    Fetch communication metrics data for the admin dashboard across all users.
    Query parameter: time_range (24h, 7d, 30d, 90d)
    """
    if not token_data.get('is_admin', False):
        logger.warning(f"Non-admin access attempt by user_id: {token_data.get('user_id', 'unknown')}")
        return jsonify({'message': 'Admin access required'}), 403
    try:
        time_range = request.args.get('time_range', '7d')
        if time_range not in ['24h', '7d', '30d', '90d']:
            logger.warning(f"Invalid time_range: {time_range}")
            return jsonify({'message': 'Invalid time range. Use 24h, 7d, 30d, or 90d'}), 400

        analyzer = HealthAnalyzer()
        data = analyzer.analyze_communication_metrics(time_range)
        logger.info(f"Retrieved communication metrics for time_range: {time_range}, admin_id: {token_data.get('user_id', 'admin')}")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in get_communication_metrics: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@admin_bp.route('/analytics/user_activity', methods=['GET'])
@token_required
def get_user_activity(token_data):
    """
    Fetch user activity data for the admin dashboard across all users.
    Query parameter: time_range (24h, 7d)
    """
    if not token_data.get('is_admin', False):
        logger.warning(f"Non-admin access attempt by user_id: {token_data.get('user_id', 'unknown')}")
        return jsonify({'message': 'Admin access required'}), 403
    try:
        time_range = request.args.get('time_range', '24h')
        if time_range not in ['24h', '7d']:
            logger.warning(f"Invalid time_range: {time_range}")
            return jsonify({'message': 'Invalid time range. Use 24h or 7d'}), 400

        analyzer = HealthAnalyzer()
        data = analyzer.analyze_user_activity(time_range)
        logger.info(f"Retrieved user activity for time_range: {time_range}, admin_id: {token_data.get('user_id', 'admin')}")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in get_user_activity: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@admin_bp.route('/analytics/health_alerts', methods=['GET'])
@token_required
def get_health_alerts(token_data):
    """
    Fetch health alerts data for the admin dashboard across all users.
    """
    if not token_data.get('is_admin', False):
        logger.warning(f"Non-admin access attempt by user_id: {token_data.get('user_id', 'unknown')}")
        return jsonify({'message': 'Admin access required'}), 403
    try:
        analyzer = HealthAnalyzer()
        data = analyzer.generate_health_alerts()
        logger.info(f"Retrieved {len(data)} health alerts, admin_id: {token_data.get('user_id', 'admin')}")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in get_health_alerts: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@admin_bp.route('/analytics/treatment_preferences', methods=['GET'])
@token_required
def get_treatment_preferences(token_data):
    """
    Fetch treatment preferences data for the admin dashboard across all users.
    Query parameter: time_range (24h, 7d, 30d, 90d)
    """
    if not token_data.get('is_admin', False):
        logger.warning(f"Non-admin access attempt by user_id: {token_data.get('user_id', 'unknown')}")
        return jsonify({'message': 'Admin access required'}), 403
    try:
        time_range = request.args.get('time_range', '7d')
        if time_range not in ['24h', '7d', '30d', '90d']:
            logger.warning(f"Invalid time_range: {time_range}")
            return jsonify({'message': 'Invalid time range. Use 24h, 7d, 30d, or 90d'}), 400

        analyzer = HealthAnalyzer()
        data = analyzer.analyze_treatment_preferences(time_range)
        logger.info(f"Retrieved treatment preferences for time_range: {time_range}, admin_id: {token_data.get('user_id', 'admin')}")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in get_treatment_preferences: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@admin_bp.route('/analytics/health_literacy', methods=['GET'])
@token_required
def get_health_literacy(token_data):
    """
    Fetch health literacy data for the admin dashboard across all users.
    Query parameter: time_range (24h, 7d, 30d, 90d)
    """
    if not token_data.get('is_admin', False):
        logger.warning(f"Non-admin access attempt by user_id: {token_data.get('user_id', 'unknown')}")
        return jsonify({'message': 'Admin access required'}), 403
    try:
        time_range = request.args.get('time_range', '7d')
        if time_range not in ['24h', '7d', '30d', '90d']:
            logger.warning(f"Invalid time_range: {time_range}")
            return jsonify({'message': 'Invalid time range. Use 24h, 7d, 30d, or 90d'}), 400

        analyzer = HealthAnalyzer()
        data = analyzer.analyze_health_literacy(time_range)
        logger.info(f"Retrieved health literacy data for time_range: {time_range}, admin_id: {token_data.get('user_id', 'admin')}")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in get_health_literacy: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@admin_bp.route('/analytics/workflow_metrics', methods=['GET'])
@token_required
def get_workflow_metrics(token_data):
    """
    Fetch workflow metrics data for the admin dashboard across all users.
    Query parameter: time_range (24h, 7d, 30d, 90d)
    """
    if not token_data.get('is_admin', False):
        logger.warning(f"Non-admin access attempt by user_id: {token_data.get('user_id', 'unknown')}")
        return jsonify({'message': 'Admin access required'}), 403
    try:
        time_range = request.args.get('time_range', '7d')
        if time_range not in ['24h', '7d', '30d', '90d']:
            logger.warning(f"Invalid time_range: {time_range}")
            return jsonify({'message': 'Invalid time range. Use 24h, 7d, 30d, or 90d'}), 400

        analyzer = HealthAnalyzer()
        data = analyzer.analyze_workflow_metrics(time_range)
        logger.info(f"Retrieved workflow metrics for time_range: {time_range}, admin_id: {token_data.get('user_id', 'admin')}")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in get_workflow_metrics: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@admin_bp.route('/analytics/ai_performance', methods=['GET'])
@token_required
def get_ai_performance(token_data):
    """
    Fetch AI performance metrics data for the admin dashboard across all users.
    Query parameter: time_range (24h, 7d, 30d, 90d)
    """
    if not token_data.get('is_admin', False):
        logger.warning(f"Non-admin access attempt by user_id: {token_data.get('user_id', 'unknown')}")
        return jsonify({'message': 'Admin access required'}), 403
    try:
        time_range = request.args.get('time_range', '7d')
        if time_range not in ['24h', '7d', '30d', '90d']:
            logger.warning(f"Invalid time_range: {time_range}")
            return jsonify({'message': 'Invalid time range. Use 24h, 7d, 30d, or 90d'}), 400

        analyzer = HealthAnalyzer()
        data = analyzer.analyze_ai_performance(time_range)
        logger.info(f"Retrieved AI performance metrics for time_range: {time_range}, admin_id: {token_data.get('user_id', 'admin')}")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in get_ai_performance: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@admin_bp.route('/analytics/conversations', methods=['GET'])
@token_required
def get_all_conversations(token_data):
    """
    Fetch all conversations across all users for the admin dashboard (for sentiment analysis).
    Query parameters: page (default 1), per_page (default 10)
    """
    if not token_data.get('is_admin', False):
        logger.warning(f"Non-admin access attempt by user_id: {token_data.get('user_id', 'unknown')}")
        return jsonify({'message': 'Admin access required'}), 403
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))

        conversations = Conversation.query\
            .filter(Conversation.messages != None)\
            .filter(db.cast(Conversation.messages, db.Text) != '[]')\
            .order_by(Conversation.updated_at.desc().nullslast())\
            .paginate(page=page, per_page=per_page, error_out=False)

        data = {
            'conversations': [{
                'id': conv.id,
                'user_id': conv.user_id,
                'created_at': conv.created_at.isoformat(),
                'updated_at': conv.updated_at.isoformat() if conv.updated_at else None,
                'preview': conv.messages[-1]['text'][:100] if conv.messages else 'No messages'
            } for conv in conversations.items],
            'total': conversations.total,
            'pages': conversations.pages,
            'current_page': page
        }
        logger.info(f"Retrieved {len(data['conversations'])} conversations, admin_id: {token_data.get('user_id', 'admin')}")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error in get_all_conversations: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@auth_bp.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint to keep the service alive."""
    logger.debug("Received ping request")
    return jsonify({'status': 'alive'}), 200
