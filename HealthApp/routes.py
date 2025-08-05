from flask import Blueprint, request, jsonify, current_app
import jwt
import datetime
from datetime import timezone
import uuid
import logging
from . import db, bcrypt
from .models import User, Conversation, MedicalProfile
from flask_jwt_extended import jwt_required
from .ai_engine import generate_personalized_response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

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

        user = User.query.filter_by(email=data['email']).first()
        if not user or not bcrypt.check_password_hash(user.password, data['password']):
            logger.error("Invalid credentials")
            return jsonify({'message': 'Invalid credentials'}), 401

        token = jwt.encode({
            'user_id': user.id,
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
                'gender': user.gender
            }
        }), 200

    except Exception as e:
        logger.error(f"Error during signin: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@auth_bp.route('/verify', methods=['GET'])
def verify():
    """Verify user token."""
    user = verify_user_from_token(request.headers.get('Authorization'))
    if not user:
        logger.error("Invalid or missing token")
        return jsonify({'message': 'Invalid or missing token'}), 401

    logger.debug(f"Token verified for user: {user.id}")
    return jsonify({
        'user': {
            'id': user.id,
            'name': user.name,
            'email': user.email,
            'language': user.language,
            'gender': user.gender
        }
    }), 200

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

            patient_info = {
                'name': user.name,
                'language': user.language,
                'gender': user.gender,
                'age': getattr(user.medical_profile, 'age', 'N/A'),
                'chronic_conditions': getattr(user.medical_profile, 'chronic_conditions', 'None'),
                'allergies': getattr(user.medical_profile, 'allergies', 'None'),
                'region': getattr(user.medical_profile, 'region', 'N/A'),
                'city': getattr(user.medical_profile, 'city', 'N/A'),
                'profession': getattr(user.medical_profile, 'profession', 'N/A'),
                'marital_status': getattr(user.medical_profile, 'marital_status', 'N/A'),
                'lifestyle': getattr(user.medical_profile, 'lifestyle', {})
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

            patient_info = {
                'name': user.name,
                'language': user.language,
                'gender': user.gender,
                'age': getattr(user.medical_profile, 'age', 'N/A'),
                'chronic_conditions': getattr(user.medical_profile, 'chronic_conditions', 'None'),
                'allergies': getattr(user.medical_profile, 'allergies', 'None'),
                'region': getattr(user.medical_profile, 'region', 'N/A'),
                'city': getattr(user.medical_profile, 'city', 'N/A'),
                'profession': getattr(user.medical_profile, 'profession', 'N/A'),
                'marital_status': getattr(user.medical_profile, 'marital_status', 'N/A'),
                'lifestyle': getattr(user.medical_profile, 'lifestyle', {})
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

@auth_bp.route('/profile', methods=['GET', 'POST'])
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

        profile.updated_at = datetime.datetime.utcnow()
        
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

@auth_bp.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint to keep the service alive."""
    logger.debug("Received ping request")
    return jsonify({'status': 'alive'}), 200