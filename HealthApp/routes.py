from flask import Blueprint, request, jsonify, current_app
from flask_cors import CORS
import jwt
import datetime
from datetime import timezone
import uuid
import logging
from . import db, bcrypt
from .models import User, Conversation
from .ai_engine import generate_personalized_response
from .speech import generate_tts_audio

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')
CORS(auth_bp, origins=["http://localhost:5173", "https://healia.netlify.app"], supports_credentials=True)

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
            'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
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
            'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
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
                'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
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

            is_mic_input = data.get('isMicInput', False)
            if not conversation:
                conversation = Conversation(
                    id=conversation_id,
                    user_id=user.id,
                    messages=[],
                    created_at=datetime.datetime.now(datetime.timezone.utc)
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
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
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
            ai_response_text = generate_personalized_response(data['message'], patient_info, session_id, conversation.messages)
            logger.debug(f"AI response received: {ai_response_text}")

            if ai_response_text.startswith("Error:"):
                logger.error(f"AI response error: {ai_response_text}")
                ai_response_text = "I'm sorry, I couldn't process your request due to an issue with the AI service. Please try again later."
                audio_base64 = None
            else:
                audio_base64 = None
                if is_mic_input:
                    language = patient_info.get('language', 'en')
                    audio_base64 = generate_tts_audio(ai_response_text, user.id, conversation.id, language)
                    if not audio_base64:
                        logger.warning("Failed to generate speech, proceeding without audio")

            ai_message = {
                'id': str(uuid.uuid4()),
                'text': ai_response_text,
                'isUser': False,
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            conversation.messages.append(ai_message)
            logger.debug(f"Added AI message: {ai_message}")

            conversation.updated_at = datetime.datetime.now(datetime.timezone.utc)
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(conversation, "messages")
            db.session.commit()
            logger.debug(f"Conversation saved, message count: {len(conversation.messages)}")
            response = {
                'id': conversation.id,
                'created_at': conversation.created_at.isoformat(),
                'updated_at': conversation.updated_at.isoformat(),
                'messages': conversation.messages,
                'preview': user_message['text'][:100],
                'audio': audio_base64 if is_mic_input else None
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

            is_mic_input = data.get('isMicInput', False)
            if data['message'] == '':
                conversation = Conversation(
                    user_id=user.id,
                    messages=[],
                    created_at=datetime.datetime.now(datetime.timezone.utc)
                )
                db.session.add(conversation)
                db.session.commit()
                logger.debug(f"Created new conversation with ID: {conversation.id}")
                return jsonify({
                    'id': conversation.id,
                    'created_at': conversation.created_at.isoformat(),
                    'updated_at': conversation.updated_at.isoformat() if conversation.updated_at else conversation.created_at.isoformat(),
                    'messages': [],
                    'preview': 'No messages yet',
                    'audio': None
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
                    created_at=datetime.datetime.now(datetime.timezone.utc)
                )
                db.session.add(conversation)
                db.session.commit()
                logger.debug(f"Created new conversation with ID: {conversation.id}")

            session_id = f"user_{user.id}_conv_{conversation.id}"
            user_message = {
                'id': str(uuid.uuid4()),
                'text': data['message'],
                'isUser': True,
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
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
            ai_response_text = generate_personalized_response(data['message'], patient_info, session_id, conversation.messages)
            logger.debug(f"AI response received: {ai_response_text}")

            if ai_response_text.startswith("Error:"):
                logger.error(f"AI response error: {ai_response_text}")
                ai_response_text = "I'm sorry, I couldn't process your request due to an issue with the AI service. Please try again later."
                audio_base64 = None
            else:
                audio_base64 = None
                if is_mic_input:
                    language = patient_info.get('language', 'en')
                    audio_base64 = generate_tts_audio(ai_response_text, user.id, conversation.id, language)
                    if not audio_base64:
                        logger.warning("Failed to generate speech, proceeding without audio")

            ai_message = {
                'id': str(uuid.uuid4()),
                'text': ai_response_text,
                'isUser': False,
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            conversation.messages.append(ai_message)
            logger.debug(f"Added AI message: {ai_message}")

            conversation.updated_at = datetime.datetime.now(datetime.timezone.utc)
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(conversation, "messages")
            db.session.commit()
            logger.debug(f"Conversation saved, message count: {len(conversation.messages)}")
            response = {
                'id': conversation.id,
                'created_at': conversation.created_at.isoformat(),
                'updated_at': conversation.updated_at.isoformat(),
                'messages': conversation.messages,
                'preview': user_message['text'][:100],
                'audio': audio_base64 if is_mic_input else None
            }
            logger.debug(f"Returning response: {response}")
            return jsonify(response), 200

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving conversation: {str(e)}")
            return jsonify({'message': f'Error saving conversation: {str(e)}'}), 500