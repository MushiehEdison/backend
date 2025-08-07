from flask import Blueprint, request, jsonify, current_app
import jwt
import datetime
from datetime import timezone
import uuid
import logging
from functools import wraps
from . import db, bcrypt
from .models import User, Conversation, MedicalProfile, UserSession, SymptomEntry, Diagnosis, SentimentRecord, HealthAlert, CommunicationMetric, TreatmentPreference, HealthLiteracy, WorkflowMetric, AIPerformance
from .ai_engine import generate_personalized_response
from .analysis import HealthAnalyzer
from statistics import mean
from sqlalchemy import func, and_
from dateutil.parser import parse
from datetime import timedelta

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
        user = None
        if decoded.get('user_id') == 'admin':
            return {'id': 'admin', 'is_admin': True}
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
            gender=data['gender'],
            created_at=datetime.datetime.now(timezone.utc)
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
                'gender': new_user.gender,
                'created_at': new_user.created_at.isoformat()
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
                    'is_admin': True,
                    'created_at': datetime.datetime.now(timezone.utc).isoformat()
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
                'is_admin': False,
                'created_at': user.created_at.isoformat()
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
                    'is_admin': True,
                    'created_at': datetime.datetime.now(timezone.utc).isoformat()
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
                'is_admin': False,
                'created_at': user.created_at.isoformat()
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
        logger.error("Invalid or missing token")
        return jsonify({'message': 'Invalid or missing token'}), 401

    try:
        if request.method == 'GET':
            conversation = Conversation.query.filter_by(user_id=user.id)\
                .order_by(Conversation.updated_at.desc().nullslast(), Conversation.created_at.desc())\
                .first()
            if not conversation:
                logger.debug(f"No conversations found for user: {user.id}")
                return jsonify({
                    'id': None,
                    'created_at': datetime.datetime.now(timezone.utc).isoformat(),
                    'updated_at': datetime.datetime.now(timezone.utc).isoformat(),
                    'messages': [],
                    'preview': 'No messages yet'
                }), 200

            logger.debug(f"Fetched latest conversation: {conversation.id}")
            return jsonify({
                'id': conversation.id,
                'created_at': conversation.created_at.isoformat(),
                'updated_at': conversation.updated_at.isoformat() if conversation.updated_at else conversation.created_at.isoformat(),
                'messages': conversation.messages or [],
                'preview': conversation.messages[-1]['text'][:100] if conversation.messages and len(conversation.messages) > 0 else 'No messages yet'
            }), 200

        elif request.method == 'POST':
            data = request.get_json()
            logger.debug(f"Conversation data received: {data}")
            if not data or 'message' not in data:
                logger.error("Message is required")
                return jsonify({'message': 'Message is required'}), 400

            conversation = Conversation(
                user_id=user.id,
                messages=[],
                created_at=datetime.datetime.now(timezone.utc)
            )
            db.session.add(conversation)
            db.session.flush()  # Get the conversation ID

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
                logger.debug(f"New conversation saved, ID: {conversation.id}")
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
        logger.error(f"Error in latest_conversation: {str(e)}")
        return jsonify({'message': f'Error: {str(e)}'}), 500

@admin_bp.route('/users', methods=['GET'])
@jwt_required()
def get_users():
    """Get all users with pagination (admin only)."""
    logger.debug("Received get users request")
    auth_header = request.headers.get('Authorization')
    user = verify_user_from_token(auth_header)
    if not user or not hasattr(user, 'is_admin') or not user['is_admin']:
        logger.error("Admin access required")
        return jsonify({'message': 'Admin access required'}), 403

    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        
        query = User.query.order_by(User.created_at.desc())
        paginated_users = query.paginate(page=page, per_page=per_page, error_out=False)
        users = paginated_users.items
        total = paginated_users.total
        pages = paginated_users.pages

        logger.debug(f"Fetched {len(users)} users, page: {page}, total: {total}")
        return jsonify({
            'users': [{
                'id': user.id,
                'name': user.name,
                'email': user.email,
                'username': user.username,
                'phone': user.phone,
                'language': user.language,
                'gender': user.gender,
                'created_at': user.created_at.isoformat(),
                'status': 'active' if user.sessions and any(s.end_time is None for s in user.sessions) else 'inactive',
                'last_active': max([s.last_active.isoformat() for s in user.sessions], default=None),
                'total_sessions': len(user.sessions),
                'avg_session_time': mean([s.duration_seconds or 0 for s in user.sessions]) / 60 if user.sessions else 0
            } for user in users],
            'total': total,
            'pages': pages,
            'current_page': page
        }), 200
    except Exception as e:
        logger.error(f"Error fetching users: {str(e)}")
        return jsonify({'message': f'Error fetching users: {str(e)}'}), 500

@admin_bp.route('/analytics/user_activity', methods=['GET'])
@jwt_required()
def user_activity():
    """Get user activity data for the specified time range (admin only)."""
    logger.debug("Received user activity request")
    auth_header = request.headers.get('Authorization')
    user = verify_user_from_token(auth_header)
    if not user or not hasattr(user, 'is_admin') or not user['is_admin']:
        logger.error("Admin access required")
        return jsonify({'message': 'Admin access required'}), 403

    try:
        time_range = request.args.get('time_range', '7d')
        now = datetime.datetime.now(timezone.utc)
        if time_range == '24h':
            start_time = now - timedelta(hours=24)
            group_by = func.date_trunc('hour', UserSession.start_time)
        elif time_range == '7d':
            start_time = now - timedelta(days=7)
            group_by = func.date_trunc('day', UserSession.start_time)
        elif time_range == '30d':
            start_time = now - timedelta(days=30)
            group_by = func.date_trunc('day', UserSession.start_time)
        else:  # 90d
            start_time = now - timedelta(days=90)
            group_by = func.date_trunc('week', UserSession.start_time)

        activity = db.session.query(
            group_by.label('time'),
            func.count(func.distinct(UserSession.user_id)).label('users')
        ).filter(UserSession.start_time >= start_time)\
         .group_by(group_by)\
         .order_by(group_by).all()

        logger.debug(f"Fetched user activity for time range: {time_range}")
        return jsonify([{
            'hour' if time_range == '24h' else 'date': row.time.strftime('%Y-%m-%d %H:%00' if time_range == '24h' else '%Y-%m-%d'),
            'users': row.users
        } for row in activity]), 200
    except Exception as e:
        logger.error(f"Error fetching user activity: {str(e)}")
        return jsonify({'message': f'Error fetching user activity: {str(e)}'}), 500

@admin_bp.route('/analytics/user_metrics', methods=['GET'])
@jwt_required()
def user_metrics():
    """Get user metrics for the Users tab (admin only)."""
    logger.debug("Received user metrics request")
    auth_header = request.headers.get('Authorization')
    user = verify_user_from_token(auth_header)
    if not user or not hasattr(user, 'is_admin') or not user['is_admin']:
        logger.error("Admin access required")
        return jsonify({'message': 'Admin access required'}), 403

    try:
        time_range = request.args.get('time_range', '7d')
        now = datetime.datetime.now(timezone.utc)
        if time_range == '24h':
            start_time = now - timedelta(hours=24)
        elif time_range == '7d':
            start_time = now - timedelta(days=7)
        elif time_range == '30d':
            start_time = now - timedelta(days=30)
        else:  # 90d
            start_time = now - timedelta(days=90)

        # New users today
        new_users_today = User.query.filter(
            User.created_at >= now.replace(hour=0, minute=0, second=0, microsecond=0),
            User.created_at < now.replace(hour=23, minute=59, second=59, microsecond=999999)
        ).count()

        # Active users
        active_users = db.session.query(func.count(func.distinct(UserSession.user_id)))\
            .filter(UserSession.start_time >= start_time, UserSession.end_time.is_(None)).scalar()

        # User retention (users active in both current and previous period)
        prev_start_time = start_time - (now - start_time)
        current_active = set(
            u.user_id for u in UserSession.query.filter(UserSession.start_time >= start_time).distinct(UserSession.user_id)
        )
        prev_active = set(
            u.user_id for u in UserSession.query.filter(
                UserSession.start_time >= prev_start_time,
                UserSession.start_time < start_time
            ).distinct(UserSession.user_id)
        )
        retained_users = len(current_active & prev_active)
        prev_total = len(prev_active)
        user_retention = (retained_users / prev_total * 100) if prev_total > 0 else 0

        # User growth
        prev_users = User.query.filter(
            User.created_at >= prev_start_time,
            User.created_at < start_time
        ).count()
        current_users = User.query.filter(User.created_at >= start_time).count()
        user_growth = ((current_users - prev_users) / prev_users * 100) if prev_users > 0 else 0

        logger.debug(f"Fetched user metrics: new_users_today={new_users_today}, active_users={active_users}, user_retention={user_retention}")
        return jsonify({
            'new_users_today': new_users_today,
            'active_users': active_users,
            'user_retention': round(user_retention, 2),
            'user_growth': round(user_growth, 2)
        }), 200
    except Exception as e:
        logger.error(f"Error fetching user metrics: {str(e)}")
        return jsonify({'message': f'Error fetching user metrics: {str(e)}'}), 500

@admin_bp.route('/analytics/conversations', methods=['GET'])
@jwt_required()
def conversation_analytics():
    """Get conversation analytics with pagination (admin only)."""
    logger.debug("Received conversation analytics request")
    auth_header = request.headers.get('Authorization')
    user = verify_user_from_token(auth_header)
    if not user or not hasattr(user, 'is_admin') or not user['is_admin']:
        logger.error("Admin access required")
        return jsonify({'message': 'Admin access required'}), 403

    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))

        query = Conversation.query.filter(
            Conversation.messages != None,
            db.cast(Conversation.messages, db.Text) != '[]'
        ).order_by(Conversation.updated_at.desc().nullslast(), Conversation.created_at.desc())

        paginated_conversations = query.paginate(page=page, per_page=per_page, error_out=False)
        conversations = paginated_conversations.items
        total = paginated_conversations.total
        pages = paginated_conversations.pages

        logger.debug(f"Fetched {len(conversations)} conversations, page: {page}, total: {total}")
        return jsonify({
            'conversations': [{
                'id': conv.id,
                'user_id': conv.user_id,
                'created_at': conv.created_at.isoformat(),
                'updated_at': conv.updated_at.isoformat() if conv.updated_at else conv.created_at.isoformat(),
                'preview': conv.messages[-1]['text'][:100] if conv.messages and len(conv.messages) > 0 else 'No messages yet',
                'message_count': len(conv.messages) if conv.messages else 0
            } for conv in conversations],
            'total': total,
            'pages': pages,
            'current_page': page
        }), 200
    except Exception as e:
        logger.error(f"Error fetching conversation analytics: {str(e)}")
        return jsonify({'message': f'Error fetching conversation analytics: {str(e)}'}), 500

@admin_bp.route('/analytics/sentiment', methods=['GET'])
@jwt_required()
def sentiment_analysis():
    """Get sentiment analysis for specified conversations (admin only)."""
    logger.debug("Received sentiment analysis request")
    auth_header = request.headers.get('Authorization')
    user = verify_user_from_token(auth_header)
    if not user or not hasattr(user, 'is_admin') or not user['is_admin']:
        logger.error("Admin access required")
        return jsonify({'message': 'Admin access required'}), 403

    try:
        conversation_ids = request.args.get('conversation_ids', '')
        if not conversation_ids:
            logger.error("No conversation IDs provided")
            return jsonify({'message': 'No conversation IDs provided'}), 400

        conversation_ids = [int(cid) for cid in conversation_ids.split(',') if cid.isdigit()]
        if not conversation_ids:
            logger.error("Invalid conversation IDs")
            return jsonify({'message': 'Invalid conversation IDs'}), 400

        sentiments = SentimentRecord.query.filter(SentimentRecord.convo_id.in_(conversation_ids)).all()
        sentiment_summary = {}
        for sentiment in sentiments:
            sentiment_summary[sentiment.sentiment_category] = sentiment_summary.get(sentiment.sentiment_category, 0) + sentiment.percentage

        total = sum(sentiment_summary.values())
        result = [
            {'name': category, 'value': round(value / len(conversation_ids), 2)}
            for category, value in sentiment_summary.items()
        ] if total > 0 else [
            {'name': 'Very Positive', 'value': 0},
            {'name': 'Positive', 'value': 0},
            {'name': 'Neutral', 'value': 0},
            {'name': 'Negative', 'value': 0},
            {'name': 'Very Negative', 'value': 0}
        ]

        logger.debug(f"Fetched sentiment analysis for {len(conversation_ids)} conversations")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error fetching sentiment analysis: {str(e)}")
        return jsonify({'message': f'Error fetching sentiment analysis: {str(e)}'}), 500

@admin_bp.route('/analytics/symptom_trends', methods=['GET'])
@jwt_required()
def symptom_trends():
    """Get symptom trends for the specified time range (admin only)."""
    logger.debug("Received symptom trends request")
    auth_header = request.headers.get('Authorization')
    user = verify_user_from_token(auth_header)
    if not user or not hasattr(user, 'is_admin') or not user['is_admin']:
        logger.error("Admin access required")
        return jsonify({'message': 'Admin access required'}), 403

    try:
        time_range = request.args.get('time_range', '7d')
        now = datetime.datetime.now(timezone.utc)
        if time_range == '24h':
            start_time = now - timedelta(hours=24)
            group_by = func.date_trunc('hour', SymptomEntry.reported_at)
            date_format = '%Y-%m-%d %H:00'
        elif time_range == '7d':
            start_time = now - timedelta(days=7)
            group_by = func.date_trunc('day', SymptomEntry.reported_at)
            date_format = '%Y-%m-%d'
        elif time_range == '30d':
            start_time = now - timedelta(days=30)
            group_by = func.date_trunc('day', SymptomEntry.reported_at)
            date_format = '%Y-%m-%d'
        else:  # 90d
            start_time = now - timedelta(days=90)
            group_by = func.date_trunc('week', SymptomEntry.reported_at)
            date_format = '%Y-%m-%d'

        # Fetch all unique symptoms
        symptoms = db.session.query(SymptomEntry.symptom_name).distinct().all()
        symptoms = [s[0].lower() for s in symptoms]

        # Initialize result dictionary
        result = {}
        for symptom in symptoms:
            query = db.session.query(
                group_by.label('date'),
                func.count(SymptomEntry.id).label('count')
            ).filter(
                SymptomEntry.reported_at >= start_time,
                func.lower(SymptomEntry.symptom_name) == symptom
            ).group_by(group_by).order_by(group_by)

            for row in query.all():
                date_str = row.date.strftime(date_format)
                if date_str not in result:
                    result[date_str] = {'date': date_str}
                result[date_str][symptom] = row.count

        # Fill in missing dates with zeros
        current_date = start_time
        delta = timedelta(hours=1) if time_range == '24h' else timedelta(days=1) if time_range in ['7d', '30d'] else timedelta(weeks=1)
        while current_date <= now:
            date_str = current_date.strftime(date_format)
            if date_str not in result:
                result[date_str] = {'date': date_str}
                for symptom in symptoms:
                    result[date_str][symptom] = 0
            else:
                for symptom in symptoms:
                    result[date_str][symptom] = result[date_str].get(symptom, 0)
            current_date += delta

        # Convert to list and sort by date
        final_result = sorted(result.values(), key=lambda x: x['date'])
        logger.debug(f"Fetched symptom trends for {len(final_result)} time points")
        return jsonify(final_result), 200
    except Exception as e:
        logger.error(f"Error fetching symptom trends: {str(e)}")
        return jsonify({'message': f'Error fetching symptom trends: {str(e)}'}), 500

@admin_bp.route('/analytics/communication_metrics', methods=['GET'])
@jwt_required()
def communication_metrics():
    """Get communication metrics for the specified time range (admin only)."""
    logger.debug("Received communication metrics request")
    auth_header = request.headers.get('Authorization')
    user = verify_user_from_token(auth_header)
    if not user or not hasattr(user, 'is_admin') or not user['is_admin']:
        logger.error("Admin access required")
        return jsonify({'message': 'Admin access required'}), 403

    try:
        time_range = request.args.get('time_range', '7d')
        now = datetime.datetime.now(timezone.utc)
        if time_range == '24h':
            start_time = now - timedelta(hours=24)
        elif time_range == '7d':
            start_time = now - timedelta(days=7)
        elif time_range == '30d':
            start_time = now - timedelta(days=30)
        else:  # 90d
            start_time = now - timedelta(days=90)

        prev_start_time = start_time - (now - start_time)
        metrics = CommunicationMetric.query.filter(CommunicationMetric.recorded_at >= start_time).all()
        prev_metrics = CommunicationMetric.query.filter(
            CommunicationMetric.recorded_at >= prev_start_time,
            CommunicationMetric.recorded_at < start_time
        ).all()

        prev_values = {m.metric_name: m.current_value for m in prev_metrics}
        result = [{
            'metric': m.metric_name,
            'current': m.current_value,
            'previous': prev_values.get(m.metric_name, 0),
            'trend': 'up' if m.current_value > prev_values.get(m.metric_name, 0) else 'down' if m.current_value < prev_values.get(m.metric_name, 0) else 'stable'
        } for m in metrics]

        logger.debug(f"Fetched {len(result)} communication metrics")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error fetching communication metrics: {str(e)}")
        return jsonify({'message': f'Error fetching communication metrics: {str(e)}'}), 500

@admin_bp.route('/analytics/diagnostic_patterns', methods=['GET'])
@jwt_required()
def diagnostic_patterns():
    """Get diagnostic patterns for the specified time range (admin only)."""
    logger.debug("Received diagnostic patterns request")
    auth_header = request.headers.get('Authorization')
    user = verify_user_from_token(auth_header)
    if not user or not hasattr(user, 'is_admin') or not user['is_admin']:
        logger.error("Admin access required")
        return jsonify({'message': 'Admin access required'}), 403

    try:
        time_range = request.args.get('time_range', '7d')
        now = datetime.datetime.now(timezone.utc)
        if time_range == '24h':
            start_time = now - timedelta(hours=24)
        elif time_range == '7d':
            start_time = now - timedelta(days=7)
        elif time_range == '30d':
            start_time = now - timedelta(days=30)
        else:  # 90d
            start_time = now - timedelta(days=90)

        prev_start_time = start_time - (now - start_time)
        diagnoses = Diagnosis.query.filter(Diagnosis.created_at >= start_time).all()
        prev_diagnoses = Diagnosis.query.filter(
            Diagnosis.created_at >= prev_start_time,
            Diagnosis.created_at < start_time
        ).all()

        condition_counts = {}
        condition_accuracies = {}
        for diag in diagnoses:
            condition_counts[diag.condition_name] = condition_counts.get(diag.condition_name, 0) + 1
            condition_accuracies[diag.condition_name] = condition_accuracies.get(diag.condition_name, []) + [diag.accuracy or 0]

        prev_counts = {}
        for diag in prev_diagnoses:
            prev_counts[diag.condition_name] = prev_counts.get(diag.condition_name, 0) + 1

        result = [{
            'condition': condition,
            'frequency': count,
            'accuracy': round(mean(accuracies), 2) if accuracies else 0,
            'trend': 'up' if count > prev_counts.get(condition, 0) else 'down' if count < prev_counts.get(condition, 0) else 'stable'
        } for condition, count, accuracies in [
            (cond, count, condition_accuracies[cond]) for cond, count in condition_counts.items()
        ]]

        logger.debug(f"Fetched {len(result)} diagnostic patterns")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error fetching diagnostic patterns: {str(e)}")
        return jsonify({'message': f'Error fetching diagnostic patterns: {str(e)}'}), 500

@admin_bp.route('/analytics/health_alerts', methods=['GET'])
@jwt_required()
def health_alerts():
    """Get health alerts (admin only)."""
    logger.debug("Received health alerts request")
    auth_header = request.headers.get('Authorization')
    user = verify_user_from_token(auth_header)
    if not user or not hasattr(user, 'is_admin') or not user['is_admin']:
        logger.error("Admin access required")
        return jsonify({'message': 'Admin access required'}), 403

    try:
        alerts = HealthAlert.query.order_by(HealthAlert.created_at.desc()).limit(10).all()
        logger.debug(f"Fetched {len(alerts)} health alerts")
        return jsonify([{
            'title': alert.title,
            'description': alert.description,
            'severity': alert.severity,
            'type': alert.alert_type,
            'time': alert.created_at.isoformat(),
            'region': alert.region
        } for alert in alerts]), 200
    except Exception as e:
        logger.error(f"Error fetching health alerts: {str(e)}")
        return jsonify({'message': f'Error fetching health alerts: {str(e)}'}), 500

@admin_bp.route('/analytics/treatment_preferences', methods=['GET'])
@jwt_required()
def treatment_preferences():
    """Get treatment preference trends (admin only)."""
    logger.debug("Received treatment preferences request")
    auth_header = request.headers.get('Authorization')
    user = verify_user_from_token(auth_header)
    if not user or not hasattr(user, 'is_admin') or not user['is_admin']:
        logger.error("Admin access required")
        return jsonify({'message': 'Admin access required'}), 403

    try:
        time_range = request.args.get('time_range', '7d')
        now = datetime.datetime.now(timezone.utc)
        if time_range == '24h':
            start_time = now - timedelta(hours=24)
        elif time_range == '7d':
            start_time = now - timedelta(days=7)
        elif time_range == '30d':
            start_time = now - timedelta(days=30)
        else:  # 90d
            start_time = now - timedelta(days=90)

        preferences = TreatmentPreference.query.filter(TreatmentPreference.recorded_at >= start_time).all()
        pref_summary = {}
        for pref in preferences:
            pref_summary[pref.treatment_type] = pref_summary.get(pref.treatment_type, [])
            pref_summary[pref.treatment_type].append(pref.preference_score)

        prev_start_time = start_time - (now - start_time)
        prev_preferences = TreatmentPreference.query.filter(
            TreatmentPreference.recorded_at >= prev_start_time,
            TreatmentPreference.recorded_at < start_time
        ).all()
        prev_summary = {p.treatment_type: mean([p.preference_score]) for p in prev_preferences if prev_preferences}

        result = [{
            'treatment': t_type,
            'percentage': round(mean(scores), 2),
            'trend': 'up' if mean(scores) > prev_summary.get(t_type, 0) else 'down' if mean(scores) < prev_summary.get(t_type, 0) else 'stable'
        } for t_type, scores in pref_summary.items()]

        logger.debug(f"Fetched {len(result)} treatment preferences")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error fetching treatment preferences: {str(e)}")
        return jsonify({'message': f'Error fetching treatment preferences: {str(e)}'}), 500

@admin_bp.route('/analytics/health_literacy', methods=['GET'])
@jwt_required()
def health_literacy():
    """Get health literacy data by demographics (admin only)."""
    logger.debug("Received health literacy request")
    auth_header = request.headers.get('Authorization')
    user = verify_user_from_token(auth_header)
    if not user or not hasattr(user, 'is_admin') or not user['is_admin']:
        logger.error("Admin access required")
        return jsonify({'message': 'Admin access required'}), 403

    try:
        time_range = request.args.get('time_range', '7d')
        now = datetime.datetime.now(timezone.utc)
        if time_range == '24h':
            start_time = now - timedelta(hours=24)
        elif time_range == '7d':
            start_time = now - timedelta(days=7)
        elif time_range == '30d':
            start_time = now - timedelta(days=30)
        else:  # 90d
            start_time = now - timedelta(days=90)

        literacy = HealthLiteracy.query.filter(HealthLiteracy.recorded_at >= start_time).all()
        literacy_summary = {}
        for record in literacy:
            group = record.age_group
            literacy_summary[group] = literacy_summary.get(group, {'understanding': [], 'engagement': []})
            literacy_summary[group]['understanding'].append(record.understanding_rate)
            literacy_summary[group]['engagement'].append(record.engagement_rate)

        result = [{
            'group': group,
            'understanding': round(mean(data['understanding']), 2) if data['understanding'] else 0,
            'engagement': round(mean(data['engagement']), 2) if data['engagement'] else 0
        } for group, data in literacy_summary.items()]

        logger.debug(f"Fetched {len(result)} health literacy records")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error fetching health literacy: {str(e)}")
        return jsonify({'message': f'Error fetching health literacy: {str(e)}'}), 500

@admin_bp.route('/analytics/workflow_metrics', methods=['GET'])
@jwt_required()
def workflow_metrics():
    """Get workflow metrics (admin only)."""
    logger.debug("Received workflow metrics request")
    auth_header = request.headers.get('Authorization')
    user = verify_user_from_token(auth_header)
    if not user or not hasattr(user, 'is_admin') or not user['is_admin']:
        logger.error("Admin access required")
        return jsonify({'message': 'Admin access required'}), 403

    try:
        time_range = request.args.get('time_range', '7d')
        now = datetime.datetime.now(timezone.utc)
        if time_range == '24h':
            start_time = now - timedelta(hours=24)
        elif time_range == '7d':
            start_time = now - timedelta(days=7)
        elif time_range == '30d':
            start_time = now - timedelta(days=30)
        else:  # 90d
            start_time = now - timedelta(days=90)

        metrics = WorkflowMetric.query.filter(WorkflowMetric.recorded_at >= start_time).all()
        result = [{
            'metric': m.metric_name,
            'value': m.value,
            'change': f"{m.change_percentage:+.2f}%" if m.change_percentage is not None else 'N/A'
        } for m in metrics]

        logger.debug(f"Fetched {len(result)} workflow metrics")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error fetching workflow metrics: {str(e)}")
        return jsonify({'message': f'Error fetching workflow metrics: {str(e)}'}), 500

@admin_bp.route('/analytics/ai_performance', methods=['GET'])
@jwt_required()
def ai_performance():
    """Get AI performance metrics (admin only)."""
    logger.debug("Received AI performance request")
    auth_header = request.headers.get('Authorization')
    user = verify_user_from_token(auth_header)
    if not user or not hasattr(user, 'is_admin') or not user['is_admin']:
        logger.error("Admin access required")
        return jsonify({'message': 'Admin access required'}), 403

    try:
        time_range = request.args.get('time_range', '7d')
        now = datetime.datetime.now(timezone.utc)
        if time_range == '24h':
            start_time = now - timedelta(hours=24)
        elif time_range == '7d':
            start_time = now - timedelta(days=7)
        elif time_range == '30d':
            start_time = now - timedelta(days=30)
        else:  # 90d
            start_time = now - timedelta(days=90)

        metrics = AIPerformance.query.filter(AIPerformance.recorded_at >= start_time).all()
        result = [{
            'metric': m.metric_name,
            'value': round(m.value, 2)
        } for m in metrics]

        # Ensure some default metrics if none exist
        if not result:
            result = [
                {'metric': 'Response Quality', 'value': 0},
                {'metric': 'Response Accuracy', 'value': 0},
                {'metric': 'Safety Compliance', 'value': 0},
                {'metric': 'Response Speed', 'value': 0}
            ]

        logger.debug(f"Fetched {len(result)} AI performance metrics")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error fetching AI performance: {str(e)}")
        return jsonify({'message': f'Error fetching AI performance: {str(e)}'}), 500
