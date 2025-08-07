import os
import logging
import pandas as pd
import requests
import json
import re
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict
import difflib
import uuid
import numpy as np  # Added for MF model computations

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = os.getenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions")

# Enhanced conversation memory structure
class ConversationMemory:
    def __init__(self):
        self.messages = []
        self.user_preferences = {"language": "en"}  # Default to English
        self.mentioned_symptoms = set()
        self.mentioned_conditions = set()
        self.emotional_state_history = []
        self.topics_discussed = set()
        self.user_concerns = []
        self.follow_up_needed = []
        self.conversation_depth = 0  # Track how many user messages

    def add_message(self, text, is_user=True, sentiment=None, topics=None):
        message = {
            'id': str(uuid.uuid4()),
            'text': text,
            'is_user': is_user,
            'timestamp': datetime.now().isoformat(),
            'sentiment': sentiment,
            'topics': topics or []
        }
        self.messages.append(message)
        if is_user:
            self.conversation_depth += 1
            if sentiment:
                self.emotional_state_history.append({
                    'sentiment': sentiment,
                    'timestamp': datetime.now().isoformat()
                })
        if topics:
            self.topics_discussed.update(topics)

    def load_history(self, messages):
        """Load conversation history from database"""
        self.messages = []
        self.mentioned_symptoms = set()
        self.mentioned_conditions = set()
        self.emotional_state_history = []
        self.topics_discussed = set()
        self.conversation_depth = 0
        for msg in messages:
            topics, symptoms, conditions = extract_entities(msg['text'])
            sentiment = detect_emotional_state(msg['text']) if msg.get('isUser') else 'EMPATHETIC'
            self.add_message(
                text=msg['text'],
                is_user=msg.get('isUser', False),
                sentiment=sentiment,
                topics=topics
            )
            self.mentioned_symptoms.update(symptoms)
            self.mentioned_conditions.update(conditions)

    def get_context_summary(self, max_messages=10):
        """Summarize recent conversation for context"""
        recent_messages = self.messages[-max_messages:]
        return {
            'recent_messages': recent_messages,
            'symptoms_mentioned': list(self.mentioned_symptoms),
            'conditions_mentioned': list(self.mentioned_conditions),
            'emotional_progression': self.emotional_state_history[-5:],
            'topics_covered': list(self.topics_discussed),
            'concerns': self.user_concerns,
            'follow_ups': self.follow_up_needed,
            'conversation_depth': self.conversation_depth
        }

# Global conversation memories
conversation_memories = defaultdict(ConversationMemory)

# Patterns for entity extraction (supporting French and English)
SYMPTOM_PATTERNS = [
    r'\b(pain|hurt|ache|sore|burning|throbbing|sharp|dull|cramping|douleur|mal|brûlure|aigu|chronique)\b',
    r'\b(fever|temperature|hot|chills|sweating|fièvre|température|chaud|froid|sueur)\b',
    r'\b(nausea|vomit|sick|dizzy|headache|migraine|nausée|vomi|malade|vertige|mal de tête)\b',
    r'\b(cough|sneeze|runny nose|congestion|sore throat|toux|éternuement|nez qui coule|gorge irritée)\b',
    r'\b(tired|fatigue|exhausted|weak|energy|fatigué|épuisé|faible|énergie)\b',
    r'\b(sleep|insomnia|restless|nightmare|sommeil|insomnie|agité|cauchemar)\b',
    r'\b(appetite|eating|weight|stomach|appétit|manger|poids|estomac)\b',
    r'\b(breathing|shortness|chest|heart|respiration|essoufflement|poitrine|cœur)\b',
    r'\b(skin|rash|itchy|red|swollen|peau|éruption|démangeaison|rouge|gonflé)\b',
    r'\b(joint|muscle|back|neck|shoulder|articulation|muscle|dos|cou|épaule)\b'
]

CONDITION_PATTERNS = [
    r'\b(diabetes|hypertension|blood pressure|sugar|diabète|hypertension|pression artérielle|sucre)\b',
    r'\b(malaria|typhoid|cholera|yellow fever|paludisme|typhoïde|choléra|fièvre jaune)\b',
    r'\b(asthma|bronchitis|pneumonia|tuberculosis|asthme|bronchite|pneumonie|tuberculose)\b',
    r'\b(cancer|tumor|growth|lump|cancer|tumeur|croissance|masse)\b',
    r'\b(depression|anxiety|stress|mental health|dépression|anxiété|stress|santé mentale)\b',
    r'\b(pregnancy|pregnant|expecting|baby|c-section|cesarean|grossesse|enceinte|bébé|césarienne)\b',
    r'\b(allergy|allergic|reaction|allergie|réaction)\b'
]

EMOTIONAL_INDICATORS = {
    'VERY_POSITIVE': r'\b(amazing|fantastic|wonderful|excellent|perfect|thrilled|ecstatic|merveilleux|fantastique|excellent|parfait|enthousiaste)\b',
    'POSITIVE': r'\b(good|better|great|happy|pleased|glad|improving|recovering|bon|meilleur|heureux|satisfait|amélioration)\b',
    'NEUTRAL': r'\b(okay|fine|normal|regular|usual|moderate|d\'accord|normal|habituel|modéré)\b',
    'CONCERNED': r'\b(worried|concerned|anxious|nervous|unsure|confused|inquiet|anxieux|nerveux|incertain|confus)\b',
    'NEGATIVE': r'\b(bad|worse|terrible|awful|sick|ill|pain|hurt|mauvais|pire|terrible|malade|douleur)\b',
    'VERY_NEGATIVE': r'\b(horrible|unbearable|excruciating|devastating|hopeless|desperate|horrible|insupportable|dévastateur|désespéré)\b'
}

# French-specific words for language detection
FRENCH_INDICATORS = [
    'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'le', 'la', 'l\'', 'les',
    'un', 'une', 'des', 'et', 'à', 'de', 'est', 'suis', 'es', 'ai', 'as', 'a', 'ce', 'cette',
    'ça', 'pour', 'avec', 'sur', 'dans', 'par', 'mais', 'ou', 'si', 'quand', 'je', 'ne', 'pas',
    'plus', 'moins', 'très', 'bien', 'merci', 'bonjour', 'au', 'aux', 'd\'', 'du', 'douleur',
    'mal', 'fièvre', 'toux', 'fatigué', 'nausée', 'vertige', 'malade'
]

def detect_language(text):
    """Detect if the input text is French or English"""
    if not text or not isinstance(text, str):
        return "en"  # Default to English if text is empty or invalid

    text_lower = text.lower().strip()
    words = re.findall(r'\b\w+\b', text_lower)
    
    if not words:
        return "en"  # Default to English for empty word list

    french_count = sum(1 for word in words if word in FRENCH_INDICATORS)
    french_ratio = french_count / len(words)

    logger.debug(f"Language detection: French words: {french_count}/{len(words)} (Ratio: {french_ratio:.2f})")
    
    return "fr" if french_ratio >= 0.3 else "en"

# --- Modified: Load clinical dataset and train MF model ---
DATASET_PATH = os.path.join(os.path.dirname(__file__), "clinical_summaries.csv")
dataset_df = pd.DataFrame()
symptom_embeddings = None
condition_embeddings = None
record_embeddings = None
symptom_to_idx = {}
condition_to_idx = {}

def train_mf_model(dataset_df, n_components=10):
    """Train a Matrix Factorization model using SVD on symptom-condition interactions"""
    global symptom_embeddings, condition_embeddings, record_embeddings, symptom_to_idx, condition_to_idx
    
    if dataset_df.empty:
        logger.warning("Cannot train MF model: Dataset is empty")
        return
    
    # Extract all unique symptoms and conditions from dataset
    all_symptoms = set()
    all_conditions = set()
    for _, row in dataset_df.iterrows():
        summary = str(row.get('summary_text', '')).lower()
        diagnosis = str(row.get('diagnosis', '')).lower()
        text = f"{summary} {diagnosis}"
        _, symptoms, conditions = extract_entities(text)
        all_symptoms.update(symptoms)
        all_conditions.update(conditions)
    
    # Create mappings
    symptom_to_idx = {s: i for i, s in enumerate(sorted(all_symptoms))}
    condition_to_idx = {c: i for i, c in enumerate(sorted(all_conditions))}
    
    # Build symptom-condition interaction matrix
    n_records = len(dataset_df)
    n_symptoms = len(symptom_to_idx)
    n_conditions = len(condition_to_idx)
    
    # Create record-symptom and record-condition matrices
    record_symptom_matrix = np.zeros((n_records, n_symptoms))
    record_condition_matrix = np.zeros((n_records, n_conditions))
    
    for idx, row in dataset_df.iterrows():
        summary = str(row.get('summary_text', '')).lower()
        diagnosis = str(row.get('diagnosis', '')).lower()
        text = f"{summary} {diagnosis}"
        _, symptoms, conditions = extract_entities(text)
        
        for symptom in symptoms:
            if symptom in symptom_to_idx:
                record_symptom_matrix[idx, symptom_to_idx[symptom]] = 1
        for condition in conditions:
            if condition in condition_to_idx:
                record_condition_matrix[idx, condition_to_idx[condition]] = 1
    
    # Combine matrices for MF
    interaction_matrix = np.hstack([record_symptom_matrix, record_condition_matrix])
    
    # Apply SVD
    try:
        U, sigma, Vt = np.linalg.svd(interaction_matrix, full_matrices=False)
        # Keep top n_components
        U = U[:, :n_components]
        sigma = np.diag(sigma[:n_components])
        Vt = Vt[:n_components, :]
        
        # Split embeddings
        record_embeddings = U @ sigma  # Record embeddings
        feature_embeddings = Vt.T  # Symptom + condition embeddings
        symptom_embeddings = feature_embeddings[:n_symptoms, :]
        condition_embeddings = feature_embeddings[n_symptoms:, :]
        
        logger.info(f"Trained MF model with {n_components} components. "
                   f"Record embeddings shape: {record_embeddings.shape}, "
                   f"Symptom embeddings shape: {symptom_embeddings.shape}, "
                   f"Condition embeddings shape: {condition_embeddings.shape}")
    except Exception as e:
        logger.error(f"Failed to train MF model: {str(e)}")
        symptom_embeddings = None
        condition_embeddings = None
        record_embeddings = None

try:
    if not os.path.exists(DATASET_PATH):
        logger.warning(f"Dataset file not found at {DATASET_PATH}. Proceeding without clinical data.")
    else:
        dataset_df = pd.read_csv(DATASET_PATH, encoding='utf-8', on_bad_lines='warn')
        if dataset_df.empty:
            logger.warning("Dataset is empty after loading")
        else:
            # Verify required columns
            required_columns = [
                'summary_id', 'patient_id', 'patient_age', 'patient_gender',
                'diagnosis', 'body_temp_c', 'blood_pressure_systolic',
                'heart_rate', 'summary_text', 'date_recorded'
            ]
            missing_columns = [col for col in required_columns if col not in dataset_df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in dataset: {missing_columns}")
                dataset_df = pd.DataFrame()  # Reset to empty DataFrame
            else:
                dataset_df = dataset_df.dropna(subset=required_columns)
                if dataset_df.empty:
                    logger.warning("Dataset is empty after dropping NA values")
                else:
                    logger.info(f"Loaded dataset with {len(dataset_df)} records")
                    train_mf_model(dataset_df)  # Train MF model
except Exception as e:
    logger.error(f"Failed to load dataset at {DATASET_PATH}: {str(e)}")
    dataset_df = pd.DataFrame()

def extract_entities(text):
    """Extract symptoms, conditions, and topics from text"""
    topics, symptoms, conditions = set(), set(), set()
    text_lower = text.lower()

    for pattern in SYMPTOM_PATTERNS:
        matches = re.findall(pattern, text_lower)
        symptoms.update(matches)
        if matches:
            topics.add('symptoms')

    for pattern in CONDITION_PATTERNS:
        matches = re.findall(pattern, text_lower)
        conditions.update(matches)
        if matches:
            topics.add('medical_conditions')

    if any(word in text_lower for word in ['diet', 'food', 'eat', 'nutrition', 'alimentation', 'nourriture', 'manger']):
        topics.add('nutrition')
    if any(word in text_lower for word in ['exercise', 'sport', 'physical', 'activity', 'exercice', 'sport', 'physique', 'activité']):
        topics.add('physical_activity')
    if any(word in text_lower for word in ['mental', 'stress', 'anxiety', 'depression', 'mental', 'stress', 'anxiété', 'dépression']):
        topics.add('mental_health')
    if any(word in text_lower for word in ['medicine', 'medication', 'drug', 'treatment', 'médicament', 'traitement']):
        topics.add('medication')

    return topics, symptoms, conditions

def detect_emotional_state(text):
    """Detect user's emotional state"""
    text_lower = text.lower()
    for state, pattern in EMOTIONAL_INDICATORS.items():
        if re.search(pattern, text_lower):
            return state
    return 'NEUTRAL'

def should_use_clinical_data(symptoms, conditions, user_input, conversation_depth):
    """Determine if clinical data should be included in response"""
    medical_keywords = [
        'what is', 'tell me about', 'explain', 'symptoms of', 'treatment for',
        'causes of', 'how to treat', 'diagnosis', 'condition', 'disease',
        'qu\'est-ce que', 'expliquez', 'symptômes de', 'traitement pour',
        'causes de', 'comment traiter', 'diagnostic', 'maladie'
    ]
    
    user_input_lower = user_input.lower()
    
    if any(phrase in user_input_lower for phrase in [
        'thank', 'thanks', 'merci', 'hello', 'hi', 'bonjour', 'how are you',
        'comment allez-vous', 'goodbye', 'au revoir', 'feeling better',
        'je me sens mieux'
    ]):
        return False
    
    has_medical_keywords = any(keyword in user_input_lower for keyword in medical_keywords)
    has_specific_symptoms = len(symptoms) > 0 and any(len(s) > 3 for s in symptoms)
    has_conditions = len(conditions) > 0
    
    return has_medical_keywords or has_specific_symptoms or has_conditions

# --- Modified: Query dataset using MF embeddings ---
def query_dataset(symptoms, conditions, max_records=3):
    """Query dataset using MF embeddings for relevant clinical records"""
    if dataset_df.empty or symptom_embeddings is None or condition_embeddings is None:
        logger.warning("Cannot query dataset: DataFrame is empty or MF model not trained")
        return []

    # Create query vector from symptoms and conditions
    n_symptoms = len(symptom_to_idx)
    n_conditions = len(condition_to_idx)
    query_vector = np.zeros(n_symptoms + n_conditions)
    
    for symptom in symptoms:
        if symptom in symptom_to_idx:
            query_vector[symptom_to_idx[symptom]] = 1
    for condition in conditions:
        if condition in condition_to_idx:
            query_vector[condition_to_idx[condition] + n_symptoms] = 1
    
    # Compute query embedding
    try:
        feature_embeddings = np.vstack([symptom_embeddings, condition_embeddings])
        query_embedding = query_vector @ feature_embeddings
    except Exception as e:
        logger.error(f"Error computing query embedding: {str(e)}")
        return []
    
    # Compute cosine similarity with record embeddings
    similarities = []
    for idx in range(len(dataset_df)):
        record_embedding = record_embeddings[idx]
        norm_query = np.linalg.norm(query_embedding)
        norm_record = np.linalg.norm(record_embedding)
        if norm_query == 0 or norm_record == 0:
            similarity = 0
        else:
            similarity = np.dot(query_embedding, record_embedding) / (norm_query * norm_record)
        similarities.append((idx, similarity))
    
    # Sort by similarity and select top records
    similarities.sort(key=lambda x: x[1], reverse=True)
    relevant_indices = [idx for idx, sim in similarities[:max_records] if sim > 0]
    
    records = []
    for idx in relevant_indices:
        row = dataset_df.iloc[idx]
        records.append({
            'age': row.get('patient_age', 'N/A'),
            'gender': row.get('patient_gender', 'N/A'),
            'diagnosis': row.get('diagnosis', 'N/A'),
            'vitals': {
                'temp': row.get('body_temp_c', 'N/A'),
                'bp': row.get('blood_pressure_systolic', 'N/A'),
                'hr': row.get('heart_rate', 'N/A')
            },
            'summary': str(row.get('summary_text', ''))[:200] + '...' if row.get('summary_text') else 'N/A'
        })
    
    logger.debug(f"Queried dataset using MF, found {len(records)} relevant records")
    return records

def call_groq_api(messages, model="llama3-70b-8192", max_tokens=500, temperature=0.7):
    """Call Groq API with improved error handling and optimized parameters"""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set in environment variables")
        return "Error: GROQ_API_KEY is not configured"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "HealthApp/2.1"
    }
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        logger.debug(f"Sending request to {GROQ_ENDPOINT} with model {model}")
        logger.debug(f"Request data: {json.dumps(data, indent=2)}")
        
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=data, timeout=30, verify=True)
        logger.debug(f"Received response status: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            logger.error(f"API returned status {response.status_code}: {response.text}")
            response.raise_for_status()
            
        result = response.json()
        logger.debug(f"Response JSON: {json.dumps(result, indent=2)}")
        
        if 'choices' in result and result['choices']:
            content = result['choices'][0]['message']['content'].strip()
            logger.info("Successfully received response from Groq API")
            return content
        else:
            logger.warning(f"No valid choices in API response: {result}")
            return "Error: No valid response from AI service"
            
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        if hasattr(http_err, 'response') and http_err.response is not None:
            logger.error(f"Response text: {http_err.response.text}")
        return f"Error: HTTP error from AI service: {http_err}"
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error occurred: {req_err}")
        return "Error: Unable to connect to AI service"
    except json.JSONDecodeError as json_err:
        logger.error(f"JSON decode error: {json_err}")
        return "Error: Invalid response format from AI service"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"Error: Unexpected issue with AI service: {str(e)}"

def format_response_for_readability(response_text):
    """Format response with proper paragraphs and remove asterisks"""
    if not response_text:
        return response_text
    
    response_text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', response_text)
    sentences = re.split(r'(?<=[.!?])\s+', response_text.strip())
    
    if len(sentences) <= 2:
        return response_text
    
    paragraphs = []
    current_paragraph = []
    
    for i, sentence in enumerate(sentences):
        current_paragraph.append(sentence)
        if (len(current_paragraph) >= 2 and 
            (i == len(sentences) - 1 or 
             any(keyword in sentence.lower() for keyword in [
                 'however', 'also', 'additionally', 'meanwhile', 'furthermore',
                 'on the other hand', 'in addition', 'moreover', 'cependant',
                 'aussi', 'de plus', 'par ailleurs'
             ]) or
             len(current_paragraph) >= 3)):
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
    
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    return '\n\n'.join(paragraphs)

def get_appropriate_emoji(emotional_state, conversation_depth):
    """Get appropriate emoji based on context (limited to face emojis only)"""
    emoji_map = {
        'VERY_POSITIVE': '😊',
        'POSITIVE': '😊', 
        'CONCERNED': '😔',
        'NEGATIVE': '😔',
        'VERY_NEGATIVE': '😔',
        'NEUTRAL': ''
    }
    
    if conversation_depth > 3 and conversation_depth % 3 != 0:
        return ''
    
    return emoji_map.get(emotional_state, '')

def build_system_prompt(patient_info, context, emotional_state, language):
    """Build a dynamic system prompt for AI with cultural and contextual optimizations"""
    name = patient_info.get('name', 'Patient')
    username = patient_info.get('username', '')
    age = patient_info.get('age', 'N/A')
    region = patient_info.get('region', 'Cameroon')
    conversation_depth = context.get('conversation_depth', 0)

    recent_messages = context['recent_messages']
    symptoms = context['symptoms_mentioned']
    conditions = context['conditions_mentioned']
    topics = context['topics_covered']
    emotional_trend = [e['sentiment'] for e in context['emotional_progression']]

    if username:
        name_instruction = f"Use the username '{username}' occasionally, not in every message."
    elif conversation_depth > 2:
        first_name = name.split()[0] if name != 'Patient' else ''
        name_instruction = (
            f"Use the name '{first_name}' sparingly, and avoid repetitive usage." if first_name 
            else "Avoid overusing names."
        )
    else:
        name_instruction = "You may use the patient's name occasionally, but don't overuse it."

    if language == "fr":
        prompt = f"""Vous êtes Dr. Healia, un assistant médical intelligent et empathique au service des patients camerounais.

**Profil du patient** :
- Nom : {name}
- Âge : {age} ans
- Région : {region}
- Langue : Français (détectée à partir de l'entrée utilisateur)

**Contexte de la conversation** :
- Messages récents : {len(recent_messages)} échanges
- Symptômes mentionnés : {', '.join(symptoms) if symptoms else 'Aucun'}
- Affections mentionnées : {', '.join(conditions) if conditions else 'Aucune'}
- Sujets abordés : {', '.join(topics) if topics else 'Aucun'}
- État émotionnel actuel : {emotional_state}
- Évolution émotionnelle : {', '.join(emotional_trend[-3:]) if emotional_trend else 'Aucune'}

**Instructions importantes** :
1. Répondez en français si l'utilisateur écrit en français, sinon en anglais.
2. {name_instruction}
3. Structurez vos réponses en paragraphes courts et faciles à lire.
4. Adoptez un ton naturel, empathique et culturellement adapté aux patients camerounais.
5. Utilisez les emojis avec modération – privilégiez les visages expressifs (😊, 😔, 🤗) dans les réponses longues, mais évitez d'en mettre dans chaque message.
6. Tenez compte de l'historique de la conversation sans répéter les informations déjà partagées.
7. Répondez UNIQUEMENT à ce qui est demandé – n'ajoutez pas de détails médicaux ou de diagnostics non mentionnés.
8. Posez des questions de suivi pertinentes en fonction des symptômes, des affections et de l'état émotionnel, sans donner l'impression d'interroger.
9. Si la demande est vague, demandez des précisions pour mieux orienter votre réponse.
10. Donnez des conseils simples, concrets et adaptés à la réalité camerounaise.
11. Utilisez des paragraphes pour éviter les gros blocs de texte fatigants pour les yeux.
12. Employez un langage simple, même pour les sujets médicaux complexes, pour que chacun puisse comprendre.
13. Parlez de manière naturelle et familière, comme un médecin local à qui l'on peut faire confiance.
"""
    else:
        prompt = f"""You are Dr. Healia, an intelligent and empathetic medical assistant supporting patients in Cameroon.

**Patient Profile**:
- Name: {name}
- Age: {age} years
- Region: {region}
- Language: English (detected from user input)

**Conversation Context**:
- Recent messages: {len(recent_messages)} exchanges
- Symptoms mentioned: {', '.join(symptoms) if symptoms else 'None'}
- Conditions mentioned: {', '.join(conditions) if conditions else 'None'}
- Topics discussed: {', '.join(topics) if topics else 'None'}
- Current emotional state: {emotional_state}
- Emotional progression: {', '.join(emotional_trend[-3:]) if emotional_trend else 'None'}

**Important Instructions**:
1. Respond in English if the user writes in English, or in French if the user writes in French.
2. {name_instruction}
3. Structure responses into short, clear paragraphs for easier reading.
4. Use a warm, natural tone that aligns with Cameroonian cultural context.
5. Use emojis sparingly – mainly expressive faces (😊, 😔, 🤗) in longer responses, but not in every message.
6. Reference the conversation history without repeating previously covered information.
7. ONLY address what the user asks – do not introduce unrelated medical information.
8. Ask thoughtful follow-up questions based on symptoms, conditions, and emotions, without making it feel like an interrogation.
9. If the request is vague, ask clarifying questions to improve your response.
10. Provide practical, accessible advice tailored to the Cameroonian setting.
11. Break long text into paragraphs so it's easier on the eyes; use emojis to keep it engaging when appropriate.
12. Avoid complex medical terms – explain things simply so even non-medical users can understand.
13. Communicate like a trusted local doctor; use familiar language that connects with Cameroonian users.
"""
    
    return prompt

def generate_personalized_response(user_input, patient_info, session_id="default", history=None):
    """Generate AI-driven, context-aware response in the detected language using patient info strategically"""
    if not user_input or not patient_info:
        default_lang = patient_info.get('language', 'en') if patient_info else 'en'
        error_msg = "J'ai besoin de plus d'informations pour vous aider correctement. Pouvez-vous partager plus de détails ? 😔" if default_lang == "fr" else "I need more information to assist you properly. Could you share more details? 😔"
        logger.warning(f"Invalid input or patient info, returning: {error_msg}")
        return error_msg

    memory = conversation_memories[session_id]
    
    detected_lang = detect_language(user_input)
    memory.user_preferences["language"] = detected_lang
    logger.debug(f"Detected language: {detected_lang}, user input: {user_input}")

    if history:
        memory.load_history(history)

    topics, symptoms, conditions = extract_entities(user_input)
    emotional_state = detect_emotional_state(user_input)
    memory.add_message(user_input, is_user=True, sentiment=emotional_state, topics=topics)
    memory.mentioned_symptoms.update(symptoms)
    memory.mentioned_conditions.update(conditions)

    context = memory.get_context_summary()

    def should_use_patient_info():
        medical_query = any(keyword in user_input.lower() for keyword in [
            'treat', 'treatment', 'help', 'manage', 'deal with', 'medicine', 'medication',
            'traiter', 'traitement', 'gérer', 'médicament', 'soigner', 'guérir'
        ])
        has_relevant_info = any([
            patient_info.get('chronic_conditions') != 'None',
            patient_info.get('allergies') != 'None',
            patient_info.get('medications'),
            patient_info.get('lifestyle', {}).get('smokes'),
            patient_info.get('lifestyle', {}).get('alcohol') != 'Never',
            patient_info.get('family_history')
        ])
        return (len(symptoms) > 0 or len(conditions) > 0 or medical_query or
                emotional_state in ['CONCERNED', 'NEGATIVE', 'VERY_NEGATIVE'] and has_relevant_info)

    dataset_records = []
    if should_use_clinical_data(symptoms, conditions, user_input, memory.conversation_depth):
        dataset_records = query_dataset(memory.mentioned_symptoms, memory.mentioned_conditions)

    system_prompt = build_system_prompt(patient_info, context, emotional_state, memory.user_preferences["language"])
    conversation_history = "\n".join([
        f"{'Utilisateur' if msg['is_user'] else 'Assistant' if memory.user_preferences['language'] == 'fr' else 'User' if msg['is_user'] else 'Assistant'}: {msg['text']}" 
        for msg in context['recent_messages'][-5:]
    ])
    
    patient_context = ""
    if should_use_patient_info():
        relevant_info = []
        if patient_info.get('chronic_conditions') != 'None' and any(c.lower() in user_input.lower() for c in patient_info.get('chronic_conditions', '').split(',')):
            relevant_info.append(f"Chronic conditions: {patient_info['chronic_conditions']}")
        if patient_info.get('allergies') != 'None' and ('allergy' in user_input.lower() or 'allergic' in user_input.lower()):
            relevant_info.append(f"Allergies: {patient_info['allergies']}")
        if patient_info.get('medications') and ('medication' in user_input.lower() or 'medicine' in user_input.lower()):
            relevant_info.append(f"Medications: {patient_info['medications']}")
        if patient_info.get('lifestyle', {}).get('smokes') and 'smoking' in user_input.lower():
            relevant_info.append(f"Smoking status: {'Smokes' if patient_info['lifestyle']['smokes'] else 'Non-smoker'}")
        if patient_info.get('family_history') and ('family' in user_input.lower() or 'history' in user_input.lower()):
            relevant_info.append(f"Family medical history: {patient_info['family_history']}")
        if patient_info.get('age') != 'N/A' and ('age' in user_input.lower() or any(c in conditions for c in ['diabetes', 'hypertension', 'cancer'])):
            relevant_info.append(f"Age: {patient_info['age']}")
        if patient_info.get('region') != 'N/A' and any(t in topics for t in ['malaria', 'yellow fever']):
            relevant_info.append(f"Region: {patient_info['region']}")

        if relevant_info:
            patient_context = f"{'Informations pertinentes sur le patient' if memory.user_preferences['language'] == 'fr' else 'Relevant Patient Information'}: " + "; ".join(relevant_info)

    additional_context = []
    if dataset_context := "\n".join([
        f"{'Cas' if memory.user_preferences['language'] == 'fr' else 'Case'} {i}: {r['diagnosis']} - {r['summary']}" 
        for i, r in enumerate(dataset_records, 1)
    ]):
        additional_context.append(f"{'Données cliniques pertinentes' if memory.user_preferences['language'] == 'fr' else 'Relevant Clinical Data'}: {dataset_context}")
    if patient_context:
        additional_context.append(patient_context)

    full_prompt = f"{conversation_history}\n\n{'Utilisateur' if memory.user_preferences['language'] == 'fr' else 'User'}: {user_input}"
    if additional_context:
        full_prompt += f"\n\n{'\n'.join(additional_context)}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt}
    ]
    
    logger.debug(f"Full API request - Messages: {json.dumps(messages, indent=2)}")
    
    response = call_groq_api(messages, max_tokens=500, temperature=0.7)

    if response and not response.startswith("Error:"):
        formatted_response = format_response_for_readability(response)
        sentences = re.split(r'(?<=[.!?])\s+', formatted_response.strip())
        paragraphs = []
        current_paragraph = []
        for i, sentence in enumerate(sentences):
            current_paragraph.append(sentence)
            if len(current_paragraph) >= 2 or i == len(sentences) - 1 or any(keyword in sentence.lower() for keyword in [
                'however', 'also', 'additionally', 'meanwhile', 'furthermore', 'on the other hand', 'in addition',
                'moreover', 'cependant', 'aussi', 'de plus', 'par ailleurs'
            ]):
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        formatted_response = '\n\n'.join(paragraphs)

        emoji = get_appropriate_emoji(emotional_state, memory.conversation_depth)
        if emoji and not any(e in formatted_response for e in ['😊', '😔', '🤗']) and (
            emotional_state in ['POSITIVE', 'VERY_POSITIVE', 'CONCERNED', 'NEGATIVE', 'VERY_NEGATIVE'] or
            any(t in topics for t in ['symptoms', 'medical_conditions', 'medication'])
        ):
            formatted_response += f" {emoji}"

        if patient_context and any(t in topics for t in ['symptoms', 'medical_conditions', 'medication']):
            follow_up = (
                "Pouvez-vous me dire si vos symptômes ont changé récemment ou si vous prenez des médicaments spécifiques pour cela ?"
                if memory.user_preferences["language"] == "fr"
                else "Could you let me know if your symptoms have changed recently or if you're taking any specific medications for this?"
            )
            formatted_response += f"\n\n{follow_up}"

        memory.add_message(formatted_response, is_user=False, sentiment="EMPATHETIC")
        logger.info(f"Generated AI response: {formatted_response[:100]}...")
        return formatted_response

    logger.error(f"API call failed. Response was: {response}")
    
    if symptoms:
        symptoms_list = list(symptoms)[:3]
        if memory.user_preferences["language"] == "fr":
            fallback = f"Je comprends que vous ressentez {', '.join(symptoms_list)}. "
            if emotional_state in ['CONCERNED', 'NEGATIVE', 'VERY_NEGATIVE']:
                fallback += "Je sais que cela peut être inquiétant. "
            if 'pain' in symptoms or 'douleur' in symptoms:
                fallback += "Pour la douleur, vous pouvez essayer de vous reposer et appliquer de la chaleur ou du froid selon ce qui vous soulage. "
            if 'stress' in user_input.lower():
                fallback += "Le stress peut effectivement aggraver certains symptômes physiques. "
            fallback += "Il serait sage de consulter un professionnel de santé si les symptômes persistent ou s'aggravent. Pouvez-vous me dire depuis quand vous ressentez cela ? 😊"
        else:
            fallback = f"I understand you're experiencing {', '.join(symptoms_list)}. "
            if emotional_state in ['CONCERNED', 'NEGATIVE', 'VERY_NEGATIVE']:
                fallback += "I know this can be concerning. "
            if 'pain' in symptoms:
                fallback += "For pain management, you might try rest and applying heat or cold depending on what feels better. "
            if 'stress' in user_input.lower():
                fallback += "Stress can indeed worsen physical symptoms. "
            fallback += "It would be wise to see a healthcare professional if symptoms persist or worsen. Can you tell me how long you've been experiencing this? 😊"
    else:
        if memory.user_preferences["language"] == "fr":
            fallback = "Je rencontre actuellement un problème technique, mais je suis là pour vous aider. Pouvez-vous me parler un peu plus de ce qui vous préoccupe ? 😊"
        else:
            fallback = "I'm experiencing a technical issue right now, but I'm here to help. Could you tell me a bit more about what's concerning you? 😊"

    if should_use_patient_info() and patient_info.get('chronic_conditions') != 'None':
        if memory.user_preferences["language"] == "fr":
            fallback += f" Je note que vous avez des antécédents de {patient_info['chronic_conditions']}."
        else:
            fallback += f" I note that you have a history of {patient_info['chronic_conditions']}."

    memory.add_message(fallback, is_user=False, sentiment="EMPATHETIC")
    logger.warning(f"Enhanced fallback response generated: {fallback}")
    return fallback

def test_conversation_flow():
    """Test the AI-driven conversation system with optimizations"""
    print("🧪 Testing Optimized AI Medical Assistant with MF Model\n")
    test_patients = [
        {
            'name': 'Marie Ngozi',
            'username': 'marie_n',
            'age': 28,
            'language': 'en',
            'region': 'Douala',
            'chronic_conditions': 'None'
        },
        {
            'name': 'Jean Dupont',
            'username': 'jean_d',
            'age': 35,
            'language': 'fr',
            'region': 'Yaoundé',
            'chronic_conditions': 'Hypertension'
        }
    ]
    test_conversations = [
        ("I'm having headaches lately", "J'ai des maux de tête récemment"),
        ("It's worse in the evenings and I feel nauseous", "C'est pire le soir et je me sens nauséeux"),
        ("I'm scheduled for a C-section next week and I'm worried", "Je dois avoir une césarienne la semaine prochaine et je suis inquiet"),
        ("Thank you for your help!", "Merci pour votre aide !"),
        ("What is malaria?", "Qu'est-ce que le paludisme ?"),
        ("Hello, how are you?", "Bonjour, comment allez-vous ?")
    ]
    session_id = "test_session"

    for patient in test_patients:
        lang = patient['language']
        print(f"\n📱 Conversation Simulation for {patient['username']} (Login preference: {'English' if lang == 'en' else 'French'}):")
        print("=" * 70)
        conversation_memories[session_id] = ConversationMemory()
        
        test_inputs = [
            test_conversations[0][1 if patient['language'] == 'en' else 0],
            test_conversations[1][0 if patient['language'] == 'fr' else 1],
            test_conversations[4][1 if patient['language'] == 'en' else 0],
            test_conversations[5][0 if patient['language'] == 'fr' else 1],
            test_conversations[2][1 if patient['language'] == 'en' else 0],
            test_conversations[3][0 if patient['language'] == 'fr' else 1]
        ]
        
        for i, message in enumerate(test_inputs, 1):
            print(f"\n👤 {'Utilisateur' if detect_language(message) == 'fr' else 'User'} (Message {i}): {message}")
            start_time = datetime.now()
            response = generate_personalized_response(message, patient, session_id, [])
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            print(f"🤖 Dr. Healia: {response}")
            print(f"⏱️  Response time: {response_time:.2f} seconds")
            print("-" * 50)

        memory = conversation_memories[session_id]
        print(f"\n🧠 Conversation Memory Summary:")
        print(f"Total messages: {len(memory.messages)}")
        print(f"Conversation depth: {memory.conversation_depth}")
        print(f"Symptoms: {memory.mentioned_symptoms}")
        print(f"Conditions: {memory.mentioned_conditions}")
        print(f"Emotional progression: {[e['sentiment'] for e in memory.emotional_state_history]}")
        print(f"Topics: {memory.topics_discussed}")
        print("=" * 70)

def debug_api_connection():
    """Test the API connection with a simple request"""
    print("🔍 Testing API Connection...")
    
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in environment variables")
        return False
    
    print(f"✅ API Key found: {GROQ_API_KEY[:10]}...")
    print(f"📡 Endpoint: {GROQ_ENDPOINT}")
    
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello"}
    ]
    
    response = call_groq_api(test_messages, max_tokens=50)
    
    if response.startswith("Error:"):
        print(f"❌ API call failed: {response}")
        return False
    else:
        print(f"✅ API call successful: {response}")
        return True

if __name__ == "__main__":
    if debug_api_connection():
        test_conversation_flow()
    else:
        print("❌ API connection failed. Please check your configuration.")
