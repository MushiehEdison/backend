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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        if is_user and sentiment:
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
            'follow_ups': self.follow_up_needed
        }

# Global conversation memories
conversation_memories = defaultdict(ConversationMemory)

# Patterns for entity extraction
SYMPTOM_PATTERNS = [
    r'\b(pain|hurt|ache|sore|burning|throbbing|sharp|dull|cramping)\b',
    r'\b(fever|temperature|hot|chills|sweating)\b',
    r'\b(nausea|vomit|sick|dizzy|headache|migraine)\b',
    r'\b(cough|sneeze|runny nose|congestion|sore throat)\b',
    r'\b(tired|fatigue|exhausted|weak|energy)\b',
    r'\b(sleep|insomnia|restless|nightmare)\b',
    r'\b(appetite|eating|weight|stomach)\b',
    r'\b(breathing|shortness|chest|heart)\b',
    r'\b(skin|rash|itchy|red|swollen)\b',
    r'\b(joint|muscle|back|neck|shoulder)\b'
]

CONDITION_PATTERNS = [
    r'\b(diabetes|hypertension|blood pressure|sugar)\b',
    r'\b(malaria|typhoid|cholera|yellow fever)\b',
    r'\b(asthma|bronchitis|pneumonia|tuberculosis)\b',
    r'\b(cancer|tumor|growth|lump)\b',
    r'\b(depression|anxiety|stress|mental health)\b',
    r'\b(pregnancy|pregnant|expecting|baby)\b',
    r'\b(allergy|allergic|reaction)\b'
]

EMOTIONAL_INDICATORS = {
    'VERY_POSITIVE': r'\b(amazing|fantastic|wonderful|excellent|perfect|thrilled|ecstatic)\b',
    'POSITIVE': r'\b(good|better|great|happy|pleased|glad|improving|recovering)\b',
    'NEUTRAL': r'\b(okay|fine|normal|regular|usual|moderate)\b',
    'CONCERNED': r'\b(worried|concerned|anxious|nervous|unsure|confused)\b',
    'NEGATIVE': r'\b(bad|worse|terrible|awful|sick|ill|pain|hurt)\b',
    'VERY_NEGATIVE': r'\b(horrible|unbearable|excruciating|devastating|hopeless|desperate)\b'
}

# Load clinical dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), "clinical_summaries.csv")
dataset_df = pd.DataFrame()
dataset_index = {}

try:
    dataset_df = pd.read_csv(DATASET_PATH).dropna(subset=[
        'summary_id', 'patient_id', 'patient_age', 'patient_gender',
        'diagnosis', 'body_temp_c', 'blood_pressure_systolic',
        'heart_rate', 'summary_text', 'date_recorded'
    ])
    for idx, row in dataset_df.iterrows():
        diagnosis = str(row['diagnosis']).lower()
        summary = str(row['summary_text']).lower()
        if diagnosis not in dataset_index:
            dataset_index[diagnosis] = []
        dataset_index[diagnosis].append(idx)
        words = re.findall(r'\b\w+\b', summary)
        for word in words:
            if len(word) > 3:
                if word not in dataset_index:
                    dataset_index[word] = []
                dataset_index[word].append(idx)
    logger.info(f"Loaded dataset with {len(dataset_df)} records and indexed {len(dataset_index)} terms")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")

def extract_entities(text):
    """Extract symptoms, conditions, and topics from user input"""
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

    if any(word in text_lower for word in ['diet', 'food', 'eat', 'nutrition']):
        topics.add('nutrition')
    if any(word in text_lower for word in ['exercise', 'sport', 'physical', 'activity']):
        topics.add('physical_activity')
    if any(word in text_lower for word in ['mental', 'stress', 'anxiety', 'depression']):
        topics.add('mental_health')
    if any(word in text_lower for word in ['medicine', 'medication', 'drug', 'treatment']):
        topics.add('medication')

    return topics, symptoms, conditions

def detect_emotional_state(text):
    """Detect user's emotional state"""
    text_lower = text.lower()
    for state, pattern in EMOTIONAL_INDICATORS.items():
        if re.search(pattern, text_lower):
            return state
    return 'NEUTRAL'

def query_dataset(symptoms, conditions, max_records=3):
    """Query dataset for relevant clinical records"""
    if dataset_df.empty:
        return []

    relevant_indices = set()
    search_terms = list(symptoms) + list(conditions)

    for term in search_terms:
        term_clean = term.lower().strip()
        if term_clean in dataset_index:
            relevant_indices.update(dataset_index[term_clean][:3])

    if not relevant_indices:
        for term in search_terms:
            for indexed_term in dataset_index.keys():
                if difflib.SequenceMatcher(None, term.lower(), indexed_term).ratio() > 0.7:
                    relevant_indices.update(dataset_index[indexed_term][:2])

    records = []
    for idx in list(relevant_indices)[:max_records]:
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
            'summary': str(row.get('summary_text', ''))[:200] + '...'
        })
    return records

def call_groq_api(messages, model="llama3-70b-8192", max_tokens=600, temperature=0.8):
    """Call Grok API with improved error handling"""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set in environment variables")
        return None

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
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=data, timeout=20, verify=True)
        response.raise_for_status()
        result = response.json()
        if 'choices' in result and result['choices']:
            content = result['choices'][0]['message']['content'].strip()
            logger.info("Successfully received response from Grok API")
            return content
        logger.warning("No valid choices in API response")
        return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}, Response: {response.text if response else 'No response'}")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error occurred: {req_err}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None

def build_system_prompt(patient_info, context, emotional_state, language):
    """Build dynamic system prompt for AI"""
    name = patient_info.get('name', 'Patient')
    age = patient_info.get('age', 'N/A')
    region = patient_info.get('region', 'Cameroon')

    recent_messages = context['recent_messages']
    symptoms = context['symptoms_mentioned']
    conditions = context['conditions_mentioned']
    topics = context['topics_covered']
    emotional_trend = [e['sentiment'] for e in context['emotional_progression']]

    if language == "fr":
        prompt = f"""Vous êtes Dr. Claude, un assistant médical intelligent et empathique pour les patients camerounais.

**Profil du patient** :
- Nom : {name}
- Âge : {age} ans
- Région : {region}
- Langue : Français

**Contexte de la conversation** :
- Messages récents : {len(recent_messages)} échanges
- Symptômes mentionnés : {', '.join(symptoms) if symptoms else 'Aucun'}
- Conditions mentionnées : {', '.join(conditions) if conditions else 'Aucune'}
- Sujets abordés : {', '.join(topics) if topics else 'Aucun'}
- État émotionnel actuel : {emotional_state}
- Progression émotionnelle : {', '.join(emotional_trend[-3:]) if emotional_trend else 'Aucune'}

**Instructions** :
1. Répondez de manière naturelle, empathique et culturellement adaptée au Cameroun.
2. Utilisez un ton conversationnel, avec des emojis pour exprimer l'empathie (😊, 😔, 🤗, 🙏).
3. Construisez sur l'historique de la conversation, en évitant les répétitions.
4. Si des données cliniques sont fournies, utilisez-les uniquement si elles sont pertinentes.
5. Posez des questions de suivi pertinentes basées sur les symptômes, conditions et émotions.
6. Si la requête est vague, demandez des précisions pour mieux aider.
7. Fournissez des conseils pratiques et accessibles, adaptés au contexte camerounais."""
    else:
        prompt = f"""You are Dr. Claude, an intelligent and empathetic medical assistant for Cameroonian patients.

**Patient Profile**:
- Name: {name}
- Age: {age} years
- Region: {region}
- Language: English

**Conversation Context**:
- Recent messages: {len(recent_messages)} exchanges
- Symptoms mentioned: {', '.join(symptoms) if symptoms else 'None'}
- Conditions mentioned: {', '.join(conditions) if conditions else 'None'}
- Topics discussed: {', '.join(topics) if topics else 'None'}
- Current emotional state: {emotional_state}
- Emotional progression: {', '.join(emotional_trend[-3:]) if emotional_trend else 'None'}

**Instructions**:
1. Respond naturally, empathetically, and in a culturally appropriate manner for Cameroon.
2. Use a conversational tone, incorporating emojis to express empathy (😊, 😔, 🤗, 🙏).
3. Build on the conversation history, avoiding repetition.
4. Use clinical data only if relevant to the mentioned symptoms or conditions.
5. Ask relevant follow-up questions based on symptoms, conditions, and emotions.
6. If the query is vague, seek clarification to provide better assistance.
7. Provide practical, accessible advice tailored to the Cameroonian context."""
    return prompt

def generate_personalized_response(user_input, patient_info, session_id="default", history=None):
    """Generate AI-driven, context-aware response"""
    if not user_input or not patient_info:
        return "I need more information to assist you properly. Could you share more details? 😊"

    memory = conversation_memories[session_id]
    memory.user_preferences["language"] = patient_info.get('language', 'en')

    # Initialize memory with database history if provided
    if history:
        memory.load_history(history)

    # Extract entities and emotional state
    topics, symptoms, conditions = extract_entities(user_input)
    emotional_state = detect_emotional_state(user_input)
    memory.add_message(user_input, is_user=True, sentiment=emotional_state, topics=topics)
    memory.mentioned_symptoms.update(symptoms)
    memory.mentioned_conditions.update(conditions)

    # Get conversation context
    context = memory.get_context_summary()

    # Query dataset if relevant
    dataset_records = query_dataset(memory.mentioned_symptoms, memory.mentioned_conditions) if symptoms or conditions else []

    # Build prompt
    system_prompt = build_system_prompt(patient_info, context, emotional_state, memory.user_preferences["language"])
    conversation_history = "\n".join([f"{'User' if msg['is_user'] else 'Assistant'}: {msg['text']}" for msg in context['recent_messages'][-5:]])
    dataset_context = "\n".join([f"Case {i}: {r['diagnosis']} - {r['summary']}" for i, r in enumerate(dataset_records, 1)]) if dataset_records else "No relevant clinical data."
    
    full_prompt = f"{conversation_history}\n\nUser: {user_input}\n\nRelevant Clinical Data: {dataset_context}"

    # Call Grok API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt}
    ]
    response = call_groq_api(messages)

    if response:
        memory.add_message(response, is_user=False, sentiment="EMPATHETIC")
        logger.info(f"Generated AI response: {response[:100]}...")
        return response

    # Fallback: Construct minimal response with context
    name = patient_info.get('name', 'Patient')
    lang = memory.user_preferences["language"]
    fallback = f"{'Je suis désolé, je rencontre un problème technique.' if lang == 'fr' else 'I’m sorry, I’m unable to connect to the AI service at the moment.'} "
    if symptoms:
        fallback += f"{'Vous avez mentionné ' if lang == 'fr' else 'You mentioned '} {', '.join(list(symptoms)[:2])}. "
    fallback += f"{'Parlez-moi plus de ce que vous ressentez pour que je puisse mieux vous aider 😊' if lang == 'fr' else 'Please tell me more about how you’re feeling so I can assist you better 😊'}"
    memory.add_message(fallback, is_user=False, sentiment="EMPATHETIC")
    logger.warning(f"Fallback response generated: {fallback}")
    return fallback

def test_conversation_flow():
    """Test the AI-driven conversation system"""
    print("🧪 Testing AI Medical Assistant\n")
    test_patient = {
        'name': 'Marie Ngozi',
        'age': 28,
        'language': 'en',
        'region': 'Douala',
        'chronic_conditions': 'None'
    }
    test_conversations = [
        "I'm having headaches lately",
        "It's worse in the evenings and I feel nauseous",
        "I’m feeling a bit better today, but still worried",
        "Thank you for your help!"
    ]
    session_id = "test_session"

    print("📱 Conversation Simulation:")
    print("=" * 50)
    for i, message in enumerate(test_conversations, 1):
        print(f"\n👤 User (Message {i}): {message}")
        response = generate_personalized_response(message, test_patient, session_id, [])
        print(f"🤖 Dr. Healia: {response}")
        print("-" * 30)

    memory = conversation_memories[session_id]
    print(f"\n🧠 Conversation Memory Summary:")
    print(f"Total messages: {len(memory.messages)}")
    print(f"Symptoms: {memory.mentioned_symptoms}")
    print(f"Emotional progression: {[e['sentiment'] for e in memory.emotional_state_history]}")
    print(f"Topics: {memory.topics_discussed}")

if __name__ == "__main__":
    test_conversation_flow()