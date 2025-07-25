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
    r'\b(pregnancy|pregnant|expecting|baby|grossesse|enceinte|bébé)\b',
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

# Load clinical dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), "clinical_summaries.csv")
dataset_df = pd.DataFrame()
dataset_index = {}

try:
    dataset_df = pd.read_csv(DATASET_PATH).dropna(subset=[
        'summary_id', 'patient_id', 'patient_age', 'patient_gender',
        'diagnosis', 'body_temp_c', 'blood_pressure_systolic',
        'heart_rate', '',
        'summary_text', 'date_recorded'
    ])
    for idx, row in dataset_df.iterrows():
        diagnosis = str(row.get('diagnosis', '')).lower()
        summary = str(row.get('summary_text', '')).lower()
        if diagnosis not in ['', '']:
            logger.warning(f"Invalid diagnosis in row {idx}: {diagnosis}")
            if diagnosis not in dataset_index:
                dataset_index[diagnosis] = []
            dataset_index[diagnosis].append(idx)
        for word in words:
            if len(word) > 3:
                if word not in dataset_index:
                    dataset_index[word] = []
                dataset_index[word].append(idx)
    logger.info(f"Loaded dataset with {len(dataset_df)} records and indexed {len(dataset_index)} terms")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")

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
        logger.debug(f"Sending request to {GROQ_ENDPOINT} with model {model}, data: {json.dumps(data, indent=2)}")
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=data, timeout=20, verify=True)
        logger.debug(f"Received response status: {response.status_code}, text: {response.text[:200]}...")
        response.raise_for_status()
        result = response.json()
        if 'choices' in result and result['choices']:
            content = result['choices'][0]['message']['content'].strip()
            logger.info("Successfully received response from Grok API")
            return content
        logger.warning("No valid choices in API response")
        return "Error: No valid response from AI service"
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}, Response: {response.text if response else 'No response'}")
        return f"Error: HTTP error from AI service: {response.status_code if response else 'Unknown'}"
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error occurred: {req_err}")
        return "Error: Unable to connect to AI service"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"Error: Unexpected issue with AI service: {str(e)}"

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
- Langue : Français (détectée à partir de l'entrée utilisateur)

**Contexte de la conversation** :
- Messages récents : {len(recent_messages)} échanges
- Symptômes mentionnés : {', '.join(symptoms) if symptoms else 'Aucun'}
- Conditions mentionnées : {', '.join(conditions) if conditions else 'Aucune'}
- Sujets abordés : {', '.join(topics) if topics else 'Aucun'}
- État émotionnel actuel : {emotional_state}
- Progression émotionnelle : {', '.join(emotional_trend[-3:]) if emotional_trend else 'Aucune'}

**Instructions** :
1. Répondez uniquement en français, de manière naturelle, empathique et culturellement adaptée au Cameroun.
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
- Language: English (detected from user input)

**Conversation Context**:
- Recent messages: {len(recent_messages)} exchanges
- Symptoms mentioned: {', '.join(symptoms) if symptoms else 'None'}
- Conditions mentioned: {', '.join(conditions) if conditions else 'None'}
- Topics discussed: {', '.join(topics) if topics else 'None'}
- Current emotional state: {emotional_state}
- Emotional progression: {', '.join(emotional_trend[-3:]) if emotional_trend else 'None'}

**Instructions**:
1. Respond only in English, naturally, empathetically, and in a culturally appropriate manner for Cameroon.
2. Use a conversational tone, incorporating emojis to express empathy (😊, 😔, 🤗, 🙏).
3. Build on the conversation history, avoiding repetition.
4. Use clinical data only if relevant to the mentioned symptoms or conditions.
5. Ask relevant follow-up questions based on symptoms, conditions, and emotions.
6. If the query is vague, seek clarification to provide better assistance.
7. Provide practical, accessible advice tailored to the Cameroonian context."""
    return prompt

def generate_personalized_response(user_input, patient_info, session_id="default", history=None):
    """Generate AI-driven, context-aware response in the detected language"""
    if not user_input or not patient_info:
        default_lang = patient_info.get('language', 'en') if patient_info else 'en'
        error_msg = "J'ai besoin de plus d'informations pour vous aider correctement. Pouvez-vous partager plus de détails ? 😊" if default_lang == "fr" else "I need more information to assist you properly. Could you share more details? 😊"
        logger.warning(f"Invalid input or patient info, returning: {error_msg}")
        return error_msg

    memory = conversation_memories[session_id]
    
    # Detect language from user input, fallback to login preference
    detected_lang = detect_language(user_input)
    memory.user_preferences["language"] = detected_lang
    logger.debug(f"Detected language: {detected_lang}, user input: {user_input}")

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
    conversation_history = "\n".join([f"{'Utilisateur' if msg['is_user'] else 'Assistant' if memory.user_preferences['language'] == 'fr' else 'User' if msg['is_user'] else 'Assistant'}: {msg['text']}" for msg in context['recent_messages'][-5:]])
    dataset_context = "\n".join([f"{'Cas' if memory.user_preferences['language'] == 'fr' else 'Case'} {i}: {r['diagnosis']} - {r['summary']}" for i, r in enumerate(dataset_records, 1)]) if dataset_records else "Aucune donnée clinique pertinente." if memory.user_preferences["language"] == "fr" else "No relevant clinical data."

    full_prompt = f"{conversation_history}\n\n{'Utilisateur' if memory.user_preferences['language'] == 'fr' else 'User'}: {user_input}\n\n{'Données cliniques pertinentes' if memory.user_preferences['language'] == 'fr' else 'Relevant Clinical Data'}: {dataset_context}"

    # Call Grok API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt}
    ]
    response = call_groq_api(messages)

    if response and not response.startswith("Error:"):
        memory.add_message(response, is_user=False, sentiment="EMPATHETIC")
        logger.info(f"Generated AI response: {response[:100]}...")
        return response

    # Fallback response in detected language
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
    test_patients = [
        {
            'name': 'Marie Ngozi',
            'age': 28,
            'language': 'en',  # Login preference, may be overridden by detection
            'region': 'Douala',
            'chronic_conditions': 'None'
        },
        {
            'name': 'Jean Dupont',
            'age': 35,
            'language': 'fr',  # Login preference, may be overridden by detection
            'region': 'Yaoundé',
            'chronic_conditions': 'Hypertension'
        }
    ]
    test_conversations = [
        ("I'm having headaches lately", "J'ai des maux de tête récemment"),
        ("It's worse in the evenings and I feel nauseous", "C'est pire le soir et je me sens nauséeux"),
        ("I’m feeling a bit better today, but still worried", "Je me sens un peu mieux aujourd'hui, mais toujours inquiet"),
        ("Thank you for your help!", "Merci pour votre aide !")
    ]
    session_id = "test_session"

    for patient in test_patients:
        lang = patient['language']
        print(f"\n📱 Conversation Simulation (Login preference: {'English' if lang == 'en' else 'French'}):")
        print("=" * 50)
        conversation_memories[session_id] = ConversationMemory()  # Reset memory
        # Test mixed inputs to verify language detection
        test_inputs = [
            test_conversations[0][1 if patient['language'] == 'en' else 0],  # Opposite language to test detection
            test_conversations[1][0 if patient['language'] == 'fr' else 1],  # Same language
            test_conversations[2][1 if patient['language'] == 'en' else 0],  # Opposite language
            test_conversations[3][0 if patient['language'] == 'fr' else 1]   # Same language
        ]
        for i, message in enumerate(test_inputs, 1):
            print(f"\n👤 {'Utilisateur' if detect_language(message) == 'fr' else 'User'} (Message {i}): {message}")
            response = generate_personalized_response(message, patient, session_id, [])
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