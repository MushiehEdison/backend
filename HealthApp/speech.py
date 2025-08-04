import os
import requests
import base64
import logging
import hashlib
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import redis
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Redis for caching
try:
    redis_url = os.getenv('REDIS_URL', None)
    if redis_url:
        redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis for caching")
    else:
        logger.warning("REDIS_URL not set in environment variables. Caching disabled.")
        redis_client = None
except Exception as e:
    logger.warning(f"Redis connection failed: {str(e)}. Caching disabled.")
    redis_client = None

def pre_cache_common_phrases():
    """Pre-cache common healthcare phrases to handle Redis data loss on restart."""
    common_phrases = [
        ("Reminder: Take your medication at 8 PM.", "en"),
        ("Rappel : Prenez vos médicaments à 20h.", "fr"),
        ("Your next appointment is tomorrow at 10 AM.", "en"),
        ("Votre prochain rendez-vous est demain à 10h.", "fr"),
        ("Please drink water regularly.", "en"),
        ("Veuillez boire de l'eau régulièrement.", "fr")
    ]
    if not redis_client:
        logger.warning("Redis unavailable, skipping pre-caching")
        return

    try:
        key_count = redis_client.dbsize()
        if key_count == 0:
            logger.info("Redis keyspace empty, likely due to restart. Pre-caching common phrases.")
        else:
            logger.info(f"Redis contains {key_count} keys, checking for common phrases.")

        for text, lang in common_phrases:
            # Ensure text is encoded to bytes
            cache_key = f"tts:{hashlib.md5(f'{text}:{lang}:en-US-natalie' if lang == 'en' else 'fr-FR-denise'.encode('utf-8')).hexdigest()}"
            if not redis_client.exists(cache_key):
                logger.info(f"Pre-caching phrase: {text[:50]}... ({lang})")
                audio, error = generate_tts_audio(text, "system", "init", lang)
                if error:
                    logger.error(f"Failed to pre-cache phrase '{text}': {error}")
                else:
                    logger.info(f"Successfully pre-cached phrase: {text[:50]}...")
    except Exception as e:
        logger.error(f"Error in pre_cache_common_phrases: {str(e)}")

def generate_tts_audio(text, user_id, conversation_id, language="en"):
    """Generate speech using Murf AI TTS API with African-accented voices and return base64 audio."""
    api_key = os.getenv("MURF_API_KEY")
    if not api_key:
        logger.error("Murf AI API key not set in environment variables")
        return None, "TTS service unavailable: API key not configured"

    if not text or not isinstance(text, str) or text.strip() == "":
        logger.error(f"Invalid or empty text input for TTS: {text}")
        return None, "Invalid or empty text provided for audio generation"

    # Sanitize and normalize text
    text = text.strip()[:1000]  # Limit to 1000 chars to avoid API abuse
    if len(text) < 3:
        logger.warning(f"Text too short for TTS: {text}")
        return None, "Text too short for audio generation"
    logger.info(f"Processing TTS for text: {text[:50]}..., language: {language}, user_id: {user_id}, conversation_id: {conversation_id}")

    # Voice map for African-like English/French accents
    voice_map = {
        "en": "en-US-natalie",  # Natural female voice, customized for African English
        "fr": "fr-FR-denise"    # Natural female voice, customized for African French
    }
    voice_id = voice_map.get(language, voice_map["en"])
    logger.debug(f"Selected voice_id: {voice_id} for language: {language}")

    # Generate cache key
    cache_key = f"tts:{hashlib.md5(f'{text}:{language}:{voice_id}'.encode('utf-8')).hexdigest()}"
    if redis_client:
        try:
            cached_audio = redis_client.get(cache_key)
            if cached_audio:
                logger.info(f"Cache hit for key: {cache_key}")
                return cached_audio, None
            else:
                logger.info(f"Cache miss for key: {cache_key}. Checking if Redis is empty.")
                if redis_client.dbsize() == 0:
                    logger.warning("Redis keyspace empty, likely due to restart. Regenerating audio.")
        except Exception as e:
            logger.warning(f"Redis cache read failed: {str(e)}. Proceeding without cache.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "voiceId": voice_id,
        "format": "MP3",
        "sample_rate": 24000,  # Fixed: Changed from sampleRate to sample_rate
        "pitch": "low",  # Lower pitch for African cadence
        "speed": 0.9,    # Slower for natural rhythm
        "prosody": "expressive"  # Melodic intonation for African-like accents
    }

    # Set up retry strategy
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    try:
        logger.debug(f"Sending TTS request to Murf AI API with payload: {json.dumps(payload, indent=2)}")
        response = session.post(
            "https://api.murf.ai/v1/speech/generate",
            json=payload,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()

        audio_content = response.content
        if not audio_content or len(audio_content) < 100:
            logger.error(f"Invalid or empty audio content received from Murf AI API: {len(audio_content)} bytes, response headers: {response.headers}")
            return None, "No audio received from TTS service"

        # Encode audio as base64
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        if not audio_base64:
            logger.error("Failed to encode audio to base64")
            return None, "Failed to encode audio data"

        # Cache the result for 1 hour
        if redis_client:
            try:
                redis_client.setex(cache_key, 3600, audio_base64)
                logger.info(f"Cached audio for key: {cache_key} with 1-hour TTL")
            except Exception as e:
                logger.warning(f"Redis cache write failed: {str(e)}")

        logger.info(f"Generated base64 audio, length: {len(audio_base64)}")
        return audio_base64, None

    except requests.exceptions.HTTPError as http_err:
        error_msg = f"TTS service error: {response.status_code} - {response.text[:200] if response else 'No response'}"
        logger.error(f"HTTP error from Murf AI API: {str(http_err)}, Response: {response.text[:200] if response else 'No response'}, Headers: {response.headers if response else 'No response'}")
        return None, error_msg
    except requests.exceptions.Timeout:
        logger.error("Murf AI API request timed out after 10 seconds")
        return None, "TTS service timed out"
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error generating TTS audio: {str(req_err)}")
        return None, "Failed to connect to TTS service"
    except Exception as e:
        logger.error(f"Unexpected error in generate_tts_audio: {str(e)}")
        return None, f"TTS generation failed: {str(e)}"

# Run pre-caching on app startup
try:
    pre_cache_common_phrases()
except Exception as e:
    logger.error(f"Failed to pre-cache phrases on startup: {str(e)}")