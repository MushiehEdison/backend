import os
import requests
import base64
import logging
import hashlib
from pathlib import Path
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

# Initialize Redis for caching (optional, configure for Render)
try:
    redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'), decode_responses=True)
    redis_client.ping()
    logger.info("Connected to Redis for caching")
except Exception as e:
    logger.warning(f"Redis connection failed: {str(e)}. Caching disabled.")
    redis_client = None

def generate_tts_audio(text, user_id, conversation_id, language="en"):
    """Generate speech using Eleven Labs TTS API and return base64 audio with caching and retries."""
    api_key = os.getenv("ELEVEN_LABS_API_KEY")
    if not api_key:
        logger.error("Eleven Labs API key not set in environment variables")
        return None, "TTS service unavailable: API key not configured"

    if not text or not isinstance(text, str) or text.strip() == "":
        logger.error("Invalid or empty text input for TTS")
        return None, "Invalid or empty text provided for audio generation"

    # Sanitize and normalize text
    text = text.strip()[:1000]  # Limit to 1000 chars to avoid API abuse
    logger.debug(f"Processing TTS for text: {text[:50]}..., language: {language}")

    # Extended voice map for multilingual support
    voice_map = {
        "en": "21m00Tcm4TlvDq8ikWAM",  # Bella: Natural, young female voice
        "fr": "flq6f7yk4E4fJM5XTYuZ",  # Ariane: Natural French female voice
    }
    voice_id = voice_map.get(language, voice_map["en"])
    logger.debug(f"Selected voice_id: {voice_id} for language: {language}")

    # Generate cache key based on text and language
    cache_key = f"tts:{hashlib.md5(f'{text}:{language}:{voice_id}'.encode()).hexdigest()}"
    if redis_client:
        try:
            cached_audio = redis_client.get(cache_key)
            if cached_audio:
                logger.info(f"Cache hit for key: {cache_key}")
                return cached_audio, None
        except Exception as e:
            logger.warning(f"Redis cache read failed: {str(e)}. Proceeding without cache.")

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.6,  # Slightly increased for consistency
            "similarity_boost": 0.8,
            "style": 0.1,  # Reduced for naturalness
            "use_speaker_boost": True
        }
    }

    # Set up retry strategy
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    try:
        logger.debug(f"Sending TTS request to Eleven Labs API with payload: {json.dumps(payload, indent=2)}")
        response = session.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            json=payload,
            headers=headers,
            timeout=10  # Reduced timeout for faster response
        )
        response.raise_for_status()

        audio_content = response.content
        if not audio_content or len(audio_content) < 100:
            logger.error("Invalid or empty audio content received from Eleven Labs API")
            return None, "Invalid audio content received from TTS service"

        # Encode audio as base64
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        if not audio_base64:
            logger.error("Failed to encode audio to base64")
            return None, "Failed to encode audio data"

        # Cache the result for 24 hours
        if redis_client:
            try:
                redis_client.setex(cache_key, 86400, audio_base64)
                logger.info(f"Cached audio for key: {cache_key}")
            except Exception as e:
                logger.warning(f"Redis cache write failed: {str(e)}")

        logger.debug(f"Generated base64 audio, length: {len(audio_base64)}")
        return audio_base64, None

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error from Eleven Labs API: {str(http_err)}, Response: {response.text[:200] if response else 'No response'}")
        return None, f"TTS service error: {response.status_code if response else 'Unknown'}"
    except requests.exceptions.Timeout:
        logger.error("Eleven Labs API request timed out")
        return None, "TTS service timed out"
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error generating TTS audio: {str(req_err)}")
        return None, "Failed to connect to TTS service"
    except Exception as e:
        logger.error(f"Unexpected error in generate_tts_audio: {str(e)}")
        return None, f"Unexpected error in TTS generation: {str(e)}"