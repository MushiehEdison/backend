import os
import requests
import base64
import logging
import hashlib
from dotenv import load_dotenv
import redis
import json
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioDataStream, SpeechSynthesisOutputFormat

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Redis for caching (optional, configure for Render)
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

def generate_tts_audio(text, user_id, conversation_id, language="en"):
    """Generate speech using Azure TTS API and return base64 audio with caching."""
    api_key = os.getenv("AZURE_TTS_KEY")
    region = os.getenv("AZURE_TTS_REGION")
    if not api_key or not region:
        logger.error("Azure TTS API key or region not set in environment variables")
        return None, "TTS service unavailable: API key or region not configured"

    if not text or not isinstance(text, str) or text.strip() == "":
        logger.error(f"Invalid or empty text input for TTS: {text}")
        return None, "Invalid or empty text provided for audio generation"

    # Sanitize and normalize text
    text = text.strip()[:1000]  # Limit to 1000 chars to avoid abuse
    if len(text) < 3:
        logger.warning(f"Text too short for TTS: {text}")
        return None, "Text too short for audio generation"
    logger.info(f"Processing TTS for text: {text[:50]}..., language: {language}, user_id: {user_id}, conversation_id: {conversation_id}")

    # Voice map for multilingual support
    voice_map = {
        "en": "en-US-AriaNeural",  # English: Aria, natural female voice
        "fr": "fr-FR-DeniseNeural",  # French: Denise
        "es": "es-ES-ElviraNeural",  # Spanish: Elvira
        "de": "de-DE-KatjaNeural",  # German: Katja
        "hi": "hi-IN-SwaraNeural",  # Hindi: Swara
        "ja": "ja-JP-NanamiNeural"  # Japanese: Nanami
    }
    voice_name = voice_map.get(language, voice_map["en"])
    logger.debug(f"Selected voice: {voice_name} for language: {language}")

    # Generate cache key based on text and language
    cache_key = f"tts:{hashlib.md5(f'{text}:{language}:{voice_name}'.encode()).hexdigest()}"
    if redis_client:
        try:
            cached_audio = redis_client.get(cache_key)
            if cached_audio:
                logger.info(f"Cache hit for key: {cache_key}")
                return cached_audio, None
        except Exception as e:
            logger.warning(f"Redis cache read failed: {str(e)}. Proceeding without cache.")

    try:
        # Initialize Azure TTS
        speech_config = SpeechConfig(subscription=api_key, region=region)
        speech_config.speech_synthesis_voice_name = voice_name
        speech_config.set_speech_synthesis_output_format(SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
        synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        logger.debug(f"Initialized Azure TTS with voice: {voice_name}")

        # Generate audio
        result = synthesizer.speak_text_async(text).get()
        if result.reason != result.reason.Succeeded:
            error_message = result.error_details or "Unknown error"
            logger.error(f"Azure TTS synthesis failed: {error_message}")
            return None, f"TTS service error: {error_message}"

        audio_stream = AudioDataStream(result)
        audio_io = BytesIO()
        audio_stream.save_to_wav_file(audio_io)
        audio_content = audio_io.getvalue()

        if not audio_content or len(audio_content) < 100:
            logger.error(f"Invalid or empty audio content received from Azure TTS: {len(audio_content)} bytes")
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

        logger.info(f"Generated base64 audio, length: {len(audio_base64)}")
        return audio_base64, None

    except Exception as e:
        logger.error(f"Error in Azure TTS generation: {str(e)}")
        return None, f"TTS service error: {str(e)}"