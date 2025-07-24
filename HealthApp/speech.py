import os
import requests
import base64
import logging
from pathlib import Path
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def generate_tts_audio(text, user_id, conversation_id, language="en"):
    """Generate speech using Eleven Labs TTS API and return base64 audio."""
    api_key = os.getenv("ELEVEN_LABS_API_KEY")
    if not api_key:
        logger.error("Eleven Labs API key not set")
        return None

    # Select voice based on language (default to Bella for English, Ariane for French)
    voice_map = {
        "en": "21m00Tcm4TlvDq8ikWAM",  # Bella: Natural, young female voice
        "fr": "flq6f7yk4E4fJM5XTYuZ"   # Ariane: Natural French female voice
    }
    voice_id = voice_map.get(language, voice_map["en"])
    logger.debug(f"Selected voice_id: {voice_id} for language: {language}")

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",  # High-quality multilingual model
        "voice_settings": {
            "stability": 0.5,  # Balanced consistency
            "similarity_boost": 0.8,  # High voice fidelity
            "style": 0.2,  # Slight expressiveness
            "use_speaker_boost": True
        }
    }

    try:
        # Make request to Eleven Labs API
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        audio_content = response.content

        # Ensure audio directory exists
        audio_dir = os.path.join("static", "audio")
        os.makedirs(audio_dir, exist_ok=True)

        # Save to user-specific file
        audio_file_path = os.path.join(audio_dir, f"user_{user_id}_conv_{conversation_id}.mp3")
        with open(audio_file_path, "wb") as f:
            f.write(audio_content)
        logger.debug(f"Saved audio to: {audio_file_path}")

        # Encode audio as base64 for secure frontend playback
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        logger.debug("Generated base64 audio successfully")
        return audio_base64

    except requests.exceptions.RequestException as e:
        logger.error(f"Error generating TTS audio: {str(e)}")
        return None