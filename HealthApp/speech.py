import os
import requests

def generate_tts_audio(text, language, user_id, conversation_id):
    api_key = os.getenv("ELEVEN_LABS_API_KEY")
    if not api_key:
        raise ValueError("Eleven Labs API key not found in environment variables")
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.6,
            "similarity_boost": 0.8,
            "style": 0.1,
            "use_speaker_boost": True
        }
    }
    response = requests.post(
        "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM",
        headers=headers,
        json=payload
    )
    if response.status_code != 200:
        raise Exception(f"HTTP error from Eleven Labs API: {response.status_code} - {response.text}")
    return response.content  # Return audio bytes