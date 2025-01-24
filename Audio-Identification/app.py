import os
from typing import Dict, List
from datetime import datetime
from contextlib import asynccontextmanager
import httpx
import base64
import numpy as np
import librosa
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from twilio.rest import Client
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from dotenv import load_dotenv
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define the local directory for Qdrant storage
QDRANT_PATH = "local_qdrant"

def convert_numpy_to_python(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    return obj

# Initialize FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create Qdrant collection if it doesn't exist
    try:
        qdrant_client.create_collection(
            collection_name="audio_embeddings",
            vectors_config=models.VectorParams(
                size=1536,  # Size for text-embedding-ada-002
                distance=models.Distance.COSINE
            )
        )
        logger.info("Created Qdrant collection")
    except Exception as e:
        logger.info(f"Collection might already exist: {e}")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down application")

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

class Settings(BaseSettings):
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_PHONE_NUMBER: str = "+14155238886"  # Twilio sandbox number. Its always constant
    USER_WHATSAPP_NUMBER: str = "+12345678900"  # Your WhatsApp number
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    class Config:
        env_file = ".env"

settings = Settings()

# Initialize clients
twilio_client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
qdrant_client = QdrantClient(path=QDRANT_PATH)
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

class AudioRecord(BaseModel):
    id: str
    features: List[float]
    metadata: Dict

class AudioProcessor:
    def __init__(self):
        self.client = openai_client
        
    async def analyze_voice_characteristics(self, audio_path: str) -> Dict:
        """Analyze voice characteristics using librosa audio features."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path)
           
            # Extract comprehensive audio features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
           
            # Calculate pitch stability (using pitch deviation)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_stability = 1.0 - np.std(pitches[magnitudes > np.max(magnitudes)*0.1])/100
           
            # Calculate rhythm stability using onset strength
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
            rhythm_stability = np.mean(pulse)
           
            # Calculate voice texture (spectral contrast)
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            texture_score = np.mean(contrast)
           
            # Normalize scores to 0-1 range
            def normalize(value, min_val=0, max_val=1):
                return float(np.clip((value - min_val) / (max_val - min_val), 0, 1))
           
            # Create feature vector of 5 normalized scores
            voice_features = [
                normalize(pitch_stability),  # Pitch stability
                normalize(texture_score, 0, 100),  # Voice texture
                normalize(rhythm_stability),  # Speech rhythm
                normalize(np.mean(spectral_centroid)),  # Vocal resonance
                normalize(np.mean(mfccs))  # Articulation clarity
            ]
           
            return {
                'voice_features': voice_features,
                'mfccs': mfccs.mean(axis=1).tolist(),
                'spectral_centroid': float(spectral_centroid.mean())
            }
           
        except Exception as e:
            logger.error(f"Error analyzing voice: {e}")
            raise

    async def transcribe_and_embed(self, audio_path: str) -> tuple[str, List[float], Dict]:
        """Transcribe audio, create embedding, and analyze voice characteristics"""
        try:
            print(f"Transcribing audio from: {audio_path}")
            
            # Get voice characteristics
            voice_analysis = await self.analyze_voice_characteristics(audio_path)
            
            # Transcribe audio
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            print(f"Transcription: {transcript.text}")
            
            # Get embedding using text-embedding-ada-002
            embedding_response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=transcript.text
            )
            
            embedding = np.array(embedding_response.data[0].embedding).tolist()  # Convert to list
            
            return transcript.text, embedding, voice_analysis
            
        except Exception as e:
            logger.error(f"Error in transcribe_and_embed: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")

async def download_audio(url: str, save_path: str):
    """Download audio file from URL with redirect handling"""
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            auth = base64.b64encode(
                f"{settings.TWILIO_ACCOUNT_SID}:{settings.TWILIO_AUTH_TOKEN}".encode()
            ).decode()
            headers = {"Authorization": f"Basic {auth}"}
            
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            with open(save_path, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Successfully downloaded audio to {save_path}")
                
    except httpx.HTTPError as e:
        logger.error(f"HTTP error downloading audio: {e}")
        if e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response headers: {e.response.headers}")
        raise HTTPException(status_code=400, detail=f"Error downloading audio: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error downloading audio: {e}")
        raise HTTPException(status_code=400, detail=f"Error downloading audio: {str(e)}")

async def search_similar_audio(features: List[float], limit: int = 5):
    """Search for similar audio in Qdrant"""
    try:
        results = qdrant_client.search(
            collection_name="audio_embeddings",
            query_vector=features,
            limit=limit
        )
        return results
    except Exception as e:
        logger.error(f"Error searching similar audio: {e}")
        return []

async def get_recommendations(transcription: str) -> str:
    """Get music recommendations based on transcription using GPT-4"""
    try:
        system_prompt = """You are a music recommendation expert who helps users find songs based on their voice recordings. 
        For each recommendation, provide:
        1. Song name with artist
        2. A short reason why this song matches their voice/content
        3. A hypothetical YouTube or Spotify link (use format: youtube.com/watch?v=XXXX or open.spotify.com/track/XXXX)
        
        Provide exactly 3 recommendations that match their voice characteristics and content.
        Format each recommendation clearly with emojis and separate them with line breaks."""

        user_prompt = f"""Based on this transcription, recommend 3 specific songs that would match well:
        Transcription: {transcription}
        
        Format example:
        🎵 Song: [Song Name] by [Artist]
        💭 Why: [Brief reason this matches their voice/content]
        🔗 Listen: [Platform link]
        
        """

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return "Unable to generate music recommendations at this time."

async def store_audio_embedding(features: List[float], voice_features: Dict, metadata: Dict):
    """Store audio features in Qdrant"""
    try:
        print(f"Storing embedding with metadata: {metadata}")
        
        point_id = str(uuid.uuid4())
        metadata['id'] = point_id
        
        # Convert all numpy types to Python native types
        voice_features = convert_numpy_to_python(voice_features)
        combined_metadata = {
            **metadata,
            'voice_analysis': voice_features
        }
        
        qdrant_client.upsert(
            collection_name="audio_embeddings",
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=features,  # Already a list
                    payload=combined_metadata
                )
            ]
        )
        print(f"Successfully stored embedding with ID: {point_id}")
        
    except Exception as e:
        print(f"Error storing embedding: {e}")
        logger.error(f"Error storing embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error storing embedding: {str(e)}")

async def format_voice_analysis(voice_features: Dict) -> str:
    """Format voice analysis results into a readable message"""
    features = voice_features['voice_features']
    return (
        "🎤 Voice Analysis:\n"
        f"- Pitch Stability: {features[0]:.2f}\n"
        f"- Voice Texture: {features[1]:.2f}\n"
        f"- Speech Rhythm: {features[2]:.2f}\n"
        f"- Vocal Resonance: {features[3]:.2f}\n"
        f"- Articulation Clarity: {features[4]:.2f}\n"
    )

async def send_whatsapp_message(to: str, message: str):
    """Send WhatsApp message using Twilio"""
    try:
        twilio_client.messages.create(
            body=message,
            from_=f"whatsapp:{settings.TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{to}"
        )
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {e}")
        raise HTTPException(status_code=500, detail=f"Error sending message: {str(e)}")

async def process_audio_message(audio_url: str, sender: str):
    """Process incoming audio and send response"""
    temp_path = "temp_audio.mp3"
    try:
        print(f"Processing audio from URL: {audio_url}")
        await download_audio(audio_url, temp_path)

        audio_processor = AudioProcessor()
        transcription, embedding, voice_analysis = await audio_processor.transcribe_and_embed(temp_path)

        similar_results = await search_similar_audio(embedding)
        voice_analysis_text = await format_voice_analysis(voice_analysis)

        if similar_results:
            match = similar_results[0]
            metadata = match.payload
            recommendations = await get_recommendations(transcription)

            message = (
                f"🎤 Voice Analysis Complete!\n"
                f"📝 Your audio: {transcription}\n\n"
                f"{voice_analysis_text}\n"
                f"🎯 Similar content score: {match.score:.2f}\n\n"
                f"🎵 Music Recommendations:\n{recommendations}"
            )
        else:
            metadata = {
                "id": str(uuid.uuid4()),
                "url": audio_url,
                "transcription": transcription,
                "timestamp": str(datetime.now())
            }
            await store_audio_embedding(embedding, voice_analysis, metadata)
            
            # Still provide recommendations even if no similar content is found
            recommendations = await get_recommendations(transcription)
            message = (
                f"🎤 Voice Analysis Complete!\n"
                f"📝 Transcription: {transcription}\n\n"
                f"{voice_analysis_text}\n"
                f"🎵 Music Recommendations:\n{recommendations}\n\n"
                f"ℹ️ Your voice profile has been stored for future matching."
            )

        await send_whatsapp_message(sender, message)

    except Exception as e:
        logger.error(f"Error processing audio message: {e}")
        error_message = "Sorry, there was an error processing your audio."
        await send_whatsapp_message(sender, error_message)
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming WhatsApp messages"""
    try:
        print("Received webhook request")
        form_data = await request.form()
        print(f"Form data: {dict(form_data)}")
        
        media_url = form_data.get("MediaUrl0")
        sender = form_data.get("From", "").replace("whatsapp:", "")
        
        print(f"Media URL: {media_url}")
        print(f"Sender: {sender}")

        if media_url:
            background_tasks.add_task(
                process_audio_message,
                media_url,
                sender
            )
            return {"status": "success", "message": "Processing audio..."}
        else:
            await send_whatsapp_message(
                settings.USER_WHATSAPP_NUMBER,
                "Please send an audio message. Text messages are not supported."
            )
            return {"status": "error", "message": "No audio received"}

    except Exception as e:
        print(f"Error in webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Audio Recognition Service is running",
        "version": "2.0.0",
        "status": "online",
        "models": {
            "transcription": "whisper-1",
            "embedding": "text-embedding-ada-002",
            "recommendation": "gpt-4o"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "qdrant": "connected" if os.path.exists(QDRANT_PATH) else "not initialized",
            "openai": "configured" if openai_client else "not configured",
            "twilio": "configured" if twilio_client else "not configured"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)