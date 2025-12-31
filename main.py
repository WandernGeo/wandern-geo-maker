"""
Wandern Geo Echo Maker Cloud Function
Processes Geo Echo media (video, audio, image) and generates AR tokens.
"""
import functions_framework
import os
import json
import logging
import base64
import hashlib
from io import BytesIO
from flask import Request, jsonify
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cloud Storage config
GCS_BUCKET = os.environ.get("GCS_BUCKET", "wandern-geo-echoes")

# Maximum sizes for free Arweave tier
MAX_AR_SIZE = 100 * 1024  # 100KB


def generate_echo_id(content: str, user_id: str, timestamp: str) -> str:
    """Generate unique echo ID from content hash."""
    data = f"{user_id}:{content}:{timestamp}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def compress_image(image_bytes: bytes, max_size: int = MAX_AR_SIZE) -> bytes:
    """Compress image to fit within Arweave free tier."""
    try:
        from PIL import Image
        
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Progressively reduce quality until under max_size
        quality = 85
        while quality > 10:
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            compressed = buffer.getvalue()
            
            if len(compressed) <= max_size:
                logger.info(f"Compressed image to {len(compressed)} bytes at quality {quality}")
                return compressed
            
            quality -= 10
        
        # If still too large, resize
        width, height = img.size
        while len(compressed) > max_size and width > 100:
            width = int(width * 0.8)
            height = int(height * 0.8)
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=50, optimize=True)
            compressed = buffer.getvalue()
        
        return compressed
        
    except Exception as e:
        logger.error(f"Image compression failed: {e}")
        raise


def extract_audio_thumbnail(audio_bytes: bytes) -> dict:
    """
    Extract metadata and generate text representation for audio echo.
    Returns a JSON-serializable representation.
    """
    # For MVP: Return metadata only
    # Future: Use Google Speech-to-Text for transcription
    return {
        "type": "audio",
        "duration_estimate_seconds": len(audio_bytes) / 16000,  # Rough estimate
        "transcription": None,  # TODO: Integrate Speech-to-Text
        "generated_at": datetime.utcnow().isoformat()
    }


def extract_video_thumbnail(video_bytes: bytes) -> bytes:
    """
    Extract first frame from video as thumbnail.
    Returns JPEG bytes.
    """
    try:
        import tempfile
        import subprocess
        
        # Write video to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video_bytes)
            video_path = f.name
        
        # Extract frame with ffmpeg
        output_path = video_path + '_thumb.jpg'
        subprocess.run([
            'ffmpeg', '-i', video_path,
            '-ss', '00:00:01',
            '-vframes', '1',
            '-q:v', '2',
            output_path
        ], capture_output=True, check=True)
        
        with open(output_path, 'rb') as f:
            thumbnail = f.read()
        
        # Cleanup
        os.unlink(video_path)
        os.unlink(output_path)
        
        return thumbnail
        
    except Exception as e:
        logger.error(f"Video thumbnail extraction failed: {e}")
        return None


def generate_ar_token(echo_data: dict) -> dict:
    """
    Generate AR token representation for the Geo Echo.
    This is the visual/data representation stored on Arweave.
    """
    token = {
        "version": "1.0",
        "type": "geo-echo-token",
        "app": "wandern",
        "echo_id": echo_data.get("echo_id"),
        "content_type": echo_data.get("content_type"),
        "compressed_size": echo_data.get("compressed_size"),
        "location": echo_data.get("location"),
        "created_at": echo_data.get("created_at"),
        "thumbnail_b64": echo_data.get("thumbnail_b64"),
        "metadata": echo_data.get("metadata", {})
    }
    
    return token


@functions_framework.http
def process_echo(request: Request):
    """
    HTTP Cloud Function to process a Geo Echo.
    
    Accepts JSON:
    {
        "echo_id": "...",
        "user_id": "...",
        "content_type": "text" | "image" | "audio" | "video",
        "content_text": "...",
        "media_b64": "base64 encoded media...",
        "location": {"lat": ..., "lng": ...},
        "created_at": "ISO timestamp"
    }
    
    Returns JSON with AR token data.
    """
    # CORS Headers
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "3600"
        }
        return ("", 204, headers)

    headers = {"Access-Control-Allow-Origin": "*"}

    try:
        data = request.get_json(silent=True)
        if not data:
            return (jsonify({"error": "Invalid JSON"}), 400, headers)
        
        echo_id = data.get("echo_id")
        user_id = data.get("user_id")
        content_type = data.get("content_type", "text")
        content_text = data.get("content_text", "")
        media_b64 = data.get("media_b64")
        location = data.get("location", {})
        created_at = data.get("created_at", datetime.utcnow().isoformat())
        
        # Generate echo_id if not provided
        if not echo_id:
            echo_id = generate_echo_id(content_text or str(media_b64)[:100], user_id, created_at)
        
        result = {
            "echo_id": echo_id,
            "content_type": content_type,
            "location": location,
            "created_at": created_at,
            "compressed_size": 0,
            "thumbnail_b64": None,
            "metadata": {}
        }
        
        if content_type == "text":
            # Text echoes: Just validate and return
            result["metadata"] = {
                "char_count": len(content_text),
                "word_count": len(content_text.split())
            }
            result["compressed_size"] = len(content_text.encode('utf-8'))
            
        elif content_type == "image" and media_b64:
            # Image echoes: Compress for Arweave
            image_bytes = base64.b64decode(media_b64)
            compressed = compress_image(image_bytes)
            result["thumbnail_b64"] = base64.b64encode(compressed).decode('utf-8')
            result["compressed_size"] = len(compressed)
            result["metadata"] = {
                "original_size": len(image_bytes),
                "compression_ratio": len(image_bytes) / len(compressed)
            }
            
        elif content_type == "audio" and media_b64:
            # Audio echoes: Extract metadata, prepare for transcription
            audio_bytes = base64.b64decode(media_b64)
            audio_meta = extract_audio_thumbnail(audio_bytes)
            result["metadata"] = audio_meta
            result["compressed_size"] = len(audio_bytes)
            
        elif content_type == "video" and media_b64:
            # Video echoes: Extract thumbnail
            video_bytes = base64.b64decode(media_b64)
            thumbnail = extract_video_thumbnail(video_bytes)
            if thumbnail:
                compressed_thumb = compress_image(thumbnail)
                result["thumbnail_b64"] = base64.b64encode(compressed_thumb).decode('utf-8')
                result["compressed_size"] = len(compressed_thumb)
            result["metadata"] = {
                "original_size": len(video_bytes),
                "has_thumbnail": thumbnail is not None
            }
        
        # Generate AR token
        ar_token = generate_ar_token(result)
        
        return (jsonify({
            "success": True,
            "ar_token": ar_token,
            "ready_for_arweave": result["compressed_size"] <= MAX_AR_SIZE
        }), 200, headers)
        
    except Exception as e:
        logger.error(f"Echo processing failed: {e}")
        return (jsonify({"error": str(e)}), 500, headers)
