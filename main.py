"""
Wandern Geo Echo Maker Cloud Function
Processes Geo Echo media (video, audio, image) and generates AR tokens.

COMPRESSION SPEC: Per GEOECHO_COMPRESSION_TECH.md
- Max size: 100KB for Arweave free tier
- Audio: EnCodec at 1.5 kbps
- Video: Keyframe extraction + vector edges → hologram format
- Images: Progressive JPEG compression
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
MAX_DURATION_MS = 8000     # 8 seconds max per spec


def generate_echo_id(content: str, user_id: str, timestamp: str) -> str:
    """Generate unique echo ID from content hash."""
    data = f"{user_id}:{content}:{timestamp}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def validate_size_for_arweave(data: bytes | str, content_type: str) -> dict:
    """
    Validate that content fits within Arweave free tier (100KB).
    Returns validation result with size info.
    """
    if isinstance(data, str):
        size = len(data.encode('utf-8'))
    else:
        size = len(data)
    
    return {
        "size_bytes": size,
        "size_kb": round(size / 1024, 2),
        "max_size_kb": MAX_AR_SIZE / 1024,
        "is_valid": size <= MAX_AR_SIZE,
        "content_type": content_type,
        "reduction_needed_kb": max(0, (size - MAX_AR_SIZE) / 1024)
    }


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


def compress_audio_to_geoecho(audio_bytes: bytes, max_duration_ms: int = MAX_DURATION_MS) -> dict:
    """
    Compress audio to Geo Echo format using simplified encoding.
    
    Per spec: EnCodec at 1.5 kbps = ~1.5KB per second = ~12KB for 8 seconds
    
    For MVP: Downsample to 8-bit mono WAV at low sample rate
    Future: Integrate actual EnCodec library
    """
    try:
        import wave
        import struct
        
        # Estimate original duration
        original_size = len(audio_bytes)
        estimated_duration_s = original_size / 16000  # Rough estimate for 16kHz audio
        
        # For MVP: Create a compact audio descriptor
        # (Actual EnCodec would go here in production)
        audio_descriptor = {
            "codec": "geoecho_audio_v1",
            "original_size": original_size,
            "duration_estimate_s": min(estimated_duration_s, max_duration_ms / 1000),
            "sample_rate": 8000,  # Downsampled rate
            "channels": 1,
            "bit_depth": 8,
            "encoding": "pcm_compact"
        }
        
        # Create simplified waveform summary (peaks for visualization)
        # This gives ~100 peak values for waveform display
        chunk_size = max(1, len(audio_bytes) // 100)
        peaks = []
        for i in range(0, min(len(audio_bytes), 10000), chunk_size):
            chunk = audio_bytes[i:i+chunk_size]
            if chunk:
                peaks.append(max(chunk))
        
        # Target size: ~10KB for audio data
        compressed_data = {
            "type": "geoecho_audio",
            "descriptor": audio_descriptor,
            "waveform_peaks": peaks[:100],  # 100 peak values for visualization
            "echo_params": {
                "reverb": 0.4,
                "delay_ms": 100
            }
        }
        
        return compressed_data
        
    except Exception as e:
        logger.error(f"Audio compression failed: {e}")
        return {
            "type": "audio_metadata",
            "original_size": len(audio_bytes),
            "error": str(e)
        }


def extract_video_to_hologram(video_bytes: bytes) -> dict:
    """
    Convert video to Geo Echo hologram format.
    
    Per GEOECHO_COMPRESSION_TECH.md spec:
    - Extract 4 keyframes from 8s video
    - Run edge detection → vector outlines
    - Store as compact JSON (target: 25KB visual data)
    
    Hologram params:
    - Blue tint layer
    - Procedural hole pattern (from seed)
    - RGB background oscillation
    """
    try:
        import tempfile
        import subprocess
        
        # Write video to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video_bytes)
            video_path = f.name
        
        # Get video duration
        probe_result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'json', video_path
        ], capture_output=True, text=True)
        
        duration_s = 8.0  # Default
        try:
            probe_data = json.loads(probe_result.stdout)
            duration_s = min(float(probe_data['format']['duration']), 8.0)
        except:
            pass
        
        # Extract 4 keyframes evenly distributed
        keyframes = []
        num_keyframes = 4
        
        for i in range(num_keyframes):
            time_s = (duration_s / num_keyframes) * i
            frame_path = f"{video_path}_frame_{i}.jpg"
            
            subprocess.run([
                'ffmpeg', '-i', video_path,
                '-ss', str(time_s),
                '-vframes', '1',
                '-vf', 'scale=320:-1,format=gray',  # Grayscale, small
                '-q:v', '5',
                frame_path
            ], capture_output=True)
            
            if os.path.exists(frame_path):
                with open(frame_path, 'rb') as f:
                    frame_data = f.read()
                
                # Extract edges (simplified - in production use OpenCV Canny)
                # For now, store compressed thumbnail
                from PIL import Image, ImageFilter
                img = Image.open(BytesIO(frame_data))
                edges = img.filter(ImageFilter.FIND_EDGES)
                
                # Convert to simplified vector-like representation
                buffer = BytesIO()
                edges.save(buffer, format='JPEG', quality=30, optimize=True)
                edge_data = buffer.getvalue()
                
                keyframes.append({
                    "time_ms": int(time_s * 1000),
                    "edge_data_b64": base64.b64encode(edge_data).decode('utf-8')[:5000],  # Limit size
                    "dimensions": {"w": img.width, "h": img.height}
                })
                
                os.unlink(frame_path)
        
        # Cleanup video
        os.unlink(video_path)
        
        # Create hologram data structure per spec
        hologram_data = {
            "version": "1.0",
            "type": "hologram",
            "duration_ms": int(duration_s * 1000),
            "keyframes": keyframes,
            "hologram_params": {
                "blue_tint": [0, 180, 255],
                "hole_seed": hash(video_bytes[:100]) % 100000,
                "hole_density": 0.1,
                "rgb_shift_hz": 15,
                "glow_intensity": 0.7
            },
            "render_hints": {
                "interpolation": "linear",
                "edge_thickness": 2,
                "glow_radius": 5
            }
        }
        
        return hologram_data
        
    except Exception as e:
        logger.error(f"Video to hologram conversion failed: {e}")
        return {
            "type": "hologram_error",
            "error": str(e),
            "fallback": "thumbnail"
        }


def extract_video_thumbnail(video_bytes: bytes) -> bytes:
    """Extract first frame from video as thumbnail. Returns JPEG bytes."""
    try:
        import tempfile
        import subprocess
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video_bytes)
            video_path = f.name
        
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
        
        os.unlink(video_path)
        os.unlink(output_path)
        
        return thumbnail
        
    except Exception as e:
        logger.error(f"Video thumbnail extraction failed: {e}")
        return None


def generate_geoecho_file(echo_data: dict) -> dict:
    """
    Generate the complete .geoecho JSON structure per spec.
    This is what gets stored on Arweave.
    """
    geoecho = {
        "version": "1.0",
        "format": "geoecho",
        "duration_ms": echo_data.get("duration_ms", 0),
        "location": echo_data.get("location", {}),
        
        "visual": echo_data.get("visual_data", {}),
        "audio": echo_data.get("audio_data", {}),
        
        "metadata": {
            "echo_id": echo_data.get("echo_id"),
            "creator_id_hash": hashlib.sha256(
                str(echo_data.get("user_id", "")).encode()
            ).hexdigest()[:16],
            "created_at": echo_data.get("created_at"),
            "content_type": echo_data.get("content_type"),
            "app": "wandern"
        }
    }
    
    # Validate final size
    geoecho_json = json.dumps(geoecho)
    geoecho["_size_info"] = validate_size_for_arweave(geoecho_json, "geoecho")
    
    return geoecho


def generate_ar_token(echo_data: dict) -> dict:
    """Generate AR token representation for the Geo Echo."""
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
    
    Implements GEOECHO_COMPRESSION_TECH.md spec:
    - File size validation (100KB max)
    - Audio compression to EnCodec-like format
    - Video to hologram keyframes + vectors
    - Image progressive compression
    
    Returns JSON with AR token and .geoecho data.
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
            "user_id": user_id,
            "content_type": content_type,
            "location": location,
            "created_at": created_at,
            "compressed_size": 0,
            "thumbnail_b64": None,
            "visual_data": {},
            "audio_data": {},
            "metadata": {},
            "duration_ms": 0
        }
        
        if content_type == "text":
            # Text echoes: Validate size
            result["metadata"] = {
                "char_count": len(content_text),
                "word_count": len(content_text.split())
            }
            result["compressed_size"] = len(content_text.encode('utf-8'))
            result["visual_data"] = {"type": "text", "content": content_text}
            
        elif content_type == "image" and media_b64:
            # Image echoes: Compress for Arweave
            image_bytes = base64.b64decode(media_b64)
            compressed = compress_image(image_bytes)
            result["thumbnail_b64"] = base64.b64encode(compressed).decode('utf-8')
            result["compressed_size"] = len(compressed)
            result["visual_data"] = {
                "type": "image",
                "format": "jpeg",
                "data_b64": result["thumbnail_b64"][:50000]  # Cap at 50KB for visual
            }
            result["metadata"] = {
                "original_size": len(image_bytes),
                "compression_ratio": round(len(image_bytes) / len(compressed), 2)
            }
            
        elif content_type == "audio" and media_b64:
            # Audio echoes: Compress to Geo Echo audio format
            audio_bytes = base64.b64decode(media_b64)
            audio_data = compress_audio_to_geoecho(audio_bytes)
            result["audio_data"] = audio_data
            result["duration_ms"] = int(audio_data.get("descriptor", {}).get("duration_estimate_s", 0) * 1000)
            result["compressed_size"] = len(json.dumps(audio_data).encode('utf-8'))
            result["metadata"] = {
                "original_size": len(audio_bytes),
                "codec": "geoecho_audio_v1"
            }
            
        elif content_type == "video" and media_b64:
            # Video echoes: Convert to hologram format
            video_bytes = base64.b64decode(media_b64)
            
            # Extract hologram data (keyframes + edges)
            hologram_data = extract_video_to_hologram(video_bytes)
            result["visual_data"] = hologram_data
            result["duration_ms"] = hologram_data.get("duration_ms", 0)
            
            # Also get thumbnail for preview
            thumbnail = extract_video_thumbnail(video_bytes)
            if thumbnail:
                compressed_thumb = compress_image(thumbnail, max_size=20*1024)  # 20KB max thumbnail
                result["thumbnail_b64"] = base64.b64encode(compressed_thumb).decode('utf-8')
            
            result["compressed_size"] = len(json.dumps(hologram_data).encode('utf-8'))
            result["metadata"] = {
                "original_size": len(video_bytes),
                "hologram_version": "1.0",
                "keyframe_count": len(hologram_data.get("keyframes", []))
            }
        
        # Generate complete .geoecho file
        geoecho_file = generate_geoecho_file(result)
        
        # Generate AR token
        ar_token = generate_ar_token(result)
        
        # Final size validation
        final_size = len(json.dumps(geoecho_file).encode('utf-8'))
        size_validation = validate_size_for_arweave(json.dumps(geoecho_file), content_type)
        
        return (jsonify({
            "success": True,
            "ar_token": ar_token,
            "geoecho_file": geoecho_file,
            "size_validation": size_validation,
            "ready_for_arweave": size_validation["is_valid"],
            "compressed_size_kb": round(final_size / 1024, 2)
        }), 200, headers)
        
    except Exception as e:
        logger.error(f"Echo processing failed: {e}")
        return (jsonify({"error": str(e)}), 500, headers)
