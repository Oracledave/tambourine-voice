"""OpenAI Realtime API integration utilities.

This module provides utilities for managing the OpenAI Realtime client,
handling transcripts, and audio format conversion.
"""

import os
from collections import deque
from typing import Any, Callable

import numpy as np
from loguru import logger

from services.openai_realtime import OpenAIRealtimeClient

# Global OpenAI client instance
_openai_client: OpenAIRealtimeClient | None = None

# Bounded in-memory transcript buffer (keep last 100 transcripts)
_transcripts_buffer: deque[dict[str, Any]] = deque(maxlen=100)

# Transcript listeners for future extensibility
_transcript_listeners: list[Callable[[dict[str, Any]], None]] = []


def transcripts_broadcast(transcript_obj: dict[str, Any]) -> None:
    """Log and store a transcript object.
    
    This function is called whenever a transcript event is received from
    OpenAI Realtime API. It logs the transcript and adds it to the in-memory
    buffer.
    
    Args:
        transcript_obj: The transcript object/event from OpenAI
    """
    logger.info(f"OpenAI transcript: {transcript_obj}")
    _transcripts_buffer.append(transcript_obj)
    
    # Notify any registered listeners
    for listener in _transcript_listeners:
        try:
            listener(transcript_obj)
        except Exception as e:
            logger.error(f"Error in transcript listener: {e}")


def register_transcript_listener(callback: Callable[[dict[str, Any]], None]) -> None:
    """Register a callback to receive transcript events.
    
    Args:
        callback: Function to call with each transcript object
    """
    _transcript_listeners.append(callback)


def get_transcripts() -> list[dict[str, Any]]:
    """Get all transcripts from the in-memory buffer.
    
    Returns:
        List of transcript objects
    """
    return list(_transcripts_buffer)


def convert_float32_to_pcm16_bytes(float_array: np.ndarray) -> bytes:
    """Convert float32 audio samples to PCM16 bytes.
    
    Converts normalized float32 audio samples (range -1.0 to 1.0) to
    16-bit PCM format (int16 range -32768 to 32767) as little-endian bytes.
    
    Args:
        float_array: NumPy array of float32 samples in range [-1.0, 1.0]
        
    Returns:
        bytes: PCM16 audio data as bytes (16-bit little-endian integers)
        
    Example:
        >>> samples = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
        >>> pcm_bytes = convert_float32_to_pcm16_bytes(samples)
        >>> len(pcm_bytes)
        8  # 4 samples * 2 bytes per sample
    """
    # Clamp values to [-1.0, 1.0] range
    float_array = np.clip(float_array, -1.0, 1.0)
    
    # Scale to int16 range and convert
    pcm16_array = (float_array * 32767.0).astype(np.int16)
    
    # Convert to bytes (little-endian)
    return pcm16_array.tobytes()


async def start_openai_client(
    on_message: Callable[[dict[str, Any]], None] | None = None,
) -> OpenAIRealtimeClient | None:
    """Create and connect the OpenAI Realtime client.
    
    Reads configuration from environment variables:
    - OPENAI_API_KEY (required)
    - OPENAI_REALTIME_MODEL (optional, default: gpt-4o-realtime-preview)
    - OPENAI_REALTIME_URL (optional, default: wss://api.openai.com/v1/realtime)
    
    Args:
        on_message: Callback for received messages (default: transcripts_broadcast)
        
    Returns:
        Connected OpenAI client, or None if API key is not configured
    """
    global _openai_client
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "OPENAI_API_KEY not set. OpenAI Realtime integration disabled. "
            "Set OPENAI_API_KEY environment variable to enable."
        )
        return None
    
    # Get optional configuration
    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
    url = os.getenv("OPENAI_REALTIME_URL", "wss://api.openai.com/v1/realtime")
    
    # Use transcripts_broadcast as default callback
    message_handler = on_message or transcripts_broadcast
    
    try:
        # Create and connect client
        _openai_client = OpenAIRealtimeClient(
            api_key=api_key,
            model=model,
            url=url,
            on_message=message_handler,
        )
        
        await _openai_client.connect()
        
        # Wait for connection to be established
        if not await _openai_client.wait_connected(timeout=10.0):
            logger.error("OpenAI Realtime connection timeout")
            return None
            
        logger.success(f"OpenAI Realtime client started (model: {model})")
        return _openai_client
        
    except Exception as e:
        logger.error(f"Failed to start OpenAI Realtime client: {e}")
        _openai_client = None
        return None


def get_openai_client() -> OpenAIRealtimeClient | None:
    """Get the global OpenAI Realtime client instance.
    
    Returns:
        The active client, or None if not initialized
    """
    return _openai_client
