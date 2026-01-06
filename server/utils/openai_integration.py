"""OpenAI Realtime integration utilities.

Provides functions to start OpenAI Realtime client, manage transcripts,
and convert audio formats.

Environment Variables:
    OPENAI_API_KEY: Required. OpenAI API key for authentication.
    OPENAI_REALTIME_MODEL: Optional. Model name (default: gpt-4o-realtime-preview)
    OPENAI_REALTIME_URL: Optional. Override WebSocket URL (default: wss://api.openai.com/v1/realtime)
"""

import os
from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger

from services.openai_realtime import OpenAIRealtimeClient

# In-memory transcript buffer
_transcripts: list[dict[str, Any]] = []

# Registered transcript listeners
_transcript_listeners: list[Callable[[dict[str, Any]], None]] = []


async def start_openai_client(
    on_message: Callable[[dict[str, Any]], None] | None = None,
) -> OpenAIRealtimeClient:
    """Start and connect OpenAI Realtime client.

    Constructs client using environment variables and establishes connection.

    Args:
        on_message: Optional callback for processing incoming messages

    Returns:
        Connected OpenAIRealtimeClient instance

    Raises:
        ValueError: If OPENAI_API_KEY is not set
        Exception: If connection fails

    Environment Variables:
        OPENAI_API_KEY: Required OpenAI API key
        OPENAI_REALTIME_MODEL: Optional model name (default: gpt-4o-realtime-preview)
        OPENAI_REALTIME_URL: Optional WebSocket URL override
    """
    # Get configuration from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
    url_override = os.getenv("OPENAI_REALTIME_URL")

    logger.info(f"Starting OpenAI Realtime client with model: {model}")

    # Create client
    client = OpenAIRealtimeClient(
        api_key=api_key,
        model=model,
        on_message=on_message,
    )

    # Override URL if specified
    if url_override:
        client.url = url_override
        logger.info(f"Using custom OpenAI Realtime URL: {url_override}")

    # Connect to API
    await client.connect()

    # Wait for connection to be established
    connected = await client.wait_connected(timeout=10.0)
    if not connected:
        raise Exception("Failed to establish connection to OpenAI Realtime API")

    return client


def transcripts_broadcast(transcript_obj: dict[str, Any]) -> None:
    """Broadcast transcript to buffer and listeners.

    Appends transcript to in-memory buffer and logs it.
    Invokes all registered listener callbacks.

    Args:
        transcript_obj: Transcript object to broadcast
    """
    # Append to buffer
    _transcripts.append(transcript_obj)

    # Log transcript
    text = transcript_obj.get("text", "")
    msg_type = transcript_obj.get("type", "unknown")
    logger.info(f"Transcript [{msg_type}]: {text}")

    # Notify listeners
    for listener in _transcript_listeners:
        try:
            listener(transcript_obj)
        except Exception as e:
            logger.error(f"Error in transcript listener: {e}")


def register_transcript_listener(callback: Callable[[dict[str, Any]], None]) -> None:
    """Register a callback to be invoked when transcripts are received.

    Args:
        callback: Function to call with transcript objects
    """
    _transcript_listeners.append(callback)
    logger.debug(f"Registered transcript listener: {callback.__name__}")


def get_transcripts() -> list[dict[str, Any]]:
    """Get all transcripts from the buffer.

    Returns:
        List of transcript objects
    """
    return _transcripts.copy()


def clear_transcripts() -> None:
    """Clear the transcript buffer."""
    _transcripts.clear()
    logger.debug("Cleared transcript buffer")


def convert_float32_to_pcm16_bytes(float_array: np.ndarray) -> bytes:
    """Convert float32 audio samples to PCM16 bytes.

    Converts floating-point audio samples in range [-1.0, 1.0] to
    16-bit signed integer PCM format.

    Args:
        float_array: NumPy array of float32 samples, shape (n_samples,) or (n_samples, n_channels)
                    Values should be in range [-1.0, 1.0]

    Returns:
        Raw bytes of int16 PCM audio data

    Example:
        >>> samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        >>> pcm_bytes = convert_float32_to_pcm16_bytes(samples)
        >>> # Result: 10 bytes (5 samples * 2 bytes per int16)

    Note:
        - Clips values outside [-1.0, 1.0] range to prevent overflow
        - Maintains original channel layout (mono/stereo/etc)
        - Output is little-endian int16 (standard PCM16 format)
    """
    # Ensure input is numpy array
    if not isinstance(float_array, np.ndarray):
        float_array = np.array(float_array, dtype=np.float32)

    # Clip to valid range [-1.0, 1.0]
    float_array = np.clip(float_array, -1.0, 1.0)

    # Convert to int16 range [-32768, 32767]
    int16_array = (float_array * 32767).astype(np.int16)

    # Convert to bytes
    return int16_array.tobytes()
