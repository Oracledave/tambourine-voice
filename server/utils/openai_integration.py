"""OpenAI Realtime API integration utilities for Tambourine server.

This module provides helper functions to integrate the OpenAI Realtime client
with the Tambourine server pipeline. It includes:
- Startup helper to initialize and connect the OpenAI client
- Audio format conversion utilities (float32 to PCM16)
- Transcript broadcasting and listener registration for future WebSocket forwarding
- In-memory transcript history for debugging and display
"""

from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger

from services.openai_realtime import OpenAIRealtimeClient

# Module-level client instance (initialized at startup)
_openai_client: OpenAIRealtimeClient | None = None

# Transcript history and listeners
_transcript_history: deque[dict[str, Any]] = deque(maxlen=100)
_transcript_listeners: list[Callable[[dict[str, Any]], None]] = []


async def start_openai_client(
    api_key: str,
    model: str,
    base_url: str,
    on_message: Callable[[dict[str, Any]], None] | None = None,
) -> OpenAIRealtimeClient:
    """Initialize and connect the OpenAI Realtime client.

    Creates an OpenAIRealtimeClient instance and establishes the WebSocket
    connection in the current event loop. Stores the client globally for
    access by the frame handler.

    Args:
        api_key: OpenAI API key
        model: Model identifier (e.g., 'gpt-4o-realtime-preview-2024-12-17')
        base_url: WebSocket base URL (e.g., 'wss://api.openai.com/v1/realtime')
        on_message: Optional callback for incoming messages (defaults to message_handler)

    Returns:
        The connected OpenAIRealtimeClient instance

    Raises:
        Exception: If connection fails
    """
    global _openai_client

    # Use default message handler if not provided
    message_callback = on_message or _default_message_handler

    # Create and connect client
    client = OpenAIRealtimeClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        on_message=message_callback,
    )

    await client.connect()
    _openai_client = client

    logger.success("OpenAI Realtime client initialized and connected")
    return client


def get_openai_client() -> OpenAIRealtimeClient | None:
    """Get the global OpenAI Realtime client instance.

    Returns:
        The client instance if connected, None otherwise
    """
    return _openai_client


async def stop_openai_client() -> None:
    """Close the OpenAI Realtime client connection.

    Should be called during server shutdown to cleanly close the WebSocket.
    """
    global _openai_client

    if _openai_client:
        await _openai_client.close()
        _openai_client = None
        logger.info("OpenAI Realtime client stopped")


def _default_message_handler(message: dict[str, Any]) -> None:
    """Default handler for OpenAI Realtime API messages.

    Logs all messages and broadcasts transcript-related events to listeners.
    This is the callback invoked by the OpenAIRealtimeClient for each received message.

    Args:
        message: Parsed JSON message from the API
    """
    msg_type = message.get("type", "unknown")
    logger.debug(f"OpenAI Realtime message: {msg_type}")

    # Handle different message types
    if msg_type == "conversation.item.input_audio_transcription.completed":
        # This is an interim transcript from the API
        transcript = message.get("transcript", "")
        logger.info(f"OpenAI Realtime transcript: '{transcript}'")
        transcripts_broadcast({"type": "interim", "text": transcript, "raw": message})

    elif msg_type == "conversation.item.created":
        # New conversation item (could include final transcript)
        item = message.get("item", {})
        if item.get("type") == "message":
            content = item.get("content", [])
            for part in content:
                if part.get("type") == "input_audio":
                    transcript = part.get("transcript", "")
                    if transcript:
                        logger.info(f"OpenAI Realtime final transcript: '{transcript}'")
                        transcripts_broadcast({"type": "final", "text": transcript, "raw": message})

    elif msg_type == "response.audio_transcript.delta":
        # Delta updates for audio transcription
        delta = message.get("delta", "")
        if delta:
            logger.debug(f"OpenAI Realtime transcript delta: '{delta}'")
            transcripts_broadcast({"type": "delta", "text": delta, "raw": message})

    elif msg_type == "response.audio_transcript.done":
        # Final audio transcription
        transcript = message.get("transcript", "")
        logger.info(f"OpenAI Realtime final audio transcript: '{transcript}'")
        transcripts_broadcast({"type": "final", "text": transcript, "raw": message})

    elif msg_type == "error":
        # Error message from API
        error = message.get("error", {})
        error_msg = error.get("message", "Unknown error")
        logger.error(f"OpenAI Realtime API error: {error_msg}")

    else:
        # Log other message types at debug level
        logger.debug(f"OpenAI Realtime message: {message}")


def transcripts_broadcast(transcript_obj: dict[str, Any]) -> None:
    """Broadcast a transcript to registered listeners and store in history.

    This function is called when a new transcript event is received from the
    OpenAI Realtime API. It logs the transcript, stores it in memory, and
    notifies all registered listeners.

    Args:
        transcript_obj: Dictionary containing transcript data with keys:
            - type: 'interim', 'final', or 'delta'
            - text: The transcript text
            - raw: Original API message (optional)
    """
    # Add to history
    _transcript_history.append(transcript_obj)

    # Log the transcript
    transcript_type = transcript_obj.get("type", "unknown")
    transcript_text = transcript_obj.get("text", "")
    logger.info(f"Transcript ({transcript_type}): {transcript_text}")

    # Notify all listeners
    for listener in _transcript_listeners:
        try:
            listener(transcript_obj)
        except Exception:
            # Log full traceback to aid debugging but don't crash
            logger.exception("Error in transcript listener callback")


def register_transcript_listener(callback: Callable[[dict[str, Any]], None]) -> None:
    """Register a callback to be notified of new transcripts.

    This allows future features (e.g., WebSocket forwarding to clients) to
    subscribe to transcript events without modifying this module.

    Args:
        callback: Function to call with transcript_obj when new transcript arrives
    """
    _transcript_listeners.append(callback)
    logger.debug(f"Registered transcript listener: {callback.__name__}")


def get_transcript_history() -> list[dict[str, Any]]:
    """Get the recent transcript history.

    Returns:
        List of recent transcript objects (up to 100 most recent)
    """
    return list(_transcript_history)


def convert_float32_to_pcm16_bytes(
    float_array: np.ndarray,
    target_sample_rate: int = 24000,
    source_sample_rate: int | None = None,
) -> bytes:
    """Convert float32 audio samples to signed 16-bit PCM bytes.

    This function takes normalized float32 audio samples (range -1.0 to 1.0)
    and converts them to signed 16-bit PCM format suitable for the OpenAI
    Realtime API.

    Args:
        float_array: NumPy array of float32 samples (range -1.0 to 1.0)
        target_sample_rate: Expected sample rate for the API (default: 24000 Hz)
        source_sample_rate: Source sample rate if known (used for warning)

    Returns:
        Raw bytes in signed 16-bit PCM format (little-endian)

    Note:
        - This function does NOT resample audio. If source_sample_rate differs
          from target_sample_rate, a warning is logged but no resampling occurs.
        - To add resampling, integrate a library like resampy or librosa here.
        - The OpenAI Realtime API expects 24kHz mono audio by default.
    """
    # Warn if sample rates don't match (resampling needed but not implemented)
    if source_sample_rate and source_sample_rate != target_sample_rate:
        logger.warning(
            f"Audio sample rate mismatch: source={source_sample_rate}Hz, "
            f"target={target_sample_rate}Hz. Resampling is required but not implemented. "
            f"Audio quality and recognition may be degraded. "
            f"To add resampling, integrate resampy or librosa in this function."
        )

    # Clip to valid range [-1.0, 1.0]
    float_array = np.clip(float_array, -1.0, 1.0)

    # Convert to int16 range [-32768, 32767]
    int16_array = (float_array * 32767).astype(np.int16)

    # Convert to bytes (little-endian)
    return int16_array.tobytes()


def bytes_to_float32_array(pcm16_bytes: bytes) -> np.ndarray:
    """Convert signed 16-bit PCM bytes to float32 array.

    Utility function for debugging or processing received audio.

    Args:
        pcm16_bytes: Raw bytes in signed 16-bit PCM format

    Returns:
        NumPy array of float32 samples normalized to range [-1.0, 1.0]
    """
    # Convert bytes to int16 array
    int16_array = np.frombuffer(pcm16_bytes, dtype=np.int16)

    # Normalize to float32 range [-1.0, 1.0]
    return int16_array.astype(np.float32) / 32767.0
