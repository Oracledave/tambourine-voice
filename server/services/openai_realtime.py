"""OpenAI Realtime API WebSocket client for streaming audio transcription.

This module provides an async WebSocket client that connects to OpenAI's Realtime API
to send audio frames and receive transcription events. The client handles:
- WebSocket connection with authentication
- Sending raw PCM16 audio frames (signed 16-bit, mono, 24kHz by default)
- Receiving and parsing JSON messages from the API
- Background message receiving loop

Audio Format Expectations:
- Sample rate: 24000 Hz (configurable, but OpenAI Realtime expects 24kHz)
- Channels: 1 (mono)
- Format: Signed 16-bit PCM (int16)
- Endianness: Little-endian

To switch models or formats:
- Change the model in OPENAI_REALTIME_MODEL env var or model parameter
- Adjust sample_rate parameter if model requires different rate
- If API requires JSON+base64 format instead of raw binary, uncomment the
  JSON encoding in send_audio_frame() and adjust message format accordingly
"""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import websockets
from loguru import logger

DEFAULT_REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-4o-realtime-preview-2024-12-17"
DEFAULT_SAMPLE_RATE = 24000


class OpenAIRealtimeClient:
    """Async WebSocket client for OpenAI Realtime API.

    This client manages a persistent WebSocket connection to stream audio frames
    to OpenAI's Realtime API and receive transcription events.

    Args:
        api_key: OpenAI API key for authentication
        model: Model identifier (default: gpt-4o-realtime-preview-2024-12-17)
        base_url: Base WebSocket URL (default: wss://api.openai.com/v1/realtime)
        on_message: Callback function called with parsed JSON for each message
        sample_rate: Expected audio sample rate in Hz (default: 24000)
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_REALTIME_URL,
        on_message: Callable[[dict[str, Any]], None] | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        """Initialize the OpenAI Realtime client."""
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.on_message = on_message
        self.sample_rate = sample_rate

        self._websocket: websockets.WebSocketClientProtocol | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._is_connected = False

    async def connect(self) -> None:
        """Connect to the OpenAI Realtime API WebSocket.

        Establishes WebSocket connection with authentication headers and
        starts the background message receiving loop.

        Raises:
            Exception: If connection fails
        """
        # Build URL with model parameter
        url = f"{self.base_url}?model={self.model}"

        # Set up authentication headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            logger.info(f"Connecting to OpenAI Realtime API: {self.base_url}")
            self._websocket = await websockets.connect(url, extra_headers=headers)
            self._is_connected = True
            logger.success(f"Connected to OpenAI Realtime API with model: {self.model}")

            # TODO: If the API requires an initial session.update message to configure
            # the session (e.g., to set modalities, instructions, or audio format),
            # send it here. Example:
            # await self._send_json({
            #     "type": "session.update",
            #     "session": {
            #         "modalities": ["text", "audio"],
            #         "instructions": "...",
            #         "input_audio_format": "pcm16",
            #         "input_audio_transcription": {"model": "whisper-1"}
            #     }
            # })

            # Start background task to receive messages
            self._recv_task = asyncio.create_task(self._recv_loop())

        except Exception as e:
            self._is_connected = False
            logger.error(f"Failed to connect to OpenAI Realtime API: {e}")
            raise

    async def send_audio_frame(self, pcm16_bytes: bytes) -> None:
        """Send a raw PCM16 audio frame to the OpenAI Realtime API.

        Sends audio data as raw binary frames. The audio must be:
        - Signed 16-bit PCM (int16)
        - Mono channel
        - Sample rate matching self.sample_rate (default 24kHz)
        - Little-endian byte order

        Args:
            pcm16_bytes: Raw PCM16 audio data as bytes

        Note:
            If the API requires JSON-wrapped base64 audio instead of raw binary,
            uncomment the JSON encoding section below and comment out the raw send.
        """
        if not self._is_connected or self._websocket is None:
            logger.warning("Cannot send audio frame: not connected to OpenAI Realtime API")
            return

        try:
            # Send raw binary PCM16 data
            # This is the default for OpenAI Realtime API
            await self._websocket.send(pcm16_bytes)

            # TODO: If API requires JSON+base64 format, uncomment this and comment out above:
            # audio_base64 = base64.b64encode(pcm16_bytes).decode("utf-8")
            # await self._send_json({
            #     "type": "input_audio_buffer.append",
            #     "audio": audio_base64
            # })

        except Exception as e:
            logger.error(f"Failed to send audio frame: {e}")

    async def close(self) -> None:
        """Close the WebSocket connection and stop the receiving loop.

        Cleanly shuts down the connection and cancels background tasks.
        """
        logger.info("Closing OpenAI Realtime API connection")
        self._is_connected = False

        # Cancel recv task
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            # Allow task cancellation without exception handling

        # Close websocket
        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        logger.info("OpenAI Realtime API connection closed")

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected to the API."""
        return self._is_connected

    async def _send_json(self, data: dict[str, Any]) -> None:
        """Send a JSON message to the WebSocket.

        Args:
            data: Dictionary to serialize and send as JSON
        """
        if self._websocket:
            await self._websocket.send(json.dumps(data))

    async def _recv_loop(self) -> None:
        """Background task that receives and processes messages from the API.

        Continuously reads messages from the WebSocket, parses JSON, and
        calls the on_message callback for each message received.
        """
        logger.info("Starting OpenAI Realtime message receive loop")

        try:
            while self._is_connected and self._websocket:
                try:
                    message = await self._websocket.recv()

                    # Parse JSON message
                    if isinstance(message, str):
                        try:
                            data = json.loads(message)
                            logger.debug(f"Received message type: {data.get('type', 'unknown')}")

                            # Call callback with parsed message
                            if self.on_message:
                                self.on_message(data)

                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON message: {e}")
                            logger.debug(f"Raw message: {message[:200]}")

                    elif isinstance(message, bytes):
                        # Handle binary messages if needed
                        logger.debug(f"Received binary message: {len(message)} bytes")

                except websockets.exceptions.ConnectionClosed:
                    logger.warning("OpenAI Realtime WebSocket connection closed")
                    self._is_connected = False
                    break

        except asyncio.CancelledError:
            logger.info("OpenAI Realtime receive loop cancelled")
        except Exception as e:
            logger.error(f"Error in OpenAI Realtime receive loop: {e}")
            self._is_connected = False
        finally:
            logger.info("OpenAI Realtime receive loop ended")
