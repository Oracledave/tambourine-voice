"""OpenAI Realtime API WebSocket client.

This module provides a WebSocket client for OpenAI's Realtime API to stream
audio frames and receive transcription responses.
"""

import asyncio
import contextlib
import json
from collections.abc import Callable
from typing import Any

import websockets
from loguru import logger


class OpenAIRealtimeClient:
    """WebSocket client for OpenAI Realtime API.

    This client connects to OpenAI's Realtime API via WebSocket, streams
    PCM16 audio frames, and receives JSON messages (including transcripts)
    via a callback handler.

    Attributes:
        api_key: OpenAI API key for authentication
        model: Model name (e.g., 'gpt-4o-realtime-preview')
        url: WebSocket endpoint URL
        on_message: Callback function for received JSON messages
        websocket: Active WebSocket connection (None when disconnected)
        _recv_task: Background task for receiving messages
        _connected: Connection state flag
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-realtime-preview",
        url: str = "wss://api.openai.com/v1/realtime",
        on_message: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize the OpenAI Realtime client.

        Args:
            api_key: OpenAI API key for authentication
            model: Model name (default: gpt-4o-realtime-preview)
            url: WebSocket endpoint URL (default: wss://api.openai.com/v1/realtime)
            on_message: Callback function to handle received JSON messages
        """
        self.api_key = api_key
        self.model = model
        self.url = url
        self.on_message = on_message
        self.websocket: Any = None  # WebSocket connection (websockets library)
        self._recv_task: asyncio.Task[None] | None = None
        self._connected = asyncio.Event()

    async def connect(self) -> None:
        """Establish WebSocket connection to OpenAI Realtime API.

        Raises:
            Exception: If connection fails
        """
        try:
            # Connect with API key in headers and model as query parameter
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            }

            url_with_params = f"{self.url}?model={self.model}"

            logger.info(f"Connecting to OpenAI Realtime API: {url_with_params}")
            self.websocket = await websockets.connect(url_with_params, extra_headers=headers)

            logger.success("Connected to OpenAI Realtime API")
            self._connected.set()

            # Start background task to receive messages
            self._recv_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            logger.error(f"Failed to connect to OpenAI Realtime API: {e}")
            self._connected.clear()
            raise

    async def _receive_loop(self) -> None:
        """Background loop to receive and parse JSON messages from WebSocket.

        Calls the on_message callback for each received message.
        """
        if not self.websocket:
            logger.error("Cannot start receive loop: no websocket connection")
            return

        try:
            async for message in self.websocket:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        logger.debug(f"Received message: {data.get('type', 'unknown')}")

                        if self.on_message:
                            self.on_message(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON message: {e}")
                else:
                    logger.debug(f"Received binary message: {len(message)} bytes")

        except websockets.exceptions.ConnectionClosed:
            logger.info("OpenAI Realtime connection closed")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
        finally:
            self._connected.clear()

    async def send_audio_frame(self, pcm16_bytes: bytes) -> None:
        """Send a PCM16 audio frame to OpenAI Realtime API.

        Note: This implementation sends raw binary PCM16 frames by default.

        TODO: The OpenAI Realtime API may require JSON-wrapped messages instead
        of raw binary frames. If you see connection errors or missing transcripts:

        1. Change this method to wrap audio in JSON format:
           ```python
           import base64
           message = {
               "type": "input_audio_buffer.append",
               "audio": base64.b64encode(pcm16_bytes).decode('utf-8')
           }
           await self.websocket.send(json.dumps(message))
           ```

        2. You may also need to send an initial session configuration message
           after connection, before sending audio frames.

        Args:
            pcm16_bytes: Raw PCM16 audio data (16-bit little-endian samples)

        Raises:
            RuntimeError: If not connected
        """
        if not self.websocket or not self._connected.is_set():
            raise RuntimeError("Not connected to OpenAI Realtime API")

        try:
            # Send raw binary frame
            # TODO: Change to JSON+base64 if required by API
            await self.websocket.send(pcm16_bytes)

        except Exception as e:
            logger.error(f"Failed to send audio frame: {e}")
            raise

    async def wait_connected(self, timeout: float = 10.0) -> bool:
        """Wait for the client to be connected.

        Args:
            timeout: Maximum seconds to wait for connection

        Returns:
            True if connected within timeout, False otherwise
        """
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    async def close(self) -> None:
        """Close the WebSocket connection and cleanup resources."""
        logger.info("Closing OpenAI Realtime connection")

        self._connected.clear()

        # Cancel receive task
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task

        # Close websocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        logger.info("OpenAI Realtime connection closed")
