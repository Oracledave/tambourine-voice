"""OpenAI Realtime API client for streaming audio transcription.

This module provides a WebSocket client for OpenAI's Realtime API, enabling
real-time audio streaming and incremental transcript handling.
"""

import asyncio
import contextlib
import json
from collections.abc import Callable
from typing import Any

import websockets
from loguru import logger


class OpenAIRealtimeClient:
    """Client for OpenAI Realtime API WebSocket connection.

    Handles streaming PCM16 audio frames to OpenAI Realtime API and
    receives incremental transcripts via WebSocket messages.

    Attributes:
        api_key: OpenAI API key for authentication
        model: Model name (e.g., 'gpt-4o-realtime-preview')
        on_message: Callback function invoked with parsed JSON message objects
        url: WebSocket URL for OpenAI Realtime API
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-realtime-preview",
        on_message: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize OpenAI Realtime client.

        Args:
            api_key: OpenAI API key
            model: Model identifier for realtime API
            on_message: Callback function called with each parsed JSON message
        """
        self.api_key = api_key
        self.model = model
        self.on_message = on_message
        self.url = "wss://api.openai.com/v1/realtime"

        self._websocket: websockets.WebSocketClientProtocol | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._connected_event = asyncio.Event()
        self._should_stop = False

    async def connect(self) -> None:
        """Establish WebSocket connection to OpenAI Realtime API.

        Opens a WebSocket connection with model query parameter and starts
        a background task to receive messages.

        Raises:
            Exception: If connection fails
        """
        try:
            # Build WebSocket URL with model query parameter
            url_with_params = f"{self.url}?model={self.model}"

            logger.info(f"Connecting to OpenAI Realtime API: {url_with_params}")

            # Connect with API key in headers
            self._websocket = await websockets.connect(
                url_with_params,
                additional_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1",
                },
            )

            logger.success("Connected to OpenAI Realtime API")
            self._connected_event.set()

            # Start background task to receive messages
            self._receive_task = asyncio.create_task(self._receive_messages())

        except Exception as e:
            logger.error(f"Failed to connect to OpenAI Realtime API: {e}")
            raise

    async def _receive_messages(self) -> None:
        """Background task to receive and process WebSocket messages.

        Reads textual JSON messages from the WebSocket and invokes the
        on_message callback with parsed objects.
        """
        if not self._websocket:
            return

        try:
            async for message in self._websocket:
                if self._should_stop:
                    break

                try:
                    # Parse JSON message
                    if isinstance(message, str):
                        data = json.loads(message)
                        logger.debug(f"Received message from OpenAI: {data.get('type', 'unknown')}")

                        # Invoke callback if provided
                        if self.on_message:
                            self.on_message(data)
                    else:
                        logger.warning(f"Received non-text message: {type(message)}")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("OpenAI Realtime WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")

    async def send_audio_frame(self, pcm16_bytes: bytes) -> None:
        """Send PCM16 audio frame to OpenAI Realtime API.

        Args:
            pcm16_bytes: Raw PCM16 audio data as bytes

        Note:
            Currently sends raw binary PCM16 frames.

            TODO: If the API expects JSON+base64 framing, modify this method to:
            1. Encode pcm16_bytes to base64 string
            2. Wrap in JSON message: {"type": "input_audio_buffer.append",
                                       "audio": base64_string}
            3. Send as text message instead of binary
        """
        if not self._websocket or not self._connected_event.is_set():
            logger.warning("Cannot send audio frame: not connected")
            return

        try:
            # Send raw binary PCM16 frame
            await self._websocket.send(pcm16_bytes)

        except Exception as e:
            logger.error(f"Failed to send audio frame: {e}")

    async def close(self) -> None:
        """Close the WebSocket connection and cleanup resources."""
        self._should_stop = True

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task

        if self._websocket:
            try:
                await self._websocket.close()
                logger.info("Closed OpenAI Realtime connection")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self._websocket = None

        self._connected_event.clear()

    async def wait_connected(self, timeout: float = 10.0) -> bool:
        """Wait for connection to be established.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if connected within timeout, False otherwise
        """
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            logger.error(f"Connection timeout after {timeout}s")
            return False
