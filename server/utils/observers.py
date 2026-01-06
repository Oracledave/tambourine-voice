"""Custom logging observer for pipeline events.

Filters frames by source to avoid duplicate logs as frames propagate through the pipeline.
Forwards audio frames to OpenAI Realtime API when available.
"""

import asyncio

import numpy as np
from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    UserSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport

from utils.logger import logger
from utils.openai_integration import convert_float32_to_pcm16_bytes, get_openai_client


class PipelineLogObserver(BaseObserver):
    """Observer that logs key pipeline events at INFO level.

    Uses source filtering to log each event only once:
    - StartFrame: logged when reaching output transport (end of pipeline)
    - Audio/Speech frames: logged when from input transport (origin)
    - Transcription: logged when from STT service (origin)
    - LLM response: logged when from LLM service (origin)
    - RTVI messages: logged when from output transport (being sent)

    Logs at DEBUG level:
    - Other frames (excluding noisy UserSpeakingFrame and MetricsFrame)
    """

    def __init__(self) -> None:
        """Initialize the observer."""
        super().__init__()
        self._llm_accumulator: str = ""
        self._is_accumulating: bool = False
        self._audio_frame_count: int = 0
        # Track speaking state to deduplicate speech events from multiple sources
        self._is_speaking: bool = False

    async def on_push_frame(self, data: FramePushed) -> None:
        """Handle frame push events and log key pipeline activities.

        Args:
            data: The frame push event data containing source, frame, and other info.
        """
        src = data.source
        frame = data.frame

        # Log pipeline start when it reaches the output transport (end of pipeline)
        if isinstance(frame, StartFrame) and isinstance(src, BaseOutputTransport):
            logger.success("Pipeline started")

        # Log audio frames from input transport (first few and periodic)
        elif isinstance(frame, InputAudioRawFrame) and isinstance(src, BaseInputTransport):
            self._audio_frame_count += 1
            if self._audio_frame_count % 500 == 0:
                logger.info(
                    f"Audio frame #{self._audio_frame_count}: "
                    f"{len(frame.audio)} bytes, {frame.sample_rate}Hz, {frame.num_channels}ch"
                )

            # Forward audio frame to OpenAI Realtime API if available
            # Note: frame.audio is the raw audio bytes from the input transport
            # InputAudioRawFrame typically contains PCM audio data
            await self._forward_audio_to_openai(frame)

        # Log transcription from STT service
        elif isinstance(frame, TranscriptionFrame) and isinstance(src, STTService):
            logger.info(f"TRANSCRIPTION: '{frame.text}'")

        # Log speech start/stop from input transport (where VAD runs)
        # Use state tracking to deduplicate - same event may come from multiple sources
        elif isinstance(frame, UserStartedSpeakingFrame) and isinstance(src, BaseInputTransport):
            if not self._is_speaking:
                self._is_speaking = True
                logger.info("Speech started")
        elif isinstance(frame, UserStoppedSpeakingFrame) and isinstance(src, BaseInputTransport):
            if self._is_speaking:
                self._is_speaking = False
                logger.info("Speech stopped")

        # Accumulate and log LLM response from LLM service
        # Use LLMTextFrame (not TextFrame) - this is what LLM services output
        elif isinstance(frame, LLMFullResponseStartFrame) and isinstance(src, LLMService):
            self._llm_accumulator = ""
            self._is_accumulating = True
        elif (
            isinstance(frame, LLMTextFrame)
            and self._is_accumulating
            and isinstance(src, LLMService)
        ):
            self._llm_accumulator += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame) and isinstance(src, LLMService):
            self._is_accumulating = False
            if self._llm_accumulator.strip():
                logger.info(f"Cleaned text: '{self._llm_accumulator.strip()}'")
            self._llm_accumulator = ""

        # Log RTVI server messages when sent from output transport
        elif isinstance(frame, RTVIServerMessageFrame) and isinstance(src, BaseOutputTransport):
            logger.info(f"Sending to client: {frame.data}")

        # Log other frames at debug level (skip noisy ones)
        elif not isinstance(frame, (UserSpeakingFrame, MetricsFrame, TextFrame, LLMTextFrame)):
            logger.debug(f"Frame: {type(frame).__name__}")

    async def _forward_audio_to_openai(self, frame: InputAudioRawFrame) -> None:
        """Forward audio frame to OpenAI Realtime API for streaming transcription.

        This method converts the audio frame to the required format (signed 16-bit PCM)
        and sends it asynchronously to the OpenAI Realtime client if connected.

        Args:
            frame: The InputAudioRawFrame containing audio data

        Note:
            - frame.audio: Raw audio bytes (typically already PCM16 from WebRTC)
            - frame.sample_rate: Sample rate in Hz (typically 16000 or 24000)
            - frame.num_channels: Number of audio channels (should be 1 for mono)
            - The conversion assumes frame.audio is already int16 PCM from the transport
            - If frame format changes, update the conversion logic here
        """
        # Get the OpenAI client (may be None if not configured)
        openai_client = get_openai_client()

        if openai_client is None or not openai_client.is_connected:
            # Don't log every frame - only log periodically to avoid spam
            if self._audio_frame_count % 1000 == 0:
                logger.debug("OpenAI Realtime client not connected, skipping audio forwarding")
            return

        try:
            # Extract audio data from frame
            # InputAudioRawFrame.audio contains the raw PCM bytes
            audio_bytes = frame.audio

            # Check if we need to convert the audio format
            # Most WebRTC audio is already 16-bit PCM, but we may receive float32
            # For now, we assume the audio is already in the correct format
            # If conversion is needed, it would happen here using convert_float32_to_pcm16_bytes

            # Log sample rate mismatch warning (OpenAI expects 24kHz by default)
            expected_rate = 24000
            if frame.sample_rate != expected_rate and self._audio_frame_count % 1000 == 0:
                logger.warning(
                    f"Audio sample rate mismatch: frame={frame.sample_rate}Hz, "
                    f"OpenAI expects={expected_rate}Hz. Audio quality may be degraded. "
                    f"Consider adding resampling with resampy or librosa."
                )

            # Send audio frame asynchronously (don't await to avoid blocking the pipeline)
            # Create a background task to send the audio frame
            asyncio.create_task(openai_client.send_audio_frame(audio_bytes))

        except Exception as e:
            logger.error(f"Error forwarding audio to OpenAI Realtime: {e}")
