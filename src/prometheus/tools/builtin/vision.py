"""VisionTool — describe images via multimodal LLM or preprocessor.

Donor pattern: NousResearch/hermes-agent tools/vision_tools.py.
Adapted for Prometheus: uses ModelProvider, BaseTool interface, base64 encoding.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)

# Magic bytes for image MIME detection
_MAGIC_BYTES: list[tuple[bytes, str]] = [
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
    (b"RIFF", "image/webp"),  # RIFF....WEBP
    (b"BM", "image/bmp"),
]


def _detect_mime(data: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    for magic, mime in _MAGIC_BYTES:
        if data[:len(magic)] == magic:
            # RIFF could be WAV or WEBP — check for WEBP marker
            if magic == b"RIFF" and data[8:12] != b"WEBP":
                continue
            return mime
    return "image/jpeg"  # default fallback


def _image_to_base64_data_url(path: str) -> str:
    """Read image file and return a data: URL."""
    data = Path(path).read_bytes()
    mime = _detect_mime(data)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


class VisionInput(BaseModel):
    image_path: str = Field(description="Local file path to the image")
    question: str = Field(
        default="Describe this image in detail.",
        description="Question or prompt about the image",
    )


class VisionTool(BaseTool):
    """Analyze an image and return a text description.

    Routes through the model provider. If the model supports multimodal input,
    the image is sent as a base64 content block. Otherwise returns an error
    suggesting a multimodal model.
    """

    name = "vision_analyze"
    description = (
        "Analyze an image file and answer questions about it. "
        "Accepts a local file path to the image."
    )
    input_model = VisionInput

    async def execute(self, arguments: VisionInput, context: ToolExecutionContext) -> ToolResult:
        path = Path(arguments.image_path)
        if not path.is_file():
            return ToolResult(output=f"Image not found: {arguments.image_path}", is_error=True)

        try:
            data_url = _image_to_base64_data_url(str(path))
        except Exception as exc:
            return ToolResult(output=f"Failed to read image: {exc}", is_error=True)

        # Build multimodal message for the provider
        # The provider must support OpenAI-style multimodal content blocks
        provider = context.metadata.get("provider")
        if provider is None:
            return ToolResult(
                output=f"Image at {path.name}: {path.stat().st_size} bytes. "
                f"No provider available for vision analysis.",
                is_error=True,
            )

        try:
            from prometheus.providers.base import ApiMessageRequest, ApiMessageCompleteEvent

            messages_payload = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": arguments.question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }]

            request = ApiMessageRequest(
                model="",  # use default
                messages=messages_payload,
                system_prompt="You are analyzing an image. Be detailed and accurate.",
                max_tokens=2000,
            )

            text_parts: list[str] = []
            async for event in provider.stream_message(request):
                if isinstance(event, ApiMessageCompleteEvent):
                    if event.message.text:
                        text_parts.append(event.message.text)
                elif hasattr(event, "text"):
                    text_parts.append(event.text)

            description = "".join(text_parts).strip()
            if not description:
                return ToolResult(
                    output=f"Image received ({path.name}, {path.stat().st_size} bytes) "
                    f"but model returned no description. The model may not support vision.",
                    is_error=False,
                )
            return ToolResult(output=description)

        except Exception as exc:
            logger.warning("Vision analysis failed: %s", exc)
            return ToolResult(
                output=f"Image at {path.name} ({path.stat().st_size} bytes). "
                f"Vision analysis unavailable: {exc}",
                is_error=True,
            )

    def is_read_only(self, arguments: BaseModel) -> bool:
        return True
