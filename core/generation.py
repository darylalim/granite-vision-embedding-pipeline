import base64
import io

from PIL import Image


def encode_image(image: Image.Image) -> str:
    """Encode a PIL Image as a base64 PNG string."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_messages(query: str, images: list[Image.Image]) -> list[dict]:
    """Build OpenAI-compatible messages for a VLM request.

    Returns a system message with grounding instructions and a user message
    containing the query text and page images as base64 data URIs.
    """
    system_message = {
        "role": "system",
        "content": (
            "You are answering questions about documents. You are given page "
            "images from the most relevant pages. Answer the question using "
            "only information visible in the provided pages. Cite which "
            "page(s) support your answer. If the pages do not contain enough "
            "information to answer, say so."
        ),
    }

    user_content: list[dict] = [{"type": "text", "text": query}]
    for image in images:
        b64 = encode_image(image)
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    user_message = {"role": "user", "content": user_content}
    return [system_message, user_message]
