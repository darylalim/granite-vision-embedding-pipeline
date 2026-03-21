import base64
import io

from PIL import Image

from core.generation import build_messages, encode_image


class TestEncodeImage:
    def test_returns_valid_base64_png(self) -> None:
        img = Image.new("RGB", (64, 64), color="red")
        result = encode_image(img)
        decoded = base64.b64decode(result)
        reloaded = Image.open(io.BytesIO(decoded))
        assert reloaded.format == "PNG"

    def test_converts_rgba_to_rgb(self) -> None:
        img = Image.new("RGBA", (64, 64), color=(255, 0, 0, 128))
        result = encode_image(img)
        decoded = base64.b64decode(result)
        reloaded = Image.open(io.BytesIO(decoded))
        assert reloaded.mode == "RGB"

    def test_converts_l_to_rgb(self) -> None:
        img = Image.new("L", (64, 64), color=128)
        result = encode_image(img)
        decoded = base64.b64decode(result)
        reloaded = Image.open(io.BytesIO(decoded))
        assert reloaded.mode == "RGB"


class TestBuildMessages:
    def test_contains_system_and_user_messages(self) -> None:
        img = Image.new("RGB", (64, 64))
        messages = build_messages("What is this?", [img])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_has_grounding_instructions(self) -> None:
        img = Image.new("RGB", (64, 64))
        messages = build_messages("test", [img])
        system_content = messages[0]["content"]
        assert "provided pages" in system_content.lower()

    def test_user_message_contains_query_text(self) -> None:
        img = Image.new("RGB", (64, 64))
        messages = build_messages("What color is this?", [img])
        user_content = messages[1]["content"]
        text_parts = [p for p in user_content if p["type"] == "text"]
        assert any("What color is this?" in p["text"] for p in text_parts)

    def test_user_message_contains_image_parts(self) -> None:
        images = [Image.new("RGB", (64, 64)) for _ in range(3)]
        messages = build_messages("test", images)
        user_content = messages[1]["content"]
        image_parts = [p for p in user_content if p["type"] == "image_url"]
        assert len(image_parts) == 3

    def test_image_urls_are_base64_data_uris(self) -> None:
        img = Image.new("RGB", (64, 64))
        messages = build_messages("test", [img])
        user_content = messages[1]["content"]
        image_part = [p for p in user_content if p["type"] == "image_url"][0]
        url = image_part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_empty_images_list(self) -> None:
        messages = build_messages("test query", [])
        user_content = messages[1]["content"]
        image_parts = [p for p in user_content if p["type"] == "image_url"]
        assert len(image_parts) == 0
