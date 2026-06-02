from vproc.config import Endpoint
from vproc.llm import client

OCR_PROMPT = (
    "You are an OCR transcription engine. Transcribe ONLY the text that is visibly "
    "rendered as characters in the image, in natural reading order (top-to-bottom, "
    "left-to-right). Do NOT infer, complete, translate, or correct spelling. If a region "
    "is unreadable, output the token [illegible] for it. Output only the literal on-screen "
    "text, with no commentary."
)


def ocr_frame(ocr: Endpoint, image_path: str, chat=client.ocr_image) -> str:
    return chat(ocr.base_url, ocr.model, image_path, OCR_PROMPT).strip()
