from vproc.config import Endpoint
from vproc.llm import client

OCR_PROMPT = (
    "You are an OCR transcription engine for screen-share frames. Transcribe ONLY the text "
    "rendered in the PRIMARY content area — the document, slide, code, spreadsheet, diagram, "
    "or window being presented. IGNORE application UI chrome: window title bars, menu bars "
    "(File/Edit/View/Window/Help), toolbars, tab strips, navigation sidebars, status bars, "
    "the OS taskbar/dock, and meeting-app controls (mute/camera/share/leave buttons, "
    "participant rosters, chat and reaction panels, timers). Read in natural order "
    "(top-to-bottom, left-to-right). Do NOT infer, complete, translate, or correct spelling. "
    "If the primary content is unreadable, or the frame shows only the meeting app with no "
    "shared content, output the single token [no shared content]. Output only the literal "
    "content text, with no commentary."
)


def ocr_frame(ocr: Endpoint, image_path: str, chat=client.ocr_image) -> str:
    text = chat(ocr.base_url, ocr.model, image_path, OCR_PROMPT).strip()
    # Normalize the "no shared content" sentinel to empty so it isn't indexed as citable
    # evidence (build_segments skips empty on_screen_text).
    if text.strip("\"' \t").lower() == "[no shared content]":
        return ""
    return text
