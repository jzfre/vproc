from vproc.config import Endpoint
from vproc.ingest.ocr import ocr_frame, OCR_PROMPT

def test_ocr_frame_calls_endpoint_with_prompt():
    calls = {}
    def fake_chat(base_url, model, image_path, prompt):
        calls.update(base_url=base_url, model=model, image_path=image_path, prompt=prompt)
        return "  Quarterly Plan  "
    ep = Endpoint("http://voyage:8000/v1", "QuantTrio/Qwen3.5-9B-AWQ")
    text = ocr_frame(ep, "/frame.png", chat=fake_chat)
    assert text == "Quarterly Plan"
    assert calls["base_url"] == "http://voyage:8000/v1"
    assert calls["image_path"] == "/frame.png"
    assert "verbatim" in OCR_PROMPT.lower() or "only" in OCR_PROMPT.lower()


def test_ocr_frame_normalizes_no_content_sentinel():
    ep = Endpoint("u", "m")
    for reply in ("[no shared content]", "  [No Shared Content]  ", '"[NO SHARED CONTENT]"',
                  "'[no shared content]'"):
        assert ocr_frame(ep, "/f.png", chat=lambda *a, r=reply: r) == ""
    # real content is preserved verbatim
    assert ocr_frame(ep, "/f.png", chat=lambda *a: "Slide: [no shared content] appears here") \
        == "Slide: [no shared content] appears here"
