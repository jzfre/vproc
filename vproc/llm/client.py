import base64
import mimetypes

from openai import OpenAI


def _client(base_url: str) -> OpenAI:
    # api_key is required by the SDK but unused by local servers
    return OpenAI(base_url=base_url, api_key="not-needed")


def embed_texts(base_url: str, model: str, texts: list[str]) -> list[list[float]]:
    resp = _client(base_url).embeddings.create(model=model, input=texts)
    return [list(d.embedding) for d in resp.data]


def chat_json(
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
    schema: dict | None = None,
) -> str:
    # Prefer constrained decoding (json_schema) when a schema is given; fall back to the
    # looser json_object mode otherwise. Newer OpenAI-compatible servers (e.g. current
    # LM Studio) reject json_object and require json_schema or text.
    if schema is not None:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }
    else:
        response_format = {"type": "json_object"}
    resp = _client(base_url).chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        response_format=response_format,
    )
    msg = resp.choices[0].message
    # Some thinking models (e.g. Qwen3 via LM Studio) emit the constrained JSON into the
    # reasoning channel and leave `content` empty; fall back to reasoning_content.
    return msg.content or getattr(msg, "reasoning_content", None) or ""


def ocr_image(base_url: str, model: str, image_path: str, prompt: str) -> str:
    data = base64.b64encode(open(image_path, "rb").read()).decode()
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    resp = _client(base_url).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
        ]}],
        temperature=0.0,
    )
    return resp.choices[0].message.content
