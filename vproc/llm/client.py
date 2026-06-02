import base64
import mimetypes

from openai import OpenAI


def _client(base_url: str) -> OpenAI:
    # api_key is required by the SDK but unused by local servers
    return OpenAI(base_url=base_url, api_key="not-needed")


def embed_texts(base_url: str, model: str, texts: list[str]) -> list[list[float]]:
    resp = _client(base_url).embeddings.create(model=model, input=texts)
    return [list(d.embedding) for d in resp.data]


def chat_json(base_url: str, model: str, system: str, user: str, temperature: float = 0.1) -> str:
    resp = _client(base_url).chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


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
