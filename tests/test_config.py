from vproc.config import load_config

def test_defaults(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("VPROC_"):
            monkeypatch.delenv(k, raising=False)
    cfg = load_config()
    assert cfg.ocr.model == "QuantTrio/Qwen3.5-9B-AWQ"
    assert cfg.grounding.base_url.endswith("/v1")
    assert cfg.port == 8765
    assert cfg.sim_floor == 0.25
    assert cfg.hhem_threshold == 0.5

def test_transcribe_defaults_to_local_in_process(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("VPROC_"):
            monkeypatch.delenv(k, raising=False)
    cfg = load_config()
    assert cfg.transcribe.base_url == ""  # empty base_url => local in-process whisper
    assert "whisper" in cfg.transcribe.model.lower()

def test_transcribe_endpoint_override(monkeypatch):
    monkeypatch.setenv("VPROC_TRANSCRIBE_BASE_URL", "http://voyage:8001/v1")
    monkeypatch.setenv("VPROC_TRANSCRIBE_MODEL", "Systran/faster-whisper-large-v3")
    cfg = load_config()
    assert cfg.transcribe.base_url == "http://voyage:8001/v1"
    assert cfg.transcribe.model == "Systran/faster-whisper-large-v3"

def test_env_overrides(monkeypatch):
    monkeypatch.setenv("VPROC_GROUNDING_BASE_URL", "http://x:9/v1")
    monkeypatch.setenv("VPROC_GROUNDING_MODEL", "my-model")
    monkeypatch.setenv("VPROC_PORT", "9000")
    cfg = load_config()
    assert cfg.grounding.base_url == "http://x:9/v1"
    assert cfg.grounding.model == "my-model"
    assert cfg.port == 9000
