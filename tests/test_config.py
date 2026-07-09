import os

from vproc.config import load_config, load_dotenv

def test_hhem_model_default_and_override(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VPROC_"):
            monkeypatch.delenv(k, raising=False)
    assert "vectara" in load_config().hhem_model
    monkeypatch.setenv("VPROC_HHEM_MODEL", "my-org/hhem")
    assert load_config().hhem_model == "my-org/hhem"

def test_load_dotenv_sets_without_override(tmp_path, monkeypatch):
    monkeypatch.delenv("VPROC_NEWVAR", raising=False)
    monkeypatch.setenv("VPROC_PRESET", "preset")
    env = tmp_path / ".env"
    env.write_text("# comment\nVPROC_NEWVAR='fromfile'\nVPROC_PRESET=fromfile\n\nnonsense-line\n")
    try:
        load_dotenv(str(env))
        assert os.environ["VPROC_NEWVAR"] == "fromfile"      # set from file
        assert os.environ["VPROC_PRESET"] == "preset"        # existing value NOT overridden
    finally:
        os.environ.pop("VPROC_NEWVAR", None)

def test_load_dotenv_missing_file_is_noop():
    load_dotenv("/no/such/.env")  # must not raise

def test_load_dotenv_export_prefix(tmp_path, monkeypatch):
    monkeypatch.delenv("VPROC_EXPORTED", raising=False)
    env = tmp_path / ".env"
    env.write_text("export VPROC_EXPORTED=http://voyage:8000/v1\n")
    try:
        load_dotenv(str(env))
        assert os.environ["VPROC_EXPORTED"] == "http://voyage:8000/v1"
        assert "export VPROC_EXPORTED" not in os.environ
    finally:
        os.environ.pop("VPROC_EXPORTED", None)

def test_load_dotenv_strips_inline_comment(tmp_path, monkeypatch):
    monkeypatch.delenv("VPROC_PORTISH", raising=False)
    monkeypatch.delenv("VPROC_URLISH", raising=False)
    env = tmp_path / ".env"
    env.write_text('VPROC_PORTISH=8765  # service port\n'
                   'VPROC_URLISH="http://voyage:8000/v1"  # embeddings\n')
    try:
        load_dotenv(str(env))
        assert os.environ["VPROC_PORTISH"] == "8765"
        assert os.environ["VPROC_URLISH"] == "http://voyage:8000/v1"
    finally:
        os.environ.pop("VPROC_PORTISH", None)
        os.environ.pop("VPROC_URLISH", None)

def test_load_dotenv_empty_key_is_ignored(tmp_path):
    env = tmp_path / ".env"
    env.write_text("=foo\n = bar\nVPROC_OK=ok\n")
    try:
        load_dotenv(str(env))  # must not raise OSError on the empty-key lines
        assert os.environ["VPROC_OK"] == "ok"
    finally:
        os.environ.pop("VPROC_OK", None)

def test_frames_dir_default_and_override(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VPROC_"):
            monkeypatch.delenv(k, raising=False)
    assert load_config().frames_dir == "./vproc_frames"
    monkeypatch.setenv("VPROC_FRAMES_DIR", "/data/frames")
    assert load_config().frames_dir == "/data/frames"

def test_grounding_thinking_default_and_off(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VPROC_"):
            monkeypatch.delenv(k, raising=False)
    assert load_config().grounding_thinking is True
    for off in ("off", "false", "0", "no", "OFF"):
        monkeypatch.setenv("VPROC_GROUNDING_THINKING", off)
        assert load_config().grounding_thinking is False, off
    monkeypatch.setenv("VPROC_GROUNDING_THINKING", "on")
    assert load_config().grounding_thinking is True

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
