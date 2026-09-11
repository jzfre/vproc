import os
from dataclasses import replace

import pytest

from vproc.config import Config, Endpoint, load_config, load_dotenv

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


def test_load_dotenv_accepts_utf8_bom_from_windows_editors(tmp_path, monkeypatch):
    monkeypatch.delenv("VPROC_FRAMES_DIR", raising=False)
    env = tmp_path / ".env"
    value = "C:/Users/Jozef/stretnutia-žluťoučký"
    env.write_bytes(("VPROC_FRAMES_DIR=" + value + "\n").encode("utf-8-sig"))
    try:
        load_dotenv(str(env))
        loaded = os.environ.get("VPROC_FRAMES_DIR")
        assert loaded == value
    finally:
        os.environ.pop("VPROC_FRAMES_DIR", None)

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

def test_diarize_model_default_and_disable(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VPROC_"):
            monkeypatch.delenv(k, raising=False)
    assert load_config().diarize_model == "pyannote/speaker-diarization-community-1"
    monkeypatch.setenv("VPROC_DIARIZE_MODEL", "")   # empty string disables diarization
    assert load_config().diarize_model == ""
    monkeypatch.setenv("VPROC_DIARIZE_MODEL", "pyannote/speaker-diarization-3.1")
    assert load_config().diarize_model == "pyannote/speaker-diarization-3.1"

def test_speaker_naming_default_and_off(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VPROC_"):
            monkeypatch.delenv(k, raising=False)
    assert load_config().speaker_naming is True
    for off in ("off", "false", "0", "no", "OFF"):
        monkeypatch.setenv("VPROC_SPEAKER_NAMING", off)
        assert load_config().speaker_naming is False, off
    monkeypatch.setenv("VPROC_SPEAKER_NAMING", "on")
    assert load_config().speaker_naming is True


@pytest.fixture
def clean_config_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("VPROC_"):
            monkeypatch.delenv(key, raising=False)


def _config(**overrides):
    ep = Endpoint("u", "m")  # direct configs support injected test/local clients
    return replace(Config(ep, ep, ep, "./index", "localhost", 8765, 0.25, 0.5, None),
                   **overrides)


@pytest.mark.parametrize("field, values", [
    ("sim_floor", [float("nan"), float("inf"), -float("inf"), -1.01, 1.01, True, "0.2"]),
    ("hhem_threshold", [float("nan"), float("inf"), -0.01, 1.01, False, "0.5"]),
    ("ocr_timeout", [float("nan"), float("inf"), -float("inf"), 0, -1, True, "60"]),
    ("ocr_max_tokens", [0, -1, True, 1.5, "1024"]),
    ("grounding_max_tokens", [0, -1, False, 1.5, "8192"]),
    ("port", [0, -1, 65536, True, 8765.5, "8765"]),
])
def test_config_rejects_invalid_numeric_values(field, values):
    for value in values:
        with pytest.raises(ValueError, match=field):
            _config(**{field: value})


@pytest.mark.parametrize("overrides", [
    {"sim_floor": -1, "hhem_threshold": 0, "port": 1},
    {"sim_floor": 1, "hhem_threshold": 1, "port": 65535},
    {"ocr_timeout": 0.01, "ocr_max_tokens": 1, "grounding_max_tokens": 1},
    {"transcribe": Endpoint("", "local-model"), "diarize_model": ""},
])
def test_config_accepts_boundaries_and_intentional_empty_values(overrides):
    _config(**overrides)


@pytest.mark.parametrize("field", ["index_path", "frames_dir", "host", "hhem_model"])
@pytest.mark.parametrize("value", ["", " \t", None, "bad\0value"])
def test_config_requires_nonblank_strings(field, value):
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


@pytest.mark.parametrize("field", ["ocr", "embed", "grounding", "transcribe"])
@pytest.mark.parametrize("model", ["", " \t", None, "bad\0model"])
def test_config_requires_endpoint_models(field, model):
    with pytest.raises(ValueError, match=field):
        _config(**{field: Endpoint("u", model)})


@pytest.mark.parametrize("field", ["ocr", "embed", "grounding"])
@pytest.mark.parametrize("url", ["", " \t", None, "bad\0url"])
def test_config_requires_remote_endpoint_values(field, url):
    with pytest.raises(ValueError, match=field):
        _config(**{field: Endpoint(url, "model")})


@pytest.mark.parametrize("value", [" \t", None, "bad\0model"])
def test_config_rejects_malformed_optional_diarization_model(value):
    with pytest.raises(ValueError, match="diarize_model"):
        _config(diarize_model=value)


@pytest.mark.parametrize("key, value", [
    ("VPROC_SIM_FLOOR", "nan"),
    ("VPROC_SIM_FLOOR", "-1.01"),
    ("VPROC_HHEM_THRESHOLD", "inf"),
    ("VPROC_HHEM_THRESHOLD", "1.01"),
    ("VPROC_OCR_TIMEOUT", "nan"),
    ("VPROC_OCR_TIMEOUT", "0"),
    ("VPROC_OCR_MAX_TOKENS", "0"),
    ("VPROC_GROUNDING_MAX_TOKENS", "-1"),
    ("VPROC_PORT", "65536"),
])
def test_load_config_rejects_invalid_numeric_environment(clean_config_env, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key.removeprefix("VPROC_").lower()):
        load_config()


@pytest.mark.parametrize("key", ["VPROC_SIM_FLOOR", "VPROC_HHEM_THRESHOLD", "VPROC_OCR_TIMEOUT",
                                 "VPROC_OCR_MAX_TOKENS", "VPROC_GROUNDING_MAX_TOKENS", "VPROC_PORT"])
def test_load_config_identifies_malformed_numeric_setting(clean_config_env, monkeypatch, key):
    monkeypatch.setenv(key, "typo")
    with pytest.raises(ValueError, match=key):
        load_config()


@pytest.mark.parametrize("key", ["VPROC_OCR_BASE_URL", "VPROC_EMBED_BASE_URL",
                                 "VPROC_GROUNDING_BASE_URL", "VPROC_TRANSCRIBE_BASE_URL"])
@pytest.mark.parametrize("value", [" \t", "voyage:8000/v1", "ftp://voyage/v1",
                                   "http:///v1", "http://host:bad/v1", "http://host:65536/v1",
                                   "http://host:0/v1", "http://bad host/v1", "http://host/\npath",
                                   "http://[::1", "http://host/v1?route=x", "http://host/v1#route"])
def test_load_config_rejects_invalid_endpoint_urls(clean_config_env, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        load_config()


@pytest.mark.parametrize("key", ["VPROC_OCR_BASE_URL", "VPROC_EMBED_BASE_URL", "VPROC_GROUNDING_BASE_URL"])
def test_load_config_rejects_empty_remote_endpoint(clean_config_env, monkeypatch, key):
    monkeypatch.setenv(key, "")
    with pytest.raises(ValueError, match=key):
        load_config()


@pytest.mark.parametrize("url", ["http://localhost:1234/v1", "https://api.example.com/v1/",
                                 "http://voyage:8000", "http://[::1]:1234/v1"])
def test_load_config_accepts_http_endpoint_urls(clean_config_env, monkeypatch, url):
    monkeypatch.setenv("VPROC_OCR_BASE_URL", url)
    assert load_config().ocr.base_url == url


@pytest.mark.parametrize("key", ["VPROC_GROUNDING_THINKING", "VPROC_SPEAKER_NAMING"])
@pytest.mark.parametrize("value", ["of", "flase", "enabled", "", "2"])
def test_load_config_rejects_boolean_typos(clean_config_env, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        load_config()


@pytest.mark.parametrize("key, field", [("VPROC_GROUNDING_THINKING", "grounding_thinking"),
                                       ("VPROC_SPEAKER_NAMING", "speaker_naming")])
@pytest.mark.parametrize("value, expected", [("on", True), ("TRUE", True), (" 1 ", True),
                                            ("yes", True), ("off", False), (" false ", False),
                                            ("0", False), ("NO", False)])
def test_load_config_accepts_explicit_booleans(clean_config_env, monkeypatch, key, field, value, expected):
    monkeypatch.setenv(key, value)
    assert getattr(load_config(), field) is expected


def test_load_dotenv_ignores_invalid_keys_and_nul_without_losing_valid_lines(tmp_path, monkeypatch):
    keys = ["BAD KEY", "BAD-KEY", "9BAD", "VPROC_NUL", "VPROC_OK"]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VPROC_PRESET", "external")
    env = tmp_path / ".env"
    env.write_text("BAD KEY=x\nBAD-KEY=x\n9BAD=x\nBAD\0KEY=x\nVPROC_NUL=bad\0value\n"
                   "VPROC_PRESET=file\nVPROC_OK=ok\n")
    load_dotenv(str(env))
    assert os.environ["VPROC_PRESET"] == "external"
    assert os.environ["VPROC_OK"] == "ok"
    for key in keys[:-1]:
        assert key not in os.environ
