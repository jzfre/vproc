from dataclasses import replace
import sys

import pytest

from vproc.config import DEFAULT_DIARIZE_MODEL, Config, Endpoint
import vproc.doctor as doctor_module


@pytest.fixture
def doctor(monkeypatch):
    module = doctor_module
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(module, "_module_available", lambda name: True)
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    return module


@pytest.fixture
def cfg(tmp_path):
    remote = Endpoint("http://user:secret@localhost:1234/v1", "remote-model")
    return Config(
        ocr=remote, embed=remote, grounding=remote, transcribe=remote,
        index_path=str(tmp_path / "new" / "index"), frames_dir=str(tmp_path / "new" / "frames"),
        host="127.0.0.1", port=8765, sim_floor=0.25, hhem_threshold=0.5,
        hf_token="hf_private_token", diarize_model="",
    )


def test_remote_preflight_reports_local_checks_without_network_or_writes(doctor, cfg, monkeypatch, capsys, tmp_path):
    import socket

    def no_network(*args, **kwargs):
        pytest.fail("doctor must not contact endpoints")

    monkeypatch.setattr(socket, "socket", no_network)
    assert doctor.run_doctor(cfg) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "not probed" in out
    assert "HF_TOKEN" not in out  # disabled diarization needs no token
    assert "secret" not in out and "hf_private_token" not in out
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("binary", ["ffmpeg", "ffprobe"])
def test_missing_media_binary_fails_with_install_guidance(doctor, cfg, monkeypatch, capsys, binary):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None if name == binary else f"/usr/bin/{name}")
    assert doctor.run_doctor(cfg) == 1
    out = capsys.readouterr().out
    assert f"FAIL {binary}" in out and "PATH" in out


def test_local_asr_rejects_non_apple_silicon(doctor, cfg, capsys):
    assert doctor.run_doctor(replace(cfg, transcribe=Endpoint("", "local-model"))) == 1
    out = capsys.readouterr().out
    assert "Apple Silicon" in out and "VPROC_TRANSCRIBE_BASE_URL" in out


def test_local_asr_requires_local_extra_on_apple_silicon(doctor, cfg, monkeypatch, capsys):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(doctor.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(doctor, "_module_available", lambda name: name != "mlx_whisper")
    assert doctor.run_doctor(replace(cfg, transcribe=Endpoint("", "local-model"))) == 1
    assert "local-asr" in capsys.readouterr().out


@pytest.mark.parametrize("module", ["torch", "transformers"])
def test_answering_requires_optional_model_packages(doctor, cfg, monkeypatch, capsys, module):
    monkeypatch.setattr(doctor, "_module_available", lambda name: name != module)
    assert doctor.run_doctor(cfg) == 1
    out = capsys.readouterr().out
    assert module in out and "answer extra" in out


def test_enabled_diarization_requires_its_extra(doctor, cfg, monkeypatch, capsys):
    monkeypatch.setattr(doctor, "_module_available", lambda name: name != "pyannote.audio")
    assert doctor.run_doctor(replace(cfg, diarize_model=DEFAULT_DIARIZE_MODEL)) == 1
    out = capsys.readouterr().out
    assert "diarization extra" in out and "VPROC_DIARIZE_MODEL" in out


def test_default_gated_diarization_requires_token_without_printing_it(doctor, cfg, capsys):
    assert doctor.run_doctor(replace(cfg, diarize_model=DEFAULT_DIARIZE_MODEL, hf_token=" ")) == 1
    out = capsys.readouterr().out
    assert "FAIL HF_TOKEN" in out and "access" in out


def test_custom_diarization_model_does_not_require_token(doctor, cfg, capsys):
    assert doctor.run_doctor(replace(cfg, diarize_model="/models/custom", hf_token=None)) == 0
    assert "FAIL HF_TOKEN" not in capsys.readouterr().out


def test_file_in_storage_parent_chain_fails_without_modifying_it(doctor, cfg, tmp_path, capsys):
    parent = tmp_path / "file"
    parent.write_text("keep")
    assert doctor.run_doctor(replace(cfg, index_path=str(parent / "db"))) == 1
    assert "FAIL VPROC_INDEX_PATH" in capsys.readouterr().out
    assert parent.read_text() == "keep"


def test_package_detection_does_not_import_optional_package(tmp_path, monkeypatch):
    package = tmp_path / "offline_optional"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('must not import')")
    (package / "audio.py").write_text("raise RuntimeError('must not import')")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert doctor_module._module_available("offline_optional.audio")
    assert not doctor_module._module_available("offline_optional.missing")
    assert "offline_optional" not in sys.modules


def test_cli_doctor_propagates_failure_status(doctor, cfg, monkeypatch, capsys):
    import vproc.cli as cli

    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.setattr("sys.argv", ["vproc", "doctor"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "FAIL ffmpeg" in capsys.readouterr().out


def test_python_before_minimum_version_fails(doctor, cfg, monkeypatch, capsys):
    monkeypatch.setattr(doctor.sys, "version_info", (3, 11, 0))
    assert doctor.run_doctor(cfg) == 1
    assert "FAIL Python" in capsys.readouterr().out


def test_storage_permission_failure_is_actionable(doctor, cfg, monkeypatch, capsys):
    monkeypatch.setattr(doctor.os, "access", lambda *args: False)
    assert doctor.run_doctor(cfg) == 1
    out = capsys.readouterr().out
    assert "FAIL VPROC_INDEX_PATH" in out and "FAIL VPROC_FRAMES_DIR" in out


# ---- offline output that makes a broken setup visible before anything runs ----

def test_disabled_ocr_is_reported_not_described_as_an_endpoint(doctor, cfg, capsys):
    assert doctor.run_doctor(replace(cfg, ocr=Endpoint("", "unused"))) == 0
    out = capsys.readouterr().out
    assert "INFO OCR: disabled" in out and "nameplate" in out


def test_endpoint_lines_show_url_without_credentials_and_the_model(doctor, cfg, capsys):
    doctor.run_doctor(cfg)
    out = capsys.readouterr().out
    assert "INFO grounding: http://localhost:1234/v1 model 'remote-model'; not probed" in out
    assert "secret" not in out and "user:" not in out


def test_config_source_is_reported(doctor, cfg, capsys, tmp_path):
    doctor.run_doctor(cfg, env_file=str(tmp_path / ".env"))
    assert f"INFO config: loaded {tmp_path / '.env'}" in capsys.readouterr().out
    assert doctor.run_doctor(cfg, env_file=None) == 0  # a warning, never a failure
    assert "WARN config: no .env" in capsys.readouterr().out


def test_misspelled_setting_is_flagged_with_the_likely_name(doctor, cfg, monkeypatch, capsys):
    monkeypatch.setenv("VPROC_GROUNDING_THINKNG", "off")
    assert doctor.run_doctor(cfg) == 0
    out = capsys.readouterr().out
    assert "WARN unknown setting VPROC_GROUNDING_THINKNG" in out
    assert "VPROC_GROUNDING_THINKING" in out


# ---- opt-in online probes (fakes only: tests never touch the network) ----

class _HttpError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code} https://user:secret@host/ token=hf_private_token")
        self.status_code = status_code


def _probes(**overrides):
    calls = []
    probes = {
        "models": lambda base_url: calls.append(("models", base_url)) or ["remote-model"],
        "embed": lambda cfg: calls.append(("embed",)) or 2,
        "chat": lambda cfg: calls.append(("chat",)),
        "hub": lambda cfg: calls.append(("hub",)),
    }
    probes.update(overrides)
    return probes, calls


def test_online_probes_every_backend_once_and_passes(doctor, cfg, capsys):
    probes, calls = _probes()
    assert doctor.run_doctor(replace(cfg, diarize_model=DEFAULT_DIARIZE_MODEL), online=True,
                             probes=probes) == 0
    out = capsys.readouterr().out
    assert calls.count(("models", "http://user:secret@localhost:1234/v1")) == 1  # one shared server
    assert ("embed",) in calls and ("chat",) in calls and ("hub",) in calls
    assert "PASS embedding probe: 2-dimensional vectors" in out
    assert "PASS Hugging Face access" in out and "PASS grounding probe" in out
    assert "secret" not in out and "hf_private_token" not in out


def test_online_rejected_key_fails_with_the_setting_to_fix(doctor, cfg, capsys):
    def unauthorized(base_url):
        raise _HttpError(401)

    probes, _ = _probes(models=unauthorized)
    assert doctor.run_doctor(cfg, online=True, probes=probes) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "401" in out and "_API_KEY" in out
    assert "secret" not in out and "hf_private_token" not in out


def test_online_tolerates_servers_without_a_model_list_and_unlisted_aliases(doctor, cfg, capsys):
    def no_listing(base_url):
        raise _HttpError(404)

    probes, _ = _probes(models=no_listing)
    assert doctor.run_doctor(cfg, online=True, probes=probes) == 0
    assert "INFO" in capsys.readouterr().out
    probes, _ = _probes(models=lambda base_url: ["some-other-model"])
    assert doctor.run_doctor(cfg, online=True, probes=probes) == 0
    assert "WARN" in capsys.readouterr().out and True


def test_online_busy_generation_backend_is_a_warning_not_a_failure(doctor, cfg, capsys):
    import httpx
    from openai import APITimeoutError

    def busy(cfg):
        raise APITimeoutError(request=httpx.Request("POST", "http://x/v1/chat/completions"))

    probes, _ = _probes(chat=busy)
    assert doctor.run_doctor(cfg, online=True, probes=probes) == 0
    assert "WARN grounding probe" in capsys.readouterr().out


def test_online_invalid_hub_token_fails_with_remedies_and_no_token(doctor, cfg, capsys):
    def rejected(cfg):
        raise _HttpError(401)

    probes, _ = _probes(hub=rejected)
    assert doctor.run_doctor(replace(cfg, diarize_model=DEFAULT_DIARIZE_MODEL), online=True,
                             probes=probes) == 1
    out = capsys.readouterr().out
    assert "FAIL Hugging Face access" in out and "HF_TOKEN" in out and "accept" in out
    assert "hf_private_token" not in out


def test_online_embedding_dimension_is_checked_against_an_existing_index(doctor, cfg, capsys, tmp_path):
    from vproc.store.lancedb_store import Store

    index = tmp_path / "index"
    row = {"id": "a", "project_id": "p", "memory_id": "m", "speaker": "S", "start_ts": 0.0,
           "end_ts": 1.0, "said_text": "x", "on_screen_text": "", "embed_text": "x",
           "source_video": "/v.mp4", "frame_path": "", "vector": [1.0, 0.0]}
    Store(str(index)).add([row], embedding_model="remote-model")
    probes, _ = _probes(embed=lambda cfg: 3)
    assert doctor.run_doctor(replace(cfg, index_path=str(index)), online=True, probes=probes) == 1
    assert "FAIL embedding probe" in capsys.readouterr().out


def test_cli_doctor_online_flag(doctor, cfg, monkeypatch):
    import vproc.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(doctor, "run_doctor", lambda cfg, **kw: seen.update(kw) or 0)
    monkeypatch.setattr("sys.argv", ["vproc", "doctor", "--online"])
    cli.main()
    assert seen["online"] is True and "env_file" in seen


def test_helper_script_setting_is_not_reported_as_unknown(doctor, cfg, monkeypatch, capsys):
    monkeypatch.setenv("VPROC_ASK_URL", "http://100.64.0.1:8765")  # read by scripts/ask.sh
    doctor.run_doctor(cfg)
    assert "unknown setting" not in capsys.readouterr().out


def test_online_index_check_expands_home_like_the_store_does(doctor, cfg, capsys, tmp_path, monkeypatch):
    from vproc.store.lancedb_store import Store

    monkeypatch.setenv("HOME", str(tmp_path))
    row = {"id": "a", "project_id": "p", "memory_id": "m", "speaker": "S", "start_ts": 0.0,
           "end_ts": 1.0, "said_text": "x", "on_screen_text": "", "embed_text": "x",
           "source_video": "/v.mp4", "frame_path": "", "vector": [1.0, 0.0]}
    Store(str(tmp_path / "index")).add([row], embedding_model="remote-model")
    probes, _ = _probes(embed=lambda cfg: 3)
    assert doctor.run_doctor(replace(cfg, index_path="~/index"), online=True, probes=probes) == 1
    assert "FAIL embedding probe" in capsys.readouterr().out


def test_online_embedding_pass_does_not_claim_an_index_it_never_saw(doctor, cfg, capsys):
    probes, _ = _probes()
    doctor.run_doctor(cfg, online=True, probes=probes)   # cfg.index_path does not exist
    out = capsys.readouterr().out
    assert "PASS embedding probe: 2-dimensional vectors" in out
    assert "compatible with the index" not in out and "no index yet" in out


def test_online_skips_the_hub_for_a_local_diarization_pipeline_file(doctor, cfg, capsys, tmp_path):
    local = tmp_path / "config.yaml"
    local.write_text("pipeline: {}")

    def hub_must_not_run(cfg):
        pytest.fail("a local pipeline needs no Hugging Face access")

    probes, _ = _probes(hub=hub_must_not_run)
    assert doctor.run_doctor(replace(cfg, diarize_model=str(local)), online=True, probes=probes) == 0
    assert "local pipeline" in capsys.readouterr().out


def test_embedding_probe_uses_short_timeout(monkeypatch, cfg):
    from vproc.llm import client

    called = {}

    def embed(base_url, model, texts, *, timeout=None):
        called["timeout"] = timeout
        return [[1.0, 0.0]]

    monkeypatch.setattr(client, "embed_texts", embed)
    assert doctor_module._default_probes()["embed"](cfg) == 2
    assert called["timeout"] == 15.0
