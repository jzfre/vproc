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
