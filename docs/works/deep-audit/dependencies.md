# Dependency security audit — 2026-11-17

## Results and scope

The audit compared locked Python distributions against pip-audit's advisory
service on Linux/Python 3.12. Counts are dependency advisory matches, not proven
exploits in vProc. Some service records repeat an ID with multiple descriptions;
we count both raw records and unique advisory IDs below.

| Install | Before: records / unique IDs / packages | After: records / unique IDs / packages |
| --- | --- | --- |
| Base application | 118 / 98 / 11 | 0 / 0 / 0 |
| All optional runtime extras | 42 / 34 / 8 | 12 / 8 / 2 |
| Optional local model server | 28 / 23 / 4 | 12 / 8 / 2 |

The runtime "before" check was taken after the base security upgrades, so its
42 records are additional optional-dependency findings. These scans evaluate
this host's applicable environment markers; they do not certify macOS/Windows
packages, native system FFmpeg libraries, future advisory databases, model
weights, or repository-hosted model code. No advisories were suppressed.

## Changes made

- Updated affected base locks: AnyIO, certifi, cryptography, filelock, MCP,
  Pillow, pydantic-settings, PyJWT, python-multipart, Starlette and urllib3.
  Security minimums in `pyproject.toml` also apply when installing a wheel
  without the source lock. The existing model feature extras remain optional.
- Updated optional runtime dependencies aiohttp, fsspec, Lightning, multidict,
  OpenTelemetry and pytorch-lightning, with corresponding minimums in the
  optional dependency groups. The local-model script's separate lock updates
  certifi and filelock; its dependency header also records applicable floors.
- Preserved the CPU PyTorch wheel source and the Transformers 4.x compatibility
  constraint. No model engine migration is hidden in this dependency update.
- MCP 1.30.0 contains the 1.x security backports, retaining `fastmcp`. Its
  [session retention advisory](https://github.com/modelcontextprotocol/python-sdk/security/advisories/GHSA-84m7-p3x7-pcfv)
  explicitly lists 1.30.0 as patched. The service also uses stateless HTTP.
- FileLock now explicitly preserves its lock file across releases; tests check
  inode stability as well as actual cross-process contention and termination.

Many matched advisories are outside the deployed code paths. For example, vProc
reads FFmpeg-generated PNGs rather than exposing an arbitrary Pillow decoder
upload endpoint; it does not run MCP client OAuth or bearer authentication.
Those distinctions informed triage but were not used to skip available updates.

## Remaining findings

The eight unique IDs below remain in both optional local-model environments.
They are absent from the base application install. The current locked
`torch==2.12.0` CPU distribution requires `setuptools<82`; the resolver rejects
`setuptools>=84.1`. The HHEM remote model targets Transformers 4.x, so changing to
5.x requires a separate model compatibility migration. The application does
not have a tested drop-in migration that removes these constraints.

| Package and advisory | Affected operation | vProc boundary and disposition |
| --- | --- | --- |
| setuptools 81.0.0 — [PYSEC-2026-3447 / CVE-2026-59890 / GHSA-h35f-9h28-mq5c](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c) | Unicode normalization bypass of `MANIFEST.in` exclusions during setuptools source-archive builds on macOS. Fixed in 83.0.0. | vProc builds its releases with Hatchling and explicit inclusion lists, not setuptools manifests. Setuptools remains installed for PyTorch; changing its version alone is incompatible with the locked runtime. |
| setuptools 81.0.0 — [GHSA-xmj3-9gf7-h52w](https://github.com/pypa/setuptools/security/advisories/GHSA-xmj3-9gf7-h52w) | Malicious wheel namespace metadata escapes the destination in setuptools' legacy Windows wheel installer. Fixed in 84.1.0. | Documented installation uses uv, not setuptools' legacy installer. Do not use the runtime environment as a general-purpose installer for untrusted wheels. Same PyTorch constraint prevents a compatible upgrade. |
| transformers 4.57.6 — [PYSEC-2025-217 / CVE-2025-14929](https://www.zerodayinitiative.com/advisories/ZDI-25-1144/) | X-CLIP checkpoint conversion deserializes attacker-controlled data. The audit record does not advertise a fixed release. | vProc does not run checkpoint-conversion utilities. Keep model sources trusted; the installed library still contains affected code. |
| transformers 4.57.6 — [PYSEC-2026-2288 / CVE-2026-1839 / GHSA-69w3-r845-3855](https://github.com/advisories/GHSA-69w3-r845-3855) | `Trainer._load_rng_state()` loads untrusted state; the described exploit requires PyTorch below 2.6. Fixed in Transformers 5.0.0. | vProc performs inference, not Trainer checkpoint resume, and its lock uses PyTorch 2.12. It does not meet those stated preconditions. |
| transformers 4.57.6 — [PYSEC-2026-2289 / CVE-2026-4372 / GHSA-29pf-2h5f-8g72](https://github.com/advisories/GHSA-29pf-2h5f-8g72) | A malicious model config can select a remote attention implementation and cause code loading. Fixed in 5.3.0. | Model loading is a relevant trust boundary. Only administrator-selected trusted model repositories belong here; untrusted model IDs/configs are not safe inputs. The HHEM integration already explicitly enables trusted remote model code. This is a residual risk, not claimed fixed. |
| transformers 4.57.6 — [PYSEC-2026-2290 / CVE-2026-5241 / GHSA-fgcw-684q-jj6r](https://github.com/advisories/GHSA-fgcw-684q-jj6r) | LightGlue nested loading overrides `trust_remote_code` from serialized configuration. Fixed in 5.5.0. | Default HHEM and sentence-embedding models are not LightGlue; vProc does not expose a LightGlue workflow. A different administrator-configured model could change reachability. |
| transformers 4.57.6 — [PYSEC-2026-3929 / CVE-2026-9856 / GHSA-xrqw-3rrv-vx5w](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w) | Path traversal through chat-template names when tokenizer/processor `save_pretrained()` writes files. Fixed in 5.10.0. | vProc does not invoke tokenizer or processor `save_pretrained()`. Do not extend this environment to save untrusted downloaded model configurations without addressing the advisory. |
| transformers 4.57.6 — [PYSEC-2026-4174 / CVE-2026-80047 / GHSA-x9r9-c232-4q39](https://github.com/advisories/GHSA-x9r9-c232-4q39) | `load_custom_generate()` fetches remote Python before the trust-consent check. The audit record does not advertise a fixed release. | vProc uses HTTP generation and local classification/embedding; it does not invoke the custom-generation API. Configuring different remote model code can add other paths, so this is not a guarantee for arbitrary repositories. |

The upstream fixes linked from the advisory records include the
[Trainer change](https://github.com/huggingface/transformers/commit/03c8082ba4594c9b8d6fe190ca9bed0e5f8ca396),
[attention configuration change](https://github.com/huggingface/transformers/commit/a7f8e7ff37d87d1a1a0c8cf607971c607741452f),
[LightGlue change](https://github.com/huggingface/transformers/commit/676559d5022b74aaa0cee1cee0842b7f27c5320e),
[tokenizer saving change](https://github.com/huggingface/transformers/commit/eaaaf8494dd5386634ae37d1d122212fdc315be5), and
[custom-generation change](https://github.com/huggingface/transformers/commit/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55).
Operation reachability above is an assessment from current vProc source, not
an upstream attestation that the entire application is unaffected.

## Reproduce

The export resolves the committed lock without upgrading it. `--no-deps` prevents
pip-audit from replacing locked dependency choices; the full exported dependency
set is scanned. `--disable-pip` avoids installing packages for the audit. Advisory
queries need network access. pip-audit exits nonzero when findings remain.

```sh
uv export --locked --no-dev --no-emit-project --no-hashes --format requirements-txt --output-file /tmp/vproc-base.txt
uv tool run pip-audit --no-deps --disable-pip -r /tmp/vproc-base.txt --format json --output /tmp/vproc-base-audit.json

uv export --locked --no-dev --all-extras --no-emit-project --no-hashes --format requirements-txt --output-file /tmp/vproc-runtime.txt
uv tool run pip-audit --no-deps --disable-pip -r /tmp/vproc-runtime.txt --format json --output /tmp/vproc-runtime-audit.json

uv export --locked --script scripts/local_models.py --no-hashes --format requirements-txt --output-file /tmp/vproc-models.txt
uv tool run pip-audit --no-deps --disable-pip -r /tmp/vproc-models.txt --format json --output /tmp/vproc-models-audit.json
```

To count unique advisory IDs in a JSON result:

```sh
python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('/tmp/vproc-runtime-audit.json').read_text())
findings = [v for d in report['dependencies'] for v in d.get('vulns', [])]
print('records:', len(findings), 'unique IDs:', len({v['id'] for v in findings}))
PY
```

## Compatibility verification

The full suite passed after the base dependency upgrade (739 tests at that point
in the concurrent audit). A later focused run passed 98 storage-lock, packaging,
local-model API, service, and MCP tests after the optional lock updates. A fresh
full run after all dependency changes passed **762 tests**, with no skips. Package
lock tests retain Linux CPU wheels and reject CUDA dependencies. These checks use
controlled model responses; they do not certify numerical quality or downloads
for all optional models, platforms, or external model repositories. The main audit
report records the final full-suite count after all agent changes are integrated.
