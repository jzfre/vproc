import sys
from importlib import metadata
from pathlib import Path
import tomllib

from vproc.config import load_config, load_dotenv
from vproc.errors import IndexBusyError
from vproc.locking import IndexWriteLock
from vproc.paths import memory_directory

USAGE = """usage: vproc [ingest <video.mp4> | serve | analyze <memory> "<question>" | speakers <memory> [--set 'SPEAKER_XX=Name'] | doctor]

  ingest     Transcribe and index a video.
  serve      Start the HTTP and MCP service.
  analyze    Ask a freeform question about a whole meeting transcript (summary, todos, topics).
  speakers   List or rename speakers in a memory.
  doctor     Check local installation prerequisites offline.
  --help     Show this help.
  --version  Show the installed version.
"""


def _version() -> str:
    try:
        return metadata.version("vproc")
    except metadata.PackageNotFoundError:
        # A source checkout can run without installed distribution metadata.
        try:
            source = Path(__file__).resolve().parents[1] / "pyproject.toml"
            project = tomllib.loads(source.read_text())["project"]
            if (isinstance(project, dict) and project.get("name") == "vproc"
                    and isinstance(project.get("version"), str)):
                return project["version"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return "0+unknown"


def _store_for(cfg):
    from vproc.store.lancedb_store import Store  # lazy: heavy

    return Store(cfg.index_path)


def _speakers(args: list[str]) -> None:
    from vproc.ingest import naming as N

    if len(args) not in (1, 3) or (len(args) == 3 and args[1] != "--set"):
        raise ValueError("usage: vproc speakers <memory> [--set 'SPEAKER_XX=Name']")
    cfg = load_config()
    memory = args[0]
    mem_dir = memory_directory(cfg.frames_dir, "default", memory)
    if len(args) == 3:
        old, _, new = args[2].partition("=")
        old, new = old.strip(), new.strip()
        if not old or not new:
            raise ValueError("usage: vproc speakers <memory> --set 'SPEAKER_XX=Name'")
        store = _store_for(cfg)
        with store.write_lock(), IndexWriteLock(cfg.frames_dir, name=".vproc-frames.lock"):
            mem_dir = memory_directory(cfg.frames_dir, "default", memory)
            try:
                m = N.load_speaker_map(mem_dir)
            except FileNotFoundError:
                m = {"mapping": {}, "suggestions": {}, "votes": []}
            n = store.update_speaker(memory, "default", old, new)
            if n == 0:
                print(f"warning: no rows matched {old}", file=sys.stderr)
                sys.exit(1)
            m["mapping"][old] = new
            m["suggestions"].pop(old, None)
            N.save_speaker_map(mem_dir, m)
            print(f"renamed {old} -> {new} in '{memory}' ({n} rows)")
        return
    try:
        m = N.load_speaker_map(mem_dir)
    except FileNotFoundError:
        print(f"no speaker map for '{memory}' (expected {mem_dir}/speakers.json)")
        return
    for spk, name in sorted(m.get("mapping", {}).items()):
        print(f"{spk} = {name}")
    for spk, s in sorted(m.get("suggestions", {}).items()):
        votes = ", ".join(f"{n} x{c}" for n, c in sorted(s.get("votes", {}).items()))
        print(f"{spk} = ? (votes: {votes or 'none'})")
        for ev in s.get("evidence", []):
            print(f"    seen at {ev['t']}s: {ev['name']}")


def main() -> None:
    try:
        _main(sys.argv[1:])
    except (IndexBusyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


def _main(args: list[str]) -> None:
    if args in (["--help"], ["-h"], ["help"]):
        print(USAGE, end="")
        return
    if args == ["--version"]:
        print(f"vproc {_version()}")
        return
    load_dotenv()
    if args == ["doctor"]:
        from vproc.doctor import run_doctor

        status = run_doctor(load_config())
        if status:
            sys.exit(status)
    elif len(args) == 2 and args[0] == "ingest":
        from vproc.ingest.pipeline import ingest_video

        n = ingest_video(args[1])
        if n == 0:
            print(f"warning: 0 segments ingested from {args[1]} (no transcript or on-screen text)",
                  file=sys.stderr)
        print(f"ingested {n} segments from {args[1]}")
    elif args == ["serve"]:
        import uvicorn

        from vproc.service import create_app

        cfg = load_config()
        uvicorn.run(create_app(cfg=cfg), host=cfg.host, port=cfg.port)
    elif len(args) == 3 and args[0] == "analyze":
        from vproc.answer.analyze import analyze_memory

        cfg = load_config()
        store = _store_for(cfg)
        print(analyze_memory(store, cfg, args[1], args[2]))
    elif len(args) >= 2 and args[0] == "speakers":
        _speakers(args[1:])
    else:
        print(USAGE, end="")
        sys.exit(1)
