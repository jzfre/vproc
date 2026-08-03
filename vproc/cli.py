import os
import sys

from vproc.config import load_config, load_dotenv


def _store_for(cfg):
    from vproc.store.lancedb_store import Store  # lazy: heavy

    return Store(cfg.index_path)


def _speakers(args: list[str]) -> None:
    from vproc.ingest import naming as N

    cfg = load_config()
    memory = args[0]
    mem_dir = os.path.join(cfg.frames_dir, "default", memory)
    is_set = len(args) >= 3 and args[1] == "--set"
    try:
        m = N.load_speaker_map(mem_dir)
    except FileNotFoundError:
        if not is_set:
            print(f"no speaker map for '{memory}' (expected {mem_dir}/speakers.json)")
            return
        m = {"mapping": {}, "suggestions": {}, "votes": []}  # bootstrap: --set works cold
    if is_set:
        old, _, new = args[2].partition("=")
        old, new = old.strip(), new.strip()
        if not old or not new:
            print("usage: vproc speakers <memory> --set 'SPEAKER_XX=Name'")
            sys.exit(1)
        n = _store_for(cfg).update_speaker(memory, "default", old, new)
        m["mapping"][old] = new
        m["suggestions"].pop(old, None)
        N.save_speaker_map(mem_dir, m)
        print(f"renamed {old} -> {new} in '{memory}' ({n} rows)")
        if n == 0:
            print(f"warning: no rows matched {old}", file=sys.stderr)
        return
    for spk, name in sorted(m.get("mapping", {}).items()):
        print(f"{spk} = {name}")
    for spk, s in sorted(m.get("suggestions", {}).items()):
        votes = ", ".join(f"{n} x{c}" for n, c in sorted(s.get("votes", {}).items()))
        print(f"{spk} = ? (votes: {votes or 'none'})")
        for ev in s.get("evidence", []):
            print(f"    seen at {ev['t']}s: {ev['name']}")


def main() -> None:
    load_dotenv()  # auto-load ./.env so `vproc` works without manually sourcing it
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "ingest":
        from vproc.ingest.pipeline import ingest_video

        n = ingest_video(args[1])
        if n == 0:
            print(f"warning: 0 segments ingested from {args[1]} (no transcript or on-screen text)",
                  file=sys.stderr)
        print(f"ingested {n} segments from {args[1]}")
    elif args[:1] == ["serve"]:
        import uvicorn

        from vproc.service import create_app

        cfg = load_config()
        uvicorn.run(create_app(), host=cfg.host, port=cfg.port)
    elif len(args) >= 2 and args[0] == "speakers":
        _speakers(args[1:])
    else:
        print("usage: vproc [ingest <video.mp4> | serve | speakers <memory> [--set 'SPEAKER_XX=Name']]")
        sys.exit(1)
