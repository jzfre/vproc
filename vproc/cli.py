import sys

from vproc.config import load_config, load_dotenv


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
    else:
        print("usage: vproc [ingest <video.mp4> | serve]")
        sys.exit(1)
