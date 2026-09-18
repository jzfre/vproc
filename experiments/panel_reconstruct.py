"""Multi-region panel-OCR transcript reconstruction (layout-agnostic, reliability-first).

For each sampled frame we OCR several candidate regions (right-biased vertical strips,
where meeting transcript panels dock) and keep whichever regions return valid
'name + M:SS + text' entries. Entries are deduped across all frames/regions by
(speaker, time). Robust to the app window moving/resizing mid-recording.

Uses the 9B vision model (robust), thinking disabled (prevents runaway-empty output),
generous token budget, and one retry on empty/failed OCR.
"""
import base64, json, re, subprocess, sys, tempfile, os
from openai import OpenAI

SAMPLE_EVERY = int(os.environ.get("SAMPLE_EVERY", "10"))
MODEL = os.environ.get("OCR_MODEL", "qwen3.5:9b-16k")
c = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")

PROMPT = (
    "This image may contain a meeting TRANSCRIPT PANEL: a vertical list of entries, each with a "
    "speaker name (e.g. 'SURNAME, Name'), a timestamp (M:SS or MM:SS), and spoken text (possibly "
    "multi-line). Extract every entry that has a VISIBLE speaker name. Skip partial/cut-off "
    "entries with no visible name. If the image has no transcript entries, return has_panel=false "
    'and an empty list. Return strict JSON: '
    '{"has_panel": bool, "entries": [{"speaker":"NAME","time":"M:SS","text":"..."}]}'
)
SCHEMA = {"type": "object", "properties": {"has_panel": {"type": "boolean"}, "entries": {"type": "array",
    "items": {"type": "object", "properties": {"speaker": {"type": "string"}, "time": {"type": "string"},
    "text": {"type": "string"}}, "required": ["speaker", "time", "text"]}}}, "required": ["has_panel", "entries"]}


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def dims(path):
    w, h = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height", "-of", "csv=p=0", path]).split(",")
    return int(w), int(h)


def duration(path):
    return float(_run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                       "-of", "default=nokey=1:noprint_wrappers=1", path]))


def regions(W, H):
    """Right-biased vertical strips at a few left-edges, each split into overlapping
    top/bottom halves so complete entries fall inside at least one crop regardless of
    where the (possibly resized/moved) panel sits."""
    out = []
    for frac_x in (0.45, 0.55, 0.62):
        x = int(W * frac_x)
        w = W - x
        if w < 300:
            continue
        for (fy, fh) in ((0.12, 0.55), (0.35, 0.55), (0.55, 0.44)):
            out.append((x, int(H * fy), w, int(H * fh)))
    return out


def crop(path, x, y, w, h, out_png):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
        "-vf", f"crop={w}:{h}:{x}:{y}", out_png, "-y"], check=True)


def ocr(path):
    data = base64.b64encode(open(path, "rb").read()).decode()
    for _attempt in range(2):  # one retry on empty/failure
        try:
            r = c.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=6144,
                messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}]}],
                response_format={"type": "json_schema", "json_schema": {"name": "t", "schema": SCHEMA}},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            content = r.choices[0].message.content or ""
            if content.strip():
                return json.loads(content).get("entries", [])
        except Exception:
            pass
    return []


def norm_time(s):
    m = re.match(r"(\d+):(\d{2})", (s or "").strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python panel_reconstruct.py <video>  "
                 "(env: SAMPLE_EVERY, OCR_MODEL)")
    VIDEO = sys.argv[1]
    W, H = dims(VIDEO)
    dur = duration(VIDEO)
    regs = regions(W, H)
    print(f"{W}x{H}, {dur:.0f}s, {len(regs)} regions/frame, every {SAMPLE_EVERY}s", file=sys.stderr)
    entries = {}
    with tempfile.TemporaryDirectory() as work:
        frame = os.path.join(work, "f.png")
        t = 0
        while t < dur:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss",
                str(t), "-i", VIDEO, "-frames:v", "1", frame, "-y"], check=True)
            found = 0
            for i, (x, y, w, h) in enumerate(regs):
                cpath = os.path.join(work, f"c{i}.png")
                crop(frame, x, y, w, h, cpath)
                for e in ocr(cpath):
                    secs = norm_time(e.get("time"))
                    spk = (e.get("speaker") or "").strip()
                    txt = (e.get("text") or "").strip()
                    if secs is None or not spk or not txt:
                        continue
                    key = (spk.lower(), secs)
                    if key not in entries or len(txt) > len(entries[key][2]):
                        entries[key] = (spk, secs, txt)
                        found += 1
            print(f"  t={t}s: +{found}, total {len(entries)}", file=sys.stderr)
            t += SAMPLE_EVERY
    ordered = sorted(entries.values(), key=lambda z: z[1])
    print(json.dumps([{"speaker": s, "start_ts": sec, "said_text": txt}
                      for s, sec, txt in ordered], indent=1))
    print(f"\nTOTAL {len(ordered)} entries, speakers {sorted(set(s for s,_,_ in ordered))}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
