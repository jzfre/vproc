from pathlib import Path


def memory_directory(frames_dir, project_id: str, memory_id: str) -> Path:
    """Resolve trusted frame storage while rejecting unsafe child identifiers/links."""
    for label, value in (("project", project_id), ("memory", memory_id)):
        if (not value or len(value) > 128 or value in {".", ".."}
                or any(c in value for c in ("/", "\\"))
                or any(ord(c) < 32 or ord(c) == 127 for c in value)):
            raise ValueError(f"Invalid {label} name")
    root = Path(frames_dir).expanduser().resolve()
    directory = root / project_id / memory_id
    # Neither the project nor memory directory may redirect writes to another
    # memory or outside frame storage. The configured root itself may be a link.
    try:
        resolved = directory.resolve()
    except (OSError, RuntimeError):
        raise ValueError("Invalid memory directory") from None
    if resolved != directory:
        raise ValueError("Memory directory must not contain symlinks")
    return directory
