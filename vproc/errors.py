class IndexCompatibilityError(ValueError):
    """The configured embedding model cannot safely use this index."""


class IndexBusyError(RuntimeError):
    """Another writer currently owns the index or its frame storage."""
