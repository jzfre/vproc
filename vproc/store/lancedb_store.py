import lancedb


class Store:
    """Embedded LanceDB hybrid store. Table is created lazily on first add
    (so the vector dimension is taken from the data)."""

    TABLE = "segments"

    def __init__(self, path: str):
        self.db = lancedb.connect(path)

    def _table(self):
        if self.TABLE in self.db.table_names():
            return self.db.open_table(self.TABLE)
        return None

    def add(self, rows: list[dict]) -> None:
        if not rows:
            return
        table = self._table()
        if table is None:
            table = self.db.create_table(self.TABLE, data=rows)
        else:
            table.add(rows)
        try:
            table.create_fts_index("embed_text", use_tantivy=False, replace=True)
        except Exception:
            pass  # FTS is best-effort; vector search still works

    def vector_search(self, vector, k: int, where: str | None = None) -> list[dict]:
        table = self._table()
        if table is None:
            return []
        q = table.search(vector).metric("cosine").limit(k)
        if where:
            q = q.where(where, prefilter=True)
        return q.to_list()

    def fts_search(self, text: str, k: int, where: str | None = None) -> list[dict]:
        table = self._table()
        if table is None:
            return []
        try:
            q = table.search(text, query_type="fts").limit(k)
            if where:
                q = q.where(where, prefilter=True)
            return q.to_list()
        except Exception:
            return []
