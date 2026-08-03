import lancedb


class Store:
    """Embedded LanceDB hybrid store. Table is created lazily on first add
    (so the vector dimension is taken from the data)."""

    TABLE = "segments"

    def __init__(self, path: str):
        self.db = lancedb.connect(path)

    def _table(self):
        # open_table directly; table_names() is paginated (default limit 10) and
        # would hide the table in a dir with many tables.
        try:
            return self.db.open_table(self.TABLE)
        except ValueError:
            return None

    def add(self, rows: list[dict]) -> None:
        if not rows:
            return
        table = self._table()
        if table is None:
            try:
                table = self.db.create_table(self.TABLE, data=rows)
            except ValueError:  # a concurrent ingest created it first
                table = self.db.open_table(self.TABLE)
                table.add(rows)
        else:
            table.add(rows)
        try:
            table.create_fts_index("embed_text", use_tantivy=False, replace=True)
        except Exception:
            pass  # FTS is best-effort; vector search still works

    def delete_memory(self, memory_id: str, project_id: str | None = None,
                      keep_ids: list[str] | None = None) -> None:
        """Remove all rows for a memory so re-ingesting replaces rather than duplicates.
        Scope to project_id when given so distinct videos sharing a filename stem
        don't destroy each other's rows. keep_ids spares freshly-inserted rows so a
        re-ingest adds new rows first and deletes only the stale ones — a mid-reingest
        failure then duplicates instead of destroys."""
        table = self._table()
        if table is None:
            return
        safe = memory_id.replace("'", "''")  # escape for the SQL-style filter
        cond = f"memory_id = '{safe}'"
        if project_id is not None:
            safe_proj = project_id.replace("'", "''")
            cond += f" AND project_id = '{safe_proj}'"
        if keep_ids:
            kept = ",".join("'%s'" % i.replace("'", "''") for i in keep_ids)
            cond += f" AND id NOT IN ({kept})"
        table.delete(cond)

    def update_speaker(self, memory_id: str, project_id: str | None,
                       old_speaker: str, new_speaker: str) -> None:
        """Manual speaker rename (vproc speakers --set). embed_text keeps the old label
        string by design — retrieval is unaffected; no re-embed."""
        table = self._table()
        if table is None:
            return
        safe = memory_id.replace("'", "''")  # escape for the SQL-style filter
        safe_speaker = old_speaker.replace("'", "''")
        where = f"memory_id = '{safe}' AND speaker = '{safe_speaker}'"
        if project_id is not None:
            safe_proj = project_id.replace("'", "''")
            where += f" AND project_id = '{safe_proj}'"
        table.update(where=where, values={"speaker": new_speaker})

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
