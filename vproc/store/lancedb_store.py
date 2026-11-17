import logging
import math
from numbers import Real

import lancedb
import pyarrow as pa

from vproc.errors import IndexCompatibilityError
from vproc.locking import IndexWriteLock


log = logging.getLogger(__name__)
_MODEL_KEY = b"vproc:embedding_model"
_DIMENSION_KEY = b"vproc:embedding_dimension"
_REBUILD_GUIDANCE = "Set VPROC_INDEX_PATH to a new directory and re-ingest with the configured embedding model."


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _unit_vector(vector) -> list[float]:
    """Cosine is scale invariant; normalize before Lance's float32 conversion.

    Finite Python floats can overflow/underflow in float32 or its squared norm.
    Scaling first keeps even extreme finite magnitudes searchable without changing
    direction. Existing ordinary unnormalized indexes remain compatible.
    """
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError("Each embedding vector must be a nonempty list of numbers")
    try:
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in vector):
            raise ValueError
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values):
            raise ValueError
    except (ValueError, OverflowError):
        raise ValueError("Each embedding vector must contain only finite numbers") from None
    scale = max(abs(value) for value in values)
    if not scale:
        raise ValueError("Each embedding vector must be nonzero for cosine search")
    scaled = [value / scale for value in values]
    norm = math.hypot(*scaled)
    return [value / norm for value in scaled]


def _normalized_rows(rows: list[dict]) -> list[dict]:
    out = [{**row, "vector": _unit_vector(row.get("vector"))} for row in rows]
    if len({len(row["vector"]) for row in out}) != 1:
        raise ValueError("Embedding vector dimensions must match")
    return out


class Store:
    """Embedded LanceDB hybrid store. Table is created lazily on first add
    (so the vector dimension is taken from the data)."""

    TABLE = "segments"

    def __init__(self, path: str):
        self.db = lancedb.connect(path)
        self._writer_lock = IndexWriteLock(path)

    def write_lock(self):
        """Share this reentrant lock with callers that stage a complete ingest."""
        return self._writer_lock

    def _table(self):
        # open_table directly; table_names() is paginated (default limit 10) and
        # would hide the table in a dir with many tables.
        try:
            return self.db.open_table(self.TABLE)
        except ValueError:
            return None

    @staticmethod
    def _embedding_identity(table) -> tuple[str, int] | None:
        metadata = table.schema.metadata or {}
        if _MODEL_KEY not in metadata and _DIMENSION_KEY not in metadata:
            return None
        try:
            model = metadata[_MODEL_KEY].decode("utf-8")
            dimension_text = metadata[_DIMENSION_KEY].decode("ascii")
            dimension = int(dimension_text)
            vector_type = table.schema.field("vector").type
            if (not model.strip() or dimension < 1 or str(dimension) != dimension_text
                    or not pa.types.is_fixed_size_list(vector_type)
                    or not pa.types.is_floating(vector_type.value_type)
                    or vector_type.list_size != dimension):
                raise ValueError("Invalid embedding identity")
        except (KeyError, UnicodeError, ValueError, TypeError):
            raise IndexCompatibilityError(
                f"Index embedding metadata is malformed. {_REBUILD_GUIDANCE}"
            ) from None
        return model, dimension

    @classmethod
    def _validate_embedding_model(cls, table, model: str, dimension: int | None = None) -> None:
        if not isinstance(model, str) or not model.strip():
            raise IndexCompatibilityError("The configured embedding model identifier must not be empty.")
        if dimension is not None and (type(dimension) is not int or dimension < 1):
            raise IndexCompatibilityError("The embedding dimension must be a positive integer.")
        if table is None:
            return
        identity = cls._embedding_identity(table)
        if identity is None:
            raise IndexCompatibilityError(
                f"This legacy index has no recorded embedding model identity. {_REBUILD_GUIDANCE}"
            )
        indexed_model, indexed_dimension = identity
        if model != indexed_model:
            raise IndexCompatibilityError(
                f"The configured embedding model differs from the index model. {_REBUILD_GUIDANCE}"
            )
        if dimension is not None and dimension != indexed_dimension:
            raise IndexCompatibilityError(
                f"Embedding dimension {dimension} differs from index dimension {indexed_dimension}. "
                f"{_REBUILD_GUIDANCE}"
            )

    def validate_embedding_model(self, model: str, dimension: int | None = None) -> None:
        """Validate before model work; writers must hold write_lock through their commit."""
        self._validate_embedding_model(self._table(), model, dimension)

    @classmethod
    def _validate_write(cls, table, model: str | None, dimension: int) -> None:
        if model is not None:
            cls._validate_embedding_model(table, model, dimension)
        elif table is not None and cls._embedding_identity(table) is not None:
            raise IndexCompatibilityError("An embedding model identifier is required to write this index.")

    def _create_table(self, rows, model: str | None, dimension: int):
        if model is None:
            return self.db.create_table(self.TABLE, data=rows)
        schema = pa.Table.from_pylist(rows).schema
        schema = schema.set(schema.get_field_index("vector"),
                            pa.field("vector", pa.list_(pa.float32(), dimension)))
        metadata = {_MODEL_KEY: model.encode("utf-8"), _DIMENSION_KEY: str(dimension).encode("ascii")}
        return self.db.create_table(self.TABLE, data=rows, schema=schema.with_metadata(metadata))

    @staticmethod
    def _rebuild_fts(table) -> None:
        try:
            table.create_fts_index("embed_text", use_tantivy=False, replace=True)
        except Exception:
            # Exception text may contain a transcript or query. The row commit succeeded.
            log.warning("FTS index rebuild failed; committed rows remain available through vector search.")

    def add(self, rows: list[dict], *, embedding_model: str | None = None) -> None:
        if not rows:
            return
        with self._writer_lock:
            rows = _normalized_rows(rows)
            dimension = len(rows[0]["vector"])
            table = self._table()
            self._validate_write(table, embedding_model, dimension)
            if table is None:
                try:
                    table = self._create_table(rows, embedding_model, dimension)
                except ValueError:
                    # A creator outside this process's locking protocol may race us.
                    table = self.db.open_table(self.TABLE)
                    self._validate_write(table, embedding_model, dimension)
                    table.add(rows)
            else:
                table.add(rows)
            self._rebuild_fts(table)

    def replace_memory(self, memory_id: str, project_id: str, rows: list[dict],
                       embedding_model: str) -> None:
        """Atomically replace exactly one memory's rows, then refresh optional FTS."""
        with self._writer_lock:
            if not memory_id or not project_id or not rows:
                raise ValueError("A replacement needs a memory, project, and nonempty rows")
            ids = set()
            for row in rows:
                if row.get("memory_id") != memory_id or row.get("project_id") != project_id:
                    raise ValueError("Replacement rows must belong to the requested memory and project")
                sid = row.get("id")
                if not isinstance(sid, str) or not sid.strip() or sid in ids:
                    raise ValueError("Replacement row ids must be nonempty and unique")
                ids.add(sid)
            rows = _normalized_rows(rows)
            dimension = len(rows[0]["vector"])
            table = self._table()
            self._validate_embedding_model(table, embedding_model, dimension)
            if table is None:
                table = self._create_table(rows, embedding_model, dimension)
            else:
                # merge_insert matches globally by id, so reject collisions before it
                # can update another memory or produce undefined duplicate matches.
                existing = table.search().where(
                    f"id IN ({','.join(_literal(sid) for sid in ids)})"
                ).select(["id", "memory_id", "project_id"]).limit(None).to_list()
                seen = set()
                for row in existing:
                    if (row["memory_id"] != memory_id or row["project_id"] != project_id
                            or row["id"] in seen):
                        raise ValueError("Replacement id collides with an existing or duplicated row")
                    seen.add(row["id"])
                condition = (f"memory_id = {_literal(memory_id)} "
                             f"AND project_id = {_literal(project_id)}")
                (table.merge_insert("id").when_matched_update_all()
                 .when_not_matched_insert_all()
                 .when_not_matched_by_source_delete(condition).execute(rows))
            self._rebuild_fts(table)

    def delete_memory(self, memory_id: str, project_id: str | None = None,
                      keep_ids: list[str] | None = None) -> None:
        """Remove all rows for a memory so re-ingesting replaces rather than duplicates.
        Scope to project_id when given so distinct videos sharing a filename stem
        don't destroy each other's rows. keep_ids spares freshly-inserted rows so a
        re-ingest adds new rows first and deletes only the stale ones — a mid-reingest
        failure then duplicates instead of destroys."""
        with self._writer_lock:
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
                       old_speaker: str, new_speaker: str) -> int:
        """Manual speaker rename (vproc speakers --set). embed_text keeps the old label
        string by design — retrieval is unaffected; no re-embed. Returns rows updated so
        the CLI can tell a real rename from a no-op typo."""
        with self._writer_lock:
            table = self._table()
            if table is None:
                return 0
            safe = memory_id.replace("'", "''")  # escape for the SQL-style filter
            safe_speaker = old_speaker.replace("'", "''")
            where = f"memory_id = '{safe}' AND speaker = '{safe_speaker}'"
            if project_id is not None:
                safe_proj = project_id.replace("'", "''")
                where += f" AND project_id = '{safe_proj}'"
            # Pre-query so a lancedb without UpdateResult.rows_updated can still report a count —
            # querying by old_speaker after the update would find nothing (already renamed).
            pending = table.count_rows(where)
            result = table.update(where=where, values={"speaker": new_speaker})
            rows_updated = getattr(result, "rows_updated", None)
            return rows_updated if rows_updated is not None else pending

    def vector_search(self, vector, k: int, where: str | None = None,
                      *, embedding_model: str | None = None) -> list[dict]:
        table = self._table()
        if embedding_model is not None or table is not None:
            vector = _unit_vector(vector)
        if embedding_model is not None:
            self._validate_embedding_model(table, embedding_model, len(vector))
        if table is None:
            return []
        q = table.search(vector).metric("cosine").limit(k)
        if where:
            q = q.where(where, prefilter=True)
        hits = q.to_list()
        if any(not math.isfinite(hit["_distance"]) for hit in hits):
            raise IndexCompatibilityError(
                f"Index contains invalid cosine distances. {_REBUILD_GUIDANCE}"
            )
        return hits

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
            log.warning("FTS search failed; continuing with vector results.")
            return []

    def memory_summaries(self, project_id: str = "default") -> list[dict]:
        """Catalogue metadata without loading transcript text or embeddings.

        Iterate projected batches so memory use grows with the catalogue, not the
        total number or size of transcript chunks in the library.
        """
        table = self._table()
        if table is None:
            return []
        query = (table.search().where(f"project_id = {_literal(project_id)}")
                 .select(["memory_id", "speaker", "end_ts"]).limit(None))
        summaries = {}
        for batch in query.to_batches(batch_size=1024):
            for row in batch.to_pylist():
                mid = row["memory_id"]
                summary = summaries.setdefault(mid, {
                    "memory_id": mid, "segment_count": 0, "duration_s": row["end_ts"], "speakers": set(),
                })
                summary["segment_count"] += 1
                summary["duration_s"] = max(summary["duration_s"], row["end_ts"])
                summary["speakers"].add(row["speaker"])
        return [{**summary, "speakers": sorted(summary["speakers"])}
                for _, summary in sorted(summaries.items())]

    def all_rows(self) -> list[dict]:
        """Every row's metadata, without reading embeddings from storage."""
        table = self._table()
        if table is None:
            return []
        columns = [name for name in table.schema.names if name != "vector"]
        return table.search().select(columns).limit(None).to_list()

    def memory_rows(self, memory_id: str, project_id: str | None = None) -> list[dict]:
        """Filter and project in the database so media reads don't load every meeting."""
        table = self._table()
        if table is None:
            return []
        safe = memory_id.replace("'", "''")
        where = f"memory_id = '{safe}'"
        if project_id is not None:
            safe_project = project_id.replace("'", "''")
            where += f" AND project_id = '{safe_project}'"
        columns = [name for name in table.schema.names if name != "vector"]
        return table.search().where(where).select(columns).limit(None).to_list()
