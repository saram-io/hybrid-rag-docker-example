import logging
import pyarrow as pa
import lancedb
from app.config import settings

logger = logging.getLogger(__name__)
_db = None
_table = None

def get_table():
    global _db, _table
    if _table is None:
        _db = lancedb.connect(settings.lancedb_uri)
        if settings.table_name in _db.table_names():
            _table = _db.open_table(settings.table_name)
        else:
            _table = None
    return _table

def init_table(dimension: int):
    global _db, _table
    _db = lancedb.connect(settings.lancedb_uri)
    if settings.table_name in _db.table_names():
        _table = _db.open_table(settings.table_name)
        logger.info("Opened table '%s' (%d rows)", settings.table_name, _table.count_rows())
    else:
        schema = pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=dimension)),
        ])
        _table = _db.create_table(settings.table_name, schema=schema)
        logger.info("Created table '%s' with dim=%d", settings.table_name, dimension)
    return _table

def add_records(records: list[dict]):
    global _table
    if _table is None:
        raise RuntimeError("Table not initialized")
    _table.add(records)
