"""
backend/memory/store.py
Private per-patient storage (SQLite): prescription text and search history.

The browser holds a random token in an HttpOnly cookie; only its SHA-256 is stored here, so the
database alone cannot be used to impersonate a patient. Every query is scoped by patient id.
The uploaded PDF itself is never written to disk - only its extracted text.
"""

import hashlib
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (id TEXT PRIMARY KEY, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT, patient TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  filename TEXT NOT NULL, pages INTEGER NOT NULL, text TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, patient TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  query TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL, UNIQUE(patient, query));
CREATE INDEX IF NOT EXISTS idx_doc_patient ON documents(patient);
CREATE INDEX IF NOT EXISTS idx_hist_patient ON history(patient, created);
"""


def patient_id(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class PatientStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or config.PRIVATE_DB_PATH)
        self._ready = False

    @contextmanager
    def _db(self):
        if not self._ready:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)
        con = sqlite3.connect(self.path, timeout=10)
        try:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            con.execute("PRAGMA secure_delete = ON")  # overwrite deleted medical text on disk
            if not self._ready:
                con.executescript(_SCHEMA)
                os.chmod(self.path, 0o600)
                self._ready = True
            with con:
                yield con
        finally:
            con.close()

    @staticmethod
    def _ensure(con, pid: str) -> None:
        con.execute("INSERT OR IGNORE INTO patients(id, created) VALUES (?, ?)", (pid, time.time()))

    # --- prescriptions ---------------------------------------------------
    def add_document(self, pid: str, filename: str, pages: int, text: str) -> int:
        with self._db() as con:
            self._ensure(con, pid)
            return con.execute("INSERT INTO documents(patient, filename, pages, text, created) VALUES (?,?,?,?,?)",
                               (pid, filename, pages, text, time.time())).lastrowid

    def documents(self, pid: str) -> List[Dict[str, Any]]:
        with self._db() as con:
            return [dict(r) for r in con.execute(
                "SELECT id, filename, pages, text, created FROM documents WHERE patient=? ORDER BY created DESC", (pid,))]

    def delete_document(self, pid: str, doc_id: int) -> bool:
        with self._db() as con:
            return con.execute("DELETE FROM documents WHERE patient=? AND id=?", (pid, doc_id)).rowcount > 0

    # --- search history --------------------------------------------------
    def add_search(self, pid: str, query: str, status: str) -> None:
        with self._db() as con:
            self._ensure(con, pid)
            con.execute("INSERT INTO history(patient, query, status, created) VALUES (?,?,?,?) "
                        "ON CONFLICT(patient, query) DO UPDATE SET status=excluded.status, created=excluded.created",
                        (pid, query, status, time.time()))
            con.execute("DELETE FROM history WHERE patient=? AND id NOT IN "
                        "(SELECT id FROM history WHERE patient=? ORDER BY created DESC, id DESC LIMIT ?)",
                        (pid, pid, config.HISTORY_LIMIT))

    def history(self, pid: str, limit: int = config.HISTORY_LIMIT) -> List[Dict[str, Any]]:
        with self._db() as con:
            return [dict(r) for r in con.execute(
                "SELECT id, query, status, created FROM history WHERE patient=? ORDER BY created DESC, id DESC LIMIT ?", (pid, limit))]

    def delete_search(self, pid: str, item_id: int) -> bool:
        with self._db() as con:
            return con.execute("DELETE FROM history WHERE patient=? AND id=?", (pid, item_id)).rowcount > 0

    def clear_history(self, pid: str) -> int:
        with self._db() as con:
            return con.execute("DELETE FROM history WHERE patient=?", (pid,)).rowcount

    def delete_patient(self, pid: str) -> None:
        with self._db() as con:
            con.execute("DELETE FROM patients WHERE id=?", (pid,))
