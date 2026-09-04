"""SQLite lead store - one row per captured lead, joinable to Twilio recordings by call_sid."""
import sqlite3
import threading
from datetime import datetime

from config import settings

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_sid TEXT UNIQUE,
    caller_number TEXT,
    name TEXT,
    requirement TEXT,
    intent_score INTEGER DEFAULT 0,
    whatsapp_sent INTEGER DEFAULT 0,
    handoff INTEGER DEFAULT 0,
    transcript TEXT DEFAULT '',
    created_at TEXT
);
"""


def _conn():
    conn = sqlite3.connect(settings.lead_store_path)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    with _lock, _conn() as c:
        c.execute(SCHEMA)


def upsert_lead(call_sid, caller_number, name=None, requirement=None,
                intent_score=0, whatsapp_sent=False, handoff=False,
                transcript=""):
    """Insert or merge a lead. Later calls enrich the same row (same call_sid)."""
    now = datetime.utcnow().isoformat(timespec="seconds")
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM leads WHERE call_sid=?", (call_sid,)).fetchone()
        if row is None:
            c.execute(
                "INSERT INTO leads (call_sid, caller_number, name, requirement, intent_score,"
                " whatsapp_sent, handoff, transcript, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (call_sid, caller_number, name, requirement, int(intent_score or 0),
                 int(bool(whatsapp_sent)), int(bool(handoff)), transcript, now),
            )
        else:
            c.execute(
                """UPDATE leads SET
                     name = COALESCE(?, name),
                     requirement = COALESCE(?, requirement),
                     intent_score = MAX(intent_score, ?),
                     whatsapp_sent = MAX(whatsapp_sent, ?),
                     handoff = MAX(handoff, ?),
                     transcript = CASE WHEN length(?) > length(transcript) THEN ? ELSE transcript END
                   WHERE call_sid = ?""",
                (name, requirement, int(intent_score or 0), int(bool(whatsapp_sent)),
                 int(bool(handoff)), transcript, transcript, call_sid),
            )
    return True


def all_leads():
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM leads ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


init()
