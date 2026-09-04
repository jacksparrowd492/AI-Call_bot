"""Callback requests: when a caller asks for a human, record it and alert sales.

A "handoff" here is NOT a live transfer. The caller is told a specialist will
call them back, the request is written to SQLite so nothing is lost if the
alert fails, and the sales agent is notified by SMS.
"""
import logging
import sqlite3
import threading
from datetime import datetime

from config import settings

log = logging.getLogger("jarvis.callbacks")

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS callbacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_sid TEXT UNIQUE,
    caller_number TEXT NOT NULL,
    name TEXT,
    requirement TEXT,
    preferred_time TEXT,
    requested_at TEXT,
    caller_notified TEXT,
    agent_notified INTEGER DEFAULT 0
);
"""


def _conn():
    conn = sqlite3.connect(settings.lead_store_path)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    with _lock, _conn() as c:
        c.execute(SCHEMA)


def record(call_sid, caller_number, name=None, requirement=None,
           preferred_time=None, caller_notified=None, agent_notified=False):
    """Insert or update the callback request for this call."""
    now = datetime.now().isoformat(timespec="seconds")
    with _lock, _conn() as c:
        row = c.execute("SELECT id FROM callbacks WHERE call_sid=?",
                        (call_sid,)).fetchone()
        if row is None:
            c.execute(
                "INSERT INTO callbacks (call_sid, caller_number, name, requirement,"
                " preferred_time, requested_at, caller_notified, agent_notified)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (call_sid, caller_number, name, requirement, preferred_time,
                 now, caller_notified, int(bool(agent_notified))),
            )
        else:
            c.execute(
                """UPDATE callbacks SET
                     name = COALESCE(?, name),
                     requirement = COALESCE(?, requirement),
                     preferred_time = COALESCE(?, preferred_time),
                     caller_notified = COALESCE(?, caller_notified),
                     agent_notified = MAX(agent_notified, ?)
                   WHERE call_sid = ?""",
                (name, requirement, preferred_time, caller_notified,
                 int(bool(agent_notified)), call_sid),
            )
    log.info("Callback recorded: %s %s time=%s", caller_number, name or "",
             preferred_time or "not specified")
    return True


def pending():
    """Callback requests the sales team has not been alerted about."""
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM callbacks WHERE agent_notified=0 ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def all_callbacks():
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM callbacks ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


init()
