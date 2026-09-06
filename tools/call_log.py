"""One row per call, in an Excel sheet the sales team can just open.

leads.db already holds this, but nobody on a sales floor opens SQLite. This
writes the same facts to .xlsx: who called, how long they talked, what they
asked about, and - when they accepted a callback - when to ring them back, in
IST as well as their own time.

Two things this has to survive, because both happen every day:

  * The sheet being OPEN in Excel when a call ends. Windows locks the file and
    openpyxl cannot save. The row is appended to a sidecar CSV instead and
    merged in on the next successful write, so a call is never lost just
    because someone was looking at the spreadsheet.
  * openpyxl not being installed. Logging a warning is correct; taking the call
    down is not.
"""
import csv
import logging
import os
import threading
from datetime import datetime

from config import settings

log = logging.getLogger("jarvis.call_log")

_lock = threading.Lock()

COLUMNS = [
    ("Call date (IST)",     18),
    ("Start (IST)",         12),
    ("End (IST)",           12),
    ("Duration (min)",      14),
    ("Caller name",         20),
    ("Phone",               18),
    ("Country",             9),
    ("Caller timezone",     20),
    ("Interested in",       28),
    ("Call summary",        60),
    ("Intent score",        12),
    ("Callback?",           11),
    ("Asked for",           22),
    ("Callback (caller local)", 24),
    ("Callback (IST)",      24),
    ("Confirmed via",       14),
    ("Sales team alerted",  18),
    ("Brochure sent",       14),
    ("Call SID",            36),
]

_HEADERS = [c[0] for c in COLUMNS]


def _path():
    return settings.call_log_path


def _sidecar():
    return _path() + ".pending.csv"


def _row_values(rec: dict):
    return [rec.get(k) for k in (
        "call_date", "start_time", "end_time", "duration_min", "name", "phone",
        "country", "caller_timezone", "requirement", "summary", "intent_score",
        "callback", "asked_for", "callback_local", "callback_ist",
        "confirmed_via", "agent_alerted", "brochure_sent", "call_sid")]


# ------------------------------------------------------------------ helpers
def _new_workbook(openpyxl):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Calls"
    ws.append(_HEADERS)
    bold = openpyxl.styles.Font(bold=True)
    fill = openpyxl.styles.PatternFill("solid", fgColor="DDEBF7")
    for i, (name, width) in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=i)
        cell.font = bold
        cell.fill = fill
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = width
    ws.freeze_panes = "A2"                 # header stays put while scrolling
    ws.auto_filter.ref = "A1:%s1" % openpyxl.utils.get_column_letter(len(COLUMNS))
    return wb


def _stash(rec):
    """Excel has the file locked. Keep the row next to it rather than lose it."""
    path = _sidecar()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if not exists:
                w.writerow(_HEADERS)
            w.writerow(_row_values(rec))
        log.warning("Call log is locked (open in Excel?) - row parked in %s",
                    os.path.basename(path))
        return True
    except Exception as e:
        log.error("Could not park the call row either: %s", e)
        return False


def _drain(ws):
    """Fold any parked rows back in, oldest first."""
    path = _sidecar()
    if not os.path.exists(path):
        return 0
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
    except Exception as e:
        log.error("Could not read the parked rows: %s", e)
        return 0

    added = 0
    for row in rows:
        if row and row[0] == _HEADERS[0]:          # skip the header line
            continue
        if any(c for c in row):
            ws.append(row)
            added += 1
    if added:
        try:
            os.remove(path)
        except Exception:
            log.warning("Merged %d parked rows but could not delete %s", added, path)
    return added


# -------------------------------------------------------------------- write
def append(rec: dict) -> bool:
    """Append one call. Returns True when it reached the sheet or the sidecar."""
    try:
        import openpyxl
    except ImportError:
        log.error("openpyxl is not installed - no Excel call log will be written. "
                  "Fix with: pip install openpyxl")
        return _stash(rec)

    path = _path()
    with _lock:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            if os.path.exists(path):
                wb = openpyxl.load_workbook(path)
                ws = wb["Calls"] if "Calls" in wb.sheetnames else wb.active
            else:
                wb = _new_workbook(openpyxl)
                ws = wb["Calls"]

            merged = _drain(ws)
            ws.append(_row_values(rec))
            wb.save(path)
        except PermissionError:
            return _stash(rec)
        except Exception as e:
            log.error("Could not write the Excel call log (%s): %s",
                      type(e).__name__, e)
            return _stash(rec)

    if merged:
        log.info("Merged %d parked row(s) back into the call log", merged)
    log.info("Call logged: %s %s duration=%s min callback=%s",
             rec.get("phone"), rec.get("name") or "", rec.get("duration_min"),
             rec.get("callback_ist") or rec.get("callback"))
    return True


def build_record(**kw) -> dict:
    """Normalise a call into the row shape, so bridge.py stays readable."""
    started = kw.get("started_at")
    ended = kw.get("ended_at") or datetime.now()
    duration = kw.get("duration_min")
    if duration is None and started:
        duration = round((ended - started).total_seconds() / 60.0, 1)

    return {
        "call_date":       started.strftime("%Y-%m-%d") if started else "",
        "start_time":      started.strftime("%H:%M:%S") if started else "",
        "end_time":        ended.strftime("%H:%M:%S") if ended else "",
        "duration_min":    duration if duration is not None else "",
        "name":            kw.get("name") or "",
        "phone":           kw.get("phone") or "",
        "country":         kw.get("country") or "",
        "caller_timezone": kw.get("caller_timezone") or "",
        "requirement":     kw.get("requirement") or "",
        "summary":         kw.get("summary") or "",
        "intent_score":    kw.get("intent_score") if kw.get("intent_score") is not None else "",
        "callback":        "Yes" if kw.get("callback") else "No",
        "asked_for":       kw.get("asked_for") or "",
        "callback_local":  kw.get("callback_local") or "",
        "callback_ist":    kw.get("callback_ist") or "",
        "confirmed_via":   kw.get("confirmed_via") or "",
        "agent_alerted":   "Yes" if kw.get("agent_alerted") else "No",
        "brochure_sent":   "Yes" if kw.get("brochure_sent") else "No",
        "call_sid":        kw.get("call_sid") or "",
    }
