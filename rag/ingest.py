"""Ingest the Karthipuram knowledge base into ChromaDB.

Usage:  python -m rag.ingest [path/to/realestatedata.json]

Supports BOTH data shapes:

  (A) legacy flat list      [{"category","intent","question","answer"}, ...]
  (B) nested brochure KB    {"project": {...},
                             "knowledge_base": [{"category","intent",
                                                 "questions":[...], "answer",
                                                 "facts":[...], "source_pages":[...]}]}

For shape (B) one document is embedded PER QUESTION (all sharing the entry's
answer). Callers phrase things in many ways; embedding each phrasing separately
matches far better than embedding one blob of nine questions.
"""
import json
import logging
import os
import sys

from rag import retriever
from rag.retriever import (client, collection, embed, DB_PATH,
                           EMBED_MODEL, EMBED_DIMS)

log = logging.getLogger("jarvis.ingest")

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "realestatedata.json")


# ------------------------------------------------------------------ loading

def load_payload(path):
    """Return (entries, project). Normalizes both schemas into a list of
    {category, intent, questions[], answer, facts[], source_pages[]}."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    project = {}
    if isinstance(raw, dict):
        project = raw.get("project") or {}
        records = raw.get("knowledge_base") or raw.get("records") or []
        if not records:
            raise ValueError(
                "Dict payload has no 'knowledge_base' (or 'records') list. "
                "Top-level keys: %s" % list(raw.keys())
            )
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("Unsupported JSON root type: %s" % type(raw).__name__)

    entries = []
    for rec in records:
        if not isinstance(rec, dict):
            log.warning("Skipping non-object record: %r", rec)
            continue
        qs = rec.get("questions")
        if qs is None:
            q = rec.get("question")
            qs = [q] if q else []
        elif isinstance(qs, str):
            qs = [qs]
        entries.append({
            "category": (rec.get("category") or "General").strip(),
            "intent": (rec.get("intent") or "").strip(),
            "questions": [str(q).strip() for q in qs if str(q).strip()],
            "answer": (rec.get("answer") or "").strip(),
            "facts": rec.get("facts") or [],
            "source_pages": rec.get("source_pages") or [],
        })
    return entries, project


# ------------------------------------------------------------------ building

def project_factsheet(project):
    """One extra document holding the hard project facts (RERA, DTCP, phone,
    address). These get asked verbatim on calls and must never be paraphrased."""
    if not project:
        return None
    lines = []
    for key, label in [
        ("name", "Project"), ("type", "Type"), ("location", "Location"),
        ("site_location", "Site address"), ("total_area_acres", "Total area (acres)"),
        ("developer", "Developer"), ("promoter_office", "Promoter office"),
        ("email", "Email"), ("website", "Website"),
        ("dtcp_number", "DTCP number"), ("rera_registration_number", "RERA registration"),
    ]:
        val = project.get(key)
        if val:
            lines.append("%s: %s" % (label, val))
    phones = project.get("phone_numbers") or []
    if phones:
        lines.append("Phone: %s" % ", ".join(str(p) for p in phones))
    if not lines:
        return None
    return ("Category: Project Fact Sheet\n"
            "Q: What are the official project details, approvals and contact information?\n"
            "A: " + " | ".join(lines))


def build_documents(entries, project=None):
    """One embedded document per question phrasing."""
    docs, ids, metadatas = [], [], []

    sheet = project_factsheet(project)
    if sheet:
        docs.append(sheet)
        ids.append("factsheet")
        metadatas.append({"category": "Project Fact Sheet",
                          "intent": "project_factsheet", "entry": -1})

    for e_i, e in enumerate(entries):
        if not e["answer"]:
            log.warning("Entry %d (%s) has no answer - skipped", e_i, e["intent"])
            continue
        if not e["questions"]:
            log.warning("Entry %d (%s) has no questions - skipped", e_i, e["intent"])
            continue

        body = "A: " + e["answer"]
        if e["facts"]:
            body += "\nKey facts: " + "; ".join(str(f) for f in e["facts"])

        for q_i, q in enumerate(e["questions"]):
            docs.append("Category: %s\nQ: %s\n%s" % (e["category"], q, body))
            ids.append("qa_%d_%d" % (e_i, q_i))
            # Chroma metadata must be str/int/float/bool - no lists.
            metadatas.append({
                "category": e["category"],
                "intent": e["intent"],
                "entry": e_i,
                "question": q,
                "source_pages": ",".join(str(p) for p in e["source_pages"]),
            })

    return docs, ids, metadatas


# ------------------------------------------------------------------ main

def main(path):
    if not os.path.exists(path):
        log.error("Data file not found: %s", path)
        sys.exit(1)

    entries, project = load_payload(path)
    n_q = sum(len(e["questions"]) for e in entries)
    log.info("Loaded %d knowledge entries (%d question phrasings) from %s",
             len(entries), n_q, path)

    docs, ids, metadatas = build_documents(entries, project)
    if not docs:
        log.warning("No usable records found - nothing to ingest.")
        return

    # Reset the collection for a clean rebuild.
    try:
        client.delete_collection(collection.name)
        log.info("Dropped existing collection '%s'", collection.name)
    except Exception as e:
        log.warning("Could not delete existing collection: %s", e)

    coll = client.get_or_create_collection(name=collection.name)

    # delete_collection() invalidated the handle that rag.retriever imported at
    # module load, so anything still holding it would raise NotFoundError on the
    # next query. Re-point the module at the new collection. Harmless when
    # ingest runs as its own process; essential when it is called in-process.
    retriever.collection = coll

    log.info("Embedding %d documents with %s (dims=%s) - this may take a minute...",
             len(docs), EMBED_MODEL, EMBED_DIMS or "native")
    embeddings = embed(docs)
    if len(embeddings) != len(docs):
        raise RuntimeError("Embedded %d vectors for %d documents - aborting "
                           "rather than writing a misaligned index."
                           % (len(embeddings), len(docs)))
    log.info("Embedded %d vectors of %d dimensions", len(embeddings),
             len(embeddings[0]) if embeddings else 0)

    # Chroma rejects very large single add() batches; chunk to be safe.
    B = 500
    for i in range(0, len(docs), B):
        coll.add(ids=ids[i:i + B], documents=docs[i:i + B],
                 embeddings=embeddings[i:i + B], metadatas=metadatas[i:i + B])

    log.info("Ingested %d documents from %d entries into '%s' at %s (count=%d)",
             len(docs), len(entries), collection.name, DB_PATH, coll.count())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)
