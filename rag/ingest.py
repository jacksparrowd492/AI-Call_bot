"""Ingest the Karthipuram knowledge base into ChromaDB.

Usage:  python -m rag.ingest [path/to/knowledge_base.json]

Supports THREE data shapes, because the project has had three:

  (A) legacy flat list      [{"category","intent","question","answer"}, ...]
  (B) nested brochure KB    {"project": {...}, "knowledge_base": [...]}
  (C) the authored KB       {"project": {...}, "entries": [...],
                             "escalations": [...], "smalltalk": {...}}

(C) is what data/knowledge_base.json is, it is what build_kb.py produces, and
it is now the default. It carries 39 entries where the old realestatedata.json
carried 36 - and the ones it adds are the ones callers actually asked about.
Transport and connectivity is the clearest case: "how is the transport
facility there?" had NOTHING to match against in the old file, so the bot
answered with a clarifying question forever instead of the answer it was
supposed to give.

WHAT GETS EMBEDDED, per entry:

  * one document PER QUESTION PHRASING, each carrying the entry's answer.
    Callers word things in many ways, and embedding each phrasing separately
    matches far better than embedding one blob of eleven questions.
  * one TOPIC document carrying the entry's topic and keywords, and NOTHING
    ELSE. This is what catches a caller who says a bare noun phrase -
    "transport service", "amenities", "water" - rather than a grammatical
    question. Those turns are most of a real phone call.

    Its shortness is the whole point and is not an oversight. Embedding
    distance grows with how much of a document the query does not account for,
    so "water" against sixty words of answer and key facts lands past the
    threshold and returns nothing - which is exactly what the first version of
    this file did, verified against the real vectors. The answer therefore
    rides in the metadata instead, and rag.retriever.build_context() puts it
    back when the hit is handed to the model. Short to match, complete to
    answer.

Escalations are deliberately NOT embedded. Their replies end in "may I have
your name?", and the callback flow in bridge.py asks that itself, in its own
order - having the model say it too got the caller asked twice. A question the
KB cannot answer already lands on the not-found line, which offers the callback
and waits for a yes.

Re-run this after ANY edit to data/knowledge_base.json. The vectors are a
build artefact of that file; editing the JSON alone changes nothing on a call.
"""
import json
import logging
import os
import sys
from datetime import datetime

from rag import retriever
from rag.retriever import (client, collection, embed, DB_PATH,
                           EMBED_MODEL, EMBED_DIMS)

log = logging.getLogger("jarvis.ingest")

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                            "knowledge_base.json")

# retriever.detect_intent() classifies a query as location / contact / price /
# general and filters on this metadata. Only a coarse bucket belongs here: the
# filter narrows the search, it does not answer anything, and a wrong bucket
# hides a good entry. Anything unmapped stays "general", which is never
# filtered on.
_CATEGORY_INTENT = {
    "location": "location",
    "nearby": "location",
    "contact": "contact",
    "promoter": "contact",
}


def _intent_for(category: str, entry_id: str) -> str:
    cat = (category or "").strip().lower()
    if cat in _CATEGORY_INTENT:
        return _CATEGORY_INTENT[cat]
    if any(w in (entry_id or "") for w in ("price", "cost", "rate")):
        return "price"
    return "general"


# ------------------------------------------------------------------ loading

def load_payload(path):
    """Return (entries, project). Normalizes every schema into a list of
    {id, category, intent, topic, questions[], keywords[], answer, facts[],
     source_pages[]}."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    project = {}
    if isinstance(raw, dict):
        project = raw.get("project") or {}
        records = (raw.get("entries")
                   or raw.get("knowledge_base")
                   or raw.get("records")
                   or [])
        if not records:
            raise ValueError(
                "Dict payload has no 'entries' (or 'knowledge_base' / "
                "'records') list. Top-level keys: %s" % list(raw.keys())
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
        category = (rec.get("category") or "General").strip()
        entry_id = (rec.get("id") or "").strip()
        entries.append({
            "id": entry_id,
            "category": category,
            # Shape (A) carried its own intent field; the authored KB does not,
            # so derive one.
            "intent": (rec.get("intent") or "").strip()
                      or _intent_for(category, entry_id),
            "topic": (rec.get("topic") or "").strip(),
            "questions": [str(q).strip() for q in qs if str(q).strip()],
            "keywords": [str(k).strip() for k in (rec.get("keywords") or [])
                         if str(k).strip()],
            "answer": (rec.get("answer") or "").strip(),
            "facts": rec.get("facts") or [],
            "source_pages": rec.get("source_pages") or [],
        })
    return entries, project


# ------------------------------------------------------------------ building

def project_factsheet(project):
    """Query-friendly documents for the facts that live on the project header
    rather than in an entry: where it is, and how to reach the office."""
    if not project:
        return []

    docs = []

    location = project.get("location", "") or project.get("city", "")
    site = project.get("site_location", "") or project.get("site_address", "")

    if location or site:
        docs.append({
            "text": """Category: Location
Q: Where is the project located?
Q: What is the location of Karthipuram?
Q: Where is it situated?
Q: Address of the project?
Q: Which area is the project?
Q: Where exactly?
A: %s | %s""" % (location, site),
            "meta": {"category": "Location", "intent": "location",
                     "entry_id": "factsheet_location"},
        })

    phones = project.get("phone_numbers") or []
    email = project.get("email", "")
    if phones or email:
        phone_str = ", ".join(str(p) for p in phones) if phones else ""
        docs.append({
            "text": """Category: Contact
Q: What is the contact number?
Q: How do I reach you?
Q: Phone number of the project?
Q: Email?
Q: Contact details?
A: Phone: %s | Email: %s""" % (phone_str, email),
            "meta": {"category": "Contact", "intent": "contact",
                     "entry_id": "factsheet_contact"},
        })

    lines = []
    for key, label in [
        ("name", "Project"),
        ("developer", "Developer"),
        ("founder", "Founder"),
        ("dtcp_number", "DTCP number"),
        ("rera_registration_number", "RERA registration"),
        ("type", "Type"),
        ("total_area_acres", "Total area (acres)"),
        ("promoter_office", "Promoter office"),
        ("website", "Website"),
    ]:
        val = project.get(key)
        if val:
            lines.append("%s: %s" % (label, val))

    if lines:
        docs.append({
            "text": """Category: Project Facts
Q: What are the project details?
Q: Give me project information
Q: Tell me about the project
Q: Project name and developer?
A: %s""" % " | ".join(lines),
            "meta": {"category": "Project Facts", "intent": "general",
                     "entry_id": "factsheet_project"},
        })

    return docs


def _body(entry) -> str:
    body = "A: " + entry["answer"]
    if entry["facts"]:
        body += "\nKey facts: " + "; ".join(str(f) for f in entry["facts"])
    return body


def build_documents(entries, project=None):
    """One embedded document per question phrasing, plus one per topic."""
    docs, ids, metadatas = [], [], []

    for i, s in enumerate(project_factsheet(project)):
        docs.append(s["text"])
        ids.append("factsheet_%d" % i)
        meta = dict(s["meta"])
        meta.setdefault("source_pages", "")
        meta.setdefault("question", "")
        meta["entry"] = -1 - i          # keeps dedupe_by_entry happy
        metadatas.append(meta)

    for e_i, e in enumerate(entries):
        if not e["answer"]:
            log.warning("Entry %d (%s) has no answer - skipped", e_i,
                        e["id"] or e["category"])
            continue
        if not e["questions"]:
            log.warning("Entry %d (%s) has no questions - skipped", e_i,
                        e["id"] or e["category"])
            continue

        body = _body(e)
        base_meta = {
            "category": e["category"],
            "intent": e["intent"],
            "entry": e_i,
            "entry_id": e["id"] or ("entry_%d" % e_i),
            "topic": e["topic"],
            "source_pages": ",".join(str(p) for p in e["source_pages"]),
            # Carried so build_context() can expand a short topic document into
            # a full answer without a second lookup.
            "answer": e["answer"],
            "facts": "; ".join(str(f) for f in e["facts"]),
        }

        for q_i, q in enumerate(e["questions"]):
            docs.append("Category: %s\nQ: %s\n%s" % (e["category"], q, body))
            ids.append("qa_%d_%d" % (e_i, q_i))
            meta = dict(base_meta)
            meta["question"] = q
            metadatas.append(meta)

        # The topic/keyword document. A caller who just says "transport
        # service" is not asking a question-shaped thing, and matching that
        # against "Is there a metro station inside the township?" is exactly
        # the miss that made the bot ask what they meant instead of answering.
        #
        # It carries NO answer text: a two-word query has to stay close to it,
        # and every extra word of answer pushes the distance up. The answer is
        # in the metadata, and build_context() restores it.
        if e["topic"] or e["keywords"]:
            topic_doc = "Category: %s\nTopic: %s" % (e["category"], e["topic"])
            if e["keywords"]:
                topic_doc += "\nKeywords: " + ", ".join(e["keywords"])
            docs.append(topic_doc)
            ids.append("topic_%d" % e_i)
            meta = dict(base_meta)
            meta["question"] = e["topic"]
            metadatas.append(meta)

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

    # Stamped so a call log or a startup line can say WHICH file the running
    # vectors came from. Rebuilding from a different KB used to be invisible.
    coll = client.get_or_create_collection(
        name=collection.name,
        metadata={"source": os.path.basename(path),
                  # The CONTENT of the file, not just its name. Editing an
                  # answer in the JSON and forgetting to re-run this script
                  # changes nothing on a call, and nothing anywhere said so -
                  # the answers a caller heard came from vectors built days
                  # earlier. retriever._check_source() now says so at startup.
                  "kb_sha1": retriever.kb_fingerprint(path),
                  "built_at": datetime.now().isoformat(timespec="seconds"),
                  "model": EMBED_MODEL})

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
