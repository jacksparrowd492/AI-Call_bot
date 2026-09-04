"""Assemble data/knowledge_base.json from the authored parts.

Run:  python build_kb.py
Kept in the repo so the KB can be re-derived and validated, not hand-edited blind.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

PROJECT = {
    "name": "Karthipuram",
    "tagline": "Step through the door to your dream place",
    "type": "Integrated township / plotted development",
    "city": "Coimbatore, Tamil Nadu",
    "site_address": ("Avinashi Road, Neelambur, Kathir Engineering College Road, "
                     "Coimbatore; Kalapatti to Arasur SH Road"),
    "total_area_acres": 190,
    "developer": "Unnamalai Promoters Pvt Ltd (Unnamalai Group)",
    "founder": "Thiru P. Karthikeyan",
    "promoter_office": ("No.583/2B1, Poongothai Nagar, Goldwins, "
                        "Coimbatore-641014, Tamilnadu"),
    "phone_numbers": ["+91 9418 646464", "+91 77780 77790"],
    "email": "info@karthipuram.com",
    "website": "www.unnamalaipromoters.com",
    "dtcp_number": "100/2023",
    "rera_registration_number": "TN/11/Layout/3352/2025",
    "rera_verify_url": "https://rera.tn.gov.in",
}

# Deterministic, zero-latency replies. These never touch the LLM or the RAG.
SMALLTALK = {
    "greeting": {
        "patterns": ["hello", "hi", "hii", "hey", "hello?", "hi there", "hey there",
                     "good morning", "good afternoon", "good evening", "vanakkam",
                     "hello sir", "hello madam", "yes hello"],
        "reply": ("Hello! I can help you with details about Karthipuram, our 190-acre "
                  "township in Coimbatore. What would you like to know?"),
    },
    # ONLY an explicit goodbye may hang up. "No, thank you" is a caller
    # declining ONE thing, not ending the call - treating it as a farewell hung
    # up on a live caller mid-conversation.
    "farewell": {
        "patterns": ["bye", "goodbye", "good bye", "bye bye", "thank you bye",
                     "thanks bye", "ok bye", "okay bye", "alright bye",
                     "thank you goodbye"],
        "reply": "Thank you for calling Karthipuram. Have a wonderful day!",
        "end_call": True,
    },
    # Declining, or signalling they are nearly done - confirm before hanging up.
    "decline": {
        "patterns": ["no thanks", "no thank you", "that's all", "thats all",
                     "nothing else", "no", "nope", "not now", "not interested",
                     "im good", "i'm good", "that's it", "thats it"],
        "reply": ("No problem. Is there anything else about the township I can "
                  "help with before you go?"),
    },
    "thanks": {
        "patterns": ["thank you", "thanks", "thank you so much", "thanks a lot",
                     "super", "great", "nice", "good", "ok thanks", "okay thanks"],
        "reply": "Happy to help! Anything else you'd like to know about the township?",
    },
    "acknowledge": {
        "patterns": ["ok", "okay", "hmm", "yeah", "yes", "yep", "right", "sure",
                     "got it", "understood", "fine", "alright", "mm", "mhm"],
        "reply": "Sure. What else would you like to know?",
    },
    "audio_check": {
        "patterns": ["can you hear me", "hello are you there", "are you there",
                     "hello hello", "is anyone there", "can you hear",
                     "are you listening", "still there"],
        "reply": "Yes, I can hear you clearly. Please go ahead.",
    },
    "identity": {
        "patterns": ["who are you", "what is your name", "who am i speaking to",
                     "are you a robot", "are you human", "are you a machine",
                     "what are you"],
        "reply": ("I'm Jarvis, the project assistant for Karthipuram. I can tell you "
                  "about the township, the amenities and the location. What would you "
                  "like to know?"),
    },
    "wellbeing": {
        "patterns": ["how are you", "how are you doing", "how do you do",
                     "hope you are well"],
        "reply": "I'm doing well, thank you for asking! How can I help you with Karthipuram?",
    },
    "repeat": {
        "patterns": ["what", "sorry", "pardon", "come again", "say that again",
                     "repeat that", "can you repeat", "i didn't hear",
                     "i didn't catch that", "once more"],
        "reply": None,  # handled specially: re-speak the previous reply
        "action": "repeat_last",
    },
    "wait": {
        "patterns": ["one minute", "hold on", "wait", "just a moment", "one second",
                     "hold please", "give me a minute"],
        "reply": "Of course, take your time. I'm here.",
    },
}


def _load(name):
    path = os.path.join(DATA, name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    entries = _load("kb_part1.json") + _load("kb_part2.json")
    escalations = _load("kb_escalation.json")

    # ---- validation ------------------------------------------------------
    problems = []
    seen = set()
    for e in entries:
        if e["id"] in seen:
            problems.append("duplicate entry id: %s" % e["id"])
        seen.add(e["id"])
        if not e.get("questions"):
            problems.append("%s has no questions" % e["id"])
        if not e.get("answer"):
            problems.append("%s has no answer" % e["id"])
        if e.get("status") not in ("confirmed", "proposed"):
            problems.append("%s has bad status %r" % (e["id"], e.get("status")))
        words = len(e.get("answer", "").split())
        if words > 65:
            problems.append("%s answer is %d words (too long to speak)" % (e["id"], words))

    for e in escalations:
        if e["id"] in seen:
            problems.append("duplicate escalation id: %s" % e["id"])
        seen.add(e["id"])
        if not e.get("reply"):
            problems.append("%s has no reply" % e["id"])
        if not e.get("questions"):
            problems.append("%s has no questions" % e["id"])

    if problems:
        print("KB VALIDATION FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    payload = {
        "schema_version": 2,
        "project": PROJECT,
        "smalltalk": SMALLTALK,
        "entries": entries,
        "escalations": escalations,
        "source_note": ("Derived from the Karthipuram brochure (24 pages). Items the "
                        "brochure does not state are routed to escalations rather than "
                        "invented."),
    }

    out = os.path.join(DATA, "knowledge_base.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    n_q = sum(len(e["questions"]) for e in entries)
    n_eq = sum(len(e["questions"]) for e in escalations)
    n_sp = sum(len(v["patterns"]) for v in SMALLTALK.values())
    print("knowledge_base.json written")
    print("  knowledge entries : %d  (%d question phrasings)" % (len(entries), n_q))
    print("  escalation topics : %d  (%d question phrasings)" % (len(escalations), n_eq))
    print("  smalltalk intents : %d  (%d patterns)" % (len(SMALLTALK), n_sp))
    print("  proposed-status entries: %s"
          % ", ".join(e["id"] for e in entries if e["status"] == "proposed"))


if __name__ == "__main__":
    main()
