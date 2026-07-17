from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import spacy
import stanza
import re

app = Flask(__name__)
CORS(app)

#LOAD NLP MODELS

nlp_spacy = spacy.load("en_core_web_sm")

stanza.download("en")
nlp_stanza = stanza.Pipeline(
    "en",
    processors="tokenize,pos,lemma,depparse,ner"
)

#LOAD VIDEO MAP
with open("video_map.json", "r", encoding="utf-8") as f:
    VIDEO_MAP = json.load(f)
with open("videos_db.json", "r", encoding="utf-8") as f:
    USER_VIDEOS = json.load(f)

#RULE SETS
REMOVE_WORDS = {
    # articles / helpers
    "the", "a", "an",
    # be-verb / aux
    "is", "am", "are", "was", "were",
    "be", "been", "being",
    # modals / aux
    "will", "shall",
    "do", "did", "does",
    "have", "has", "had",
    # connectors 
    "to", "of", "in", "on", "at", "for",
    "that", "this",
    "very", "just", "really", "about",
    "because", "but", "while", "although",
    "from", "by", "with", "and", "or",
    # dummy
    "it",        
    "too",       
    "already",   
    "still"
}

NEG_WORDS = {"not", "never", "no", "n't", "cannot", "cant", "don't", "doesn't", "didn't"}

TIME_WORDS = {
    "yesterday", "today", "tomorrow", "morning", "evening", "night",
    "last", "next", "week", "month", "year", "now", "soon"
}

WH_WORDS = {"what", "why", "where", "who", "whom", "when", "how", "which"}

AUX_VERBS = {
    "is", "am", "are", "was", "were",
    "be", "been", "being",
    "do", "does", "did",
    "can", "could", "shall", "should", "may", "might", "will", "would",
    "must"
}

PRONOUN_MAP = {
    "i": "I",
    "me": "ME",
    "you": "YOU",
    "he": "HE",
    "she": "SHE",
    "they": "THEY",
    "them": "THEY",
    "we": "WE",
    "us": "WE"
}


LOC_WORDS = {
    "here", "there", "outside", "inside",
    "home", "school", "college", "class", "room",
    "market", "hospital", "office", "kitchen", "ground", "park"
}


MANUAL_LEMMA = {
    "went": "go",
    "gone": "go",
    "came": "come",
    "coming": "come",
    "walked": "walk",
    "walking": "walk",
    "ate": "eat",
    "eating": "eat",
    "bought": "buy",
    "buying": "buy",
    "ran": "run",
    "running": "run",
    "finished": "finish",
    "finishing": "finish",
    "completed": "complete",
    "completing": "complete"
}


MW_PHRASES = {
    "go home": ["GO", "HOME"],
    "take bath": ["BATH"],
    "feel hungry": ["HUNGRY"],
    "come here": ["COME", "HERE"]
}



def unique_preserve(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def clean_tokens(tokens):
    return [t for t in tokens if re.match(r"^[A-Za-z]+$", t) and t.lower() not in REMOVE_WORDS]

# ------------------- POST PROCESS -------------------

def post_process(gloss):
    
    fixes = {
        "GOING": "GO",
        "COMING": "COME",
        "EATING": "EAT",
        "FEELING": "FEEL"
    }
    out = []
    for g in gloss:
        g = fixes.get(g.upper(), g.upper())
        out.append(g)

    
    out = [g for g in out if g not in ("IT", "TOO")]

    return unique_preserve(out)

# MAIN GLOSS BUILDER 

def improved_build_gloss(parts):
    text = parts["raw_text"].lower()

    # Multi-word phrase detection 
    mw_gloss = []
    for phrase, mapped in MW_PHRASES.items():
        if phrase in text:
            mw_gloss.extend(mapped)

    # components
    wh = clean_tokens([w.upper() for w in parts["wh"]])
    time = clean_tokens([t.upper() for t in parts["time"]])
    loc = clean_tokens([l.upper() for l in parts["locations"]])

    sub = [PRONOUN_MAP.get(s.lower(), s.upper()) for s in parts["subjects"]]
    obj = [PRONOUN_MAP.get(o.lower(), o.upper()) for o in parts["objects"]]

    sub = clean_tokens(sub)
    obj = clean_tokens(obj)

    others = clean_tokens([o.upper() for o in parts["others"]])

    
    main_verbs = []
    raw_verb_lemmas = []
    for v in parts["verbs"]:
        lemma = MANUAL_LEMMA.get(v.lower(), v)
        raw_verb_lemmas.append(lemma.lower())
        if lemma.lower() not in AUX_VERBS:
            main_verbs.append(lemma.upper())

    
    if not main_verbs and any(v in ("do", "did", "does") for v in raw_verb_lemmas):
        main_verbs.append("DO")

    
    if parts.get("aspect_done"):
        
        if "DONE" not in main_verbs:
            main_verbs.append("DONE")

    # ISL order: TIME → LOCATION → SUBJECT → OBJECT → MW_PHRASES → OTHERS → VERB(S)
    
    if wh:
        gloss = (
            time +
            sub +
            loc +
            obj +
            mw_gloss +
            others +
            main_verbs
        )
    else:
        gloss = (
            time +
            loc +
            sub +
            obj +
            mw_gloss +
            others +
            main_verbs
        )

    # Negation / CAN rule:
    if parts["neg"] and parts.get("modal_can"):
        gloss += ["NOT", "CAN"]
    elif parts["neg"]:
        gloss.append("NOT")

    # WH at end
    gloss.extend(wh)

    # Question marker
    if parts["q"]:
        gloss.append("Q")

    return post_process(gloss)

#SCORING

def compute_structural_score(gloss, text, meta):
    if not gloss:
        return 0.0

    score = 0.0

    if meta.get("has_subject"): score += 0.10
    if meta.get("has_object"): score += 0.07
    if meta.get("has_location"): score += 0.10
    if meta.get("has_time"): score += 0.06
    if meta.get("has_verb"): score += 0.20
    if meta.get("has_neg"): score += 0.05
    if meta.get("is_question"): score += 0.05

    text_lower = text.lower()
    matches = sum(1 for g in gloss if g.lower() in text_lower)
    score += min(0.20, matches / max(1, len(gloss)) * 0.20)

    sign_matches = sum(1 for g in gloss if g.upper() in VIDEO_MAP)
    score += min(0.15, sign_matches / max(1, len(gloss)) * 0.15)

    if len(gloss) > 15:
        score -= 0.10

    return max(0.0, min(1.0, score))

# ------------------- VIDEO MAP -------------------

def map_to_videos(gloss):
    mapped = []
    missing = []

    for g in gloss:
        key = g.strip().upper()
        url = None

        # FIRST CHECK built-in dataset (video_map.json)
        if key in VIDEO_MAP:
            url = VIDEO_MAP[key]

        # SECOND CHECK Supabase user-uploaded dataset (videos_db.json)
        elif key in USER_VIDEOS:
            url = USER_VIDEOS[key]

        # THIRD TRY PARTIAL MATCH: match keywords like WALK → WALK FAST
        if not url:
            for k, v in VIDEO_MAP.items():
                if key in k.upper().split():
                    url = v
                    break

        # FOURTH TRY PARTIAL MATCH IN USER VIDEOS
        if not url:
            for k, v in USER_VIDEOS.items():
                if key in k.upper().split():
                    url = v
                    break

        # FIFTH IF  STILL not found → missing
        if url:
            mapped.append({"gloss": key, "url": url})
        else:
            missing.append(key)

    return mapped, missing


#  SPACY MODEL

def spacy_model(text):
    doc = nlp_spacy(text)

    subjects, objects, locations, verbs = [], [], [], []
    others, time_tokens, wh_tokens = [], [], []
    has_neg = False
    is_question = text.strip().endswith("?")
    aspect_done = False
    modal_can = False

    for tok in doc:
        lemma_raw = tok.lemma_.lower()
        lemma = MANUAL_LEMMA.get(lemma_raw, tok.lemma_)
        dep = tok.dep_
        pos = tok.pos_
        ent = tok.ent_type_

        # WH
        if lemma.lower() in WH_WORDS:
            wh_tokens.append(lemma)
            is_question = True

        # NEG
        if lemma.lower() in NEG_WORDS or tok.text.lower() in NEG_WORDS:
            has_neg = True

        # aspect "DONE"
        if lemma.lower() in {"already", "finish", "complete", "done", "finished", "completed"}:
            aspect_done = True

        # CAN modal
        if lemma.lower() in {"can", "could"} or tok.text.lower() in {"cannot", "cant"}:
            modal_can = True

        # TIME
        if ent in ("DATE", "TIME") or lemma.lower() in TIME_WORDS:
            time_tokens.append(lemma)
            continue

        # LOCATION: entity, location words, or pobj
        if ent in ("GPE", "LOC") or lemma.lower() in LOC_WORDS or dep == "pobj":
            locations.append(lemma)
            continue

        # SUBJECT / OBJECT
        if dep in ("nsubj", "nsubjpass"):
            subjects.append(lemma)
            continue

        if dep in ("obj", "dobj", "iobj"):
            objects.append(lemma)
            continue

        # VERB
        if pos in ("VERB", "AUX"):
            verbs.append(lemma)
            continue

        # OTHERS
        if pos in ("NOUN", "ADJ", "ADV") and lemma.lower() not in REMOVE_WORDS:
            others.append(lemma)

    parts = {
        "raw_text": text,
        "wh": wh_tokens,
        "time": time_tokens,
        "locations": locations,
        "subjects": subjects,
        "objects": objects,
        "others": others,
        "verbs": verbs,
        "neg": has_neg,
        "q": is_question,
        "aspect_done": aspect_done,
        "modal_can": modal_can
    }

    gloss = improved_build_gloss(parts)

    meta = {
        "has_subject": bool(subjects),
        "has_object": bool(objects),
        "has_location": bool(locations),
        "has_time": bool(time_tokens),
        "has_verb": bool(verbs),
        "has_neg": has_neg,
        "is_question": is_question
    }

    return gloss, compute_structural_score(gloss, text, meta)

# ------------------- STANZA MODEL -------------------

def stanza_model(text):
    doc = nlp_stanza(text)

    subjects, objects, locations, verbs = [], [], [], []
    others, time_tokens, wh_tokens = [], [], []
    has_neg = False
    is_question = text.strip().endswith("?")
    aspect_done = False
    modal_can = False

    for sent in doc.sentences:
        for w in sent.words:
            lemma_raw = w.lemma.lower()
            lemma = MANUAL_LEMMA.get(lemma_raw, w.lemma)
            dep = w.deprel
            upos = w.upos
            word_text = w.text.lower()

            # WH
            if lemma.lower() in WH_WORDS:
                wh_tokens.append(lemma)
                is_question = True

            # NEG
            if lemma.lower() in NEG_WORDS or word_text in NEG_WORDS:
                has_neg = True

            # aspect "DONE"
            if lemma.lower() in {"already", "finish", "complete", "done", "finished", "completed"}:
                aspect_done = True

            # CAN modal
            if lemma.lower() in {"can", "could"} or word_text in {"cannot", "cant"}:
                modal_can = True

            # TIME
            if lemma.lower() in TIME_WORDS:
                time_tokens.append(lemma)
                continue

            # LOCATION
            if dep.startswith("obl") or lemma.lower() in LOC_WORDS:
                locations.append(lemma)
                continue

            # SUBJECT / OBJECT
            if dep.startswith("nsubj"):
                subjects.append(lemma)
                continue

            if dep in ("obj", "iobj"):
                objects.append(lemma)
                continue

            # VERB
            if upos in ("VERB", "AUX"):
                verbs.append(lemma)
                continue

            # OTHERS
            if upos in ("NOUN", "ADJ", "ADV") and lemma.lower() not in REMOVE_WORDS:
                others.append(lemma)

    parts = {
        "raw_text": text,
        "wh": wh_tokens,
        "time": time_tokens,
        "locations": locations,
        "subjects": subjects,
        "objects": objects,
        "others": others,
        "verbs": verbs,
        "neg": has_neg,
        "q": is_question,
        "aspect_done": aspect_done,
        "modal_can": modal_can
    }

    gloss = improved_build_gloss(parts)

    meta = {
        "has_subject": bool(subjects),
        "has_object": bool(objects),
        "has_location": bool(locations),
        "has_time": bool(time_tokens),
        "has_verb": bool(verbs),
        "has_neg": has_neg,
        "is_question": is_question
    }

    return gloss, compute_structural_score(gloss, text, meta)

# ------------------- API ROUTE -------------------

@app.route("/nlp_models", methods=["POST"])
def nlp_models():
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    models = [
        ("SpaCy", spacy_model),
        ("Stanza", stanza_model)
    ]

    results = []
    best = None

    for name, fn in models:
        try:
            gloss, score = fn(text)
        except Exception:
            gloss, score = [], 0.0

        videos, missing = map_to_videos(gloss)

        info = {
            "model": name,
            "gloss": gloss,
            "score": score,
            "score_percent": round(score * 100, 2),
            "videos": videos,
            "missing": missing
        }
        results.append(info)

        if best is None or score > best["score"]:
            best = info

    return jsonify({
        "text": text,
        "results": results,
        "best": best
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
