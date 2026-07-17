
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
import uuid
import json
import os
import re

app = Flask(__name__)
CORS(app)

# ---------------- SUPABASE CONFIG ----------------
SUPABASE_URL = "SUPABASE_URL"
SUPABASE_KEY = "SUPABASE_KEY"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET = "isl-videos"

# ---------------- JSON DB (NEW FORMAT) ----------------
DB_FILE = "videos_db.json"

# Load clean dictionary (KEY = GLOSS, VALUE = URL)
try:
    with open(DB_FILE, "r", encoding="utf8") as f:
        DB = json.load(f)
except:
    DB = {}   # { "GO": "url", "EAT": "url", ... }


def save_db():
    # Sort keys alphabetically for clean JSON
    sorted_db = dict(sorted(DB.items()))
    with open(DB_FILE, "w", encoding="utf8") as f:
        json.dump(sorted_db, f, indent=4, ensure_ascii=False)


# ---------------- UPLOAD VIDEO ----------------
@app.post("/upload_video")
def upload_video():
    name = request.form.get("name", "").strip()
    desc = request.form.get("description", "")
    file = request.files.get("video")

    if not name or not file:
        return jsonify({"success": False, "error": "Missing fields"}), 400

    # Clean gloss name
    gloss = name.upper().replace(" ", "_")

    # sanitize filename
    original = file.filename
    safe_original = re.sub(r'[^a-zA-Z0-9._-]', '_', original)

    # unique ID prefix
    video_id = str(uuid.uuid4())[:8]
    filename = f"{video_id}_{safe_original}"

    # convert file -> bytes
    file_bytes = file.read()

    # upload to supabase
    try:
        supabase.storage.from_(BUCKET).upload(
            path=filename,
            file=file_bytes,
            file_options={"content-type": file.content_type}
        )
    except Exception as e:
        print("UPLOAD ERROR:", e)
        return jsonify({"success": False, "error": "Upload failed"}), 500

    # public URL
    public_url = supabase.storage.from_(BUCKET).get_public_url(filename)

    # SAVE IN NEW JSON STYLE
    DB[gloss] = public_url
    save_db()

    return jsonify({
        "success": True,
        "video_id": video_id,
        "gloss": gloss,
        "url": public_url
    })


# ---------------- GET VIDEO BY GLOSS ----------------
@app.get("/get_video")
def get_video():
    gloss = request.args.get("id", "").strip().upper()

    if gloss not in DB:
        return jsonify({"success": False})

    return jsonify({
        "success": True,
        "gloss": gloss,
        "url": DB[gloss]
    })


# ---------------- LIST ALL VIDEOS ----------------
@app.get("/list_videos")
def list_videos():
    videos = []

    for gloss, url in DB.items():
        videos.append({
            "gloss": gloss,
            "url": url
        })

    return jsonify({
        "success": True,
        "videos": videos
    })


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
