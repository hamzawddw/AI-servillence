"""
app.py — AI Surveillance Cloud API Server
Deploy to Railway: https://railway.app
All camera processing happens on this server.
"""

import os
import sys
import threading
import base64
import time
from datetime import datetime
from flask import Flask, jsonify, request, Response, send_file
from flask_cors import CORS
import sqlite3

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY   = os.environ.get("SECRET_KEY", "ai_camera_2024")
DB_PATH      = "surveillance.db"
FACES_DIR    = "faces"
LOGS_DIR     = "logs"
UNKNOWN_DIR  = "unknown"

os.makedirs(FACES_DIR,   exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ── SQLite DB (cloud-compatible, no SQL Server needed) ────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS people (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        name    TEXT,
        age     INTEGER,
        city    TEXT,
        history TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS logs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        person_name TEXT,
        status      TEXT,
        emotion     TEXT,
        age_est     INTEGER,
        gender      TEXT,
        camera_id   INTEGER DEFAULT 0,
        screenshot  TEXT,
        detected_at TEXT DEFAULT (datetime('now'))
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS visitor_log (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        name             TEXT,
        time_in          TEXT,
        time_out         TEXT,
        duration_seconds INTEGER,
        camera_id        INTEGER DEFAULT 0
    )""")
    conn.commit()
    conn.close()

init_db()

# ── Auth middleware ───────────────────────────────────────────────────────────
def check_auth(req):
    return req.headers.get("X-API-Key") == SECRET_KEY

def auth_error():
    return jsonify({"error": "Unauthorized"}), 401

# ── Camera state (shared across threads) ──────────────────────────────────────
camera_state = {
    "running":      False,
    "last_frame":   None,
    "last_status":  "offline",
    "last_name":    "",
    "last_emotion": {},
    "fps":          0,
    "lock":         threading.Lock(),
}

# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Status ────────────────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    return jsonify({
        "status":  "online",
        "camera":  camera_state["running"],
        "version": "1.0.0",
        "time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


# ── People (known faces DB) ───────────────────────────────────────────────────
@app.route("/api/people", methods=["GET"])
def get_people():
    if not check_auth(request): return auth_error()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT * FROM people ORDER BY name").fetchall()
    conn.close()
    return jsonify([{"id":r[0],"name":r[1],"age":r[2],"city":r[3],"history":r[4]} for r in rows])

@app.route("/api/people", methods=["POST"])
def add_person():
    if not check_auth(request): return auth_error()
    d = request.json
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO people (name,age,city,history) VALUES (?,?,?,?)",
                 (d["name"], d.get("age",0), d.get("city",""), d.get("history","")))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/api/people/<int:pid>", methods=["DELETE"])
def delete_person(pid):
    if not check_auth(request): return auth_error()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM people WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


# ── Faces (upload photos) ─────────────────────────────────────────────────────
@app.route("/api/faces", methods=["GET"])
def list_faces():
    if not check_auth(request): return auth_error()
    files = [f for f in os.listdir(FACES_DIR)
             if f.lower().endswith((".jpg",".jpeg",".png",".webp"))]
    return jsonify({"faces": files})

@app.route("/api/faces/upload", methods=["POST"])
def upload_face():
    if not check_auth(request): return auth_error()
    data     = request.json
    name     = data.get("name", "unknown")
    img_b64  = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "No image provided"}), 400
    # Remove data:image/jpeg;base64, prefix if present
    if "," in img_b64:
        img_b64 = img_b64.split(",")[1]
    img_bytes = base64.b64decode(img_b64)
    # Save with index to support multiple photos per person
    existing  = [f for f in os.listdir(FACES_DIR) if f.startswith(name + "_") or f.startswith(name + ".")]
    idx       = len(existing)
    filepath  = os.path.join(FACES_DIR, f"{name}_{idx}.jpg")
    with open(filepath, "wb") as f:
        f.write(img_bytes)
    # Reload face database
    try:
        import face_db
        face_db._loaded = False
        face_db.load_faces()
    except Exception:
        pass
    return jsonify({"success": True, "saved": filepath})

@app.route("/api/faces/<filename>", methods=["DELETE"])
def delete_face(filename):
    if not check_auth(request): return auth_error()
    path = os.path.join(FACES_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"success": True})
    return jsonify({"error": "File not found"}), 404


# ── Detection Logs ────────────────────────────────────────────────────────────
@app.route("/api/logs")
def get_logs():
    if not check_auth(request): return auth_error()
    limit = request.args.get("limit", 50)
    conn  = sqlite3.connect(DB_PATH)
    rows  = conn.execute(
        "SELECT * FROM logs ORDER BY detected_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return jsonify([{
        "id": r[0], "person_name": r[1], "status": r[2],
        "emotion": r[3], "age_est": r[4], "gender": r[5],
        "camera_id": r[6], "screenshot": r[7], "detected_at": r[8]
    } for r in rows])

@app.route("/api/logs/clear", methods=["DELETE"])
def clear_logs():
    if not check_auth(request): return auth_error()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM logs")
    conn.commit()
    conn.close()
    return jsonify({"success": True})


# ── Visitor Log ───────────────────────────────────────────────────────────────
@app.route("/api/visitors")
def get_visitors():
    if not check_auth(request): return auth_error()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT * FROM visitor_log ORDER BY time_in DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return jsonify([{
        "id": r[0], "name": r[1], "time_in": r[2],
        "time_out": r[3], "duration_seconds": r[4], "camera_id": r[5]
    } for r in rows])


# ── Screenshots ───────────────────────────────────────────────────────────────
@app.route("/api/screenshot/<path:filename>")
def get_screenshot(filename):
    if not check_auth(request): return auth_error()
    path = os.path.join(LOGS_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return jsonify({"error": "Not found"}), 404


# ── Live camera frame (MJPEG stream) ──────────────────────────────────────────
def generate_stream():
    while True:
        with camera_state["lock"]:
            frame = camera_state["last_frame"]
        if frame is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.05)

@app.route("/api/stream")
def live_stream():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/frame")
def single_frame():
    """Single JPEG frame — for Flutter app polling."""
    if not check_auth(request): return auth_error()
    with camera_state["lock"]:
        frame = camera_state["last_frame"]
    if frame:
        return Response(frame, mimetype="image/jpeg")
    return jsonify({"error": "No frame available"}), 404

@app.route("/api/camera/frame", methods=["POST"])
def receive_frame():
    """
    PC camera pushes frames here.
    main.py on your PC sends frames to this endpoint.
    """
    data    = request.json
    img_b64 = data.get("frame", "")
    status  = data.get("status", "")
    name    = data.get("name", "")
    emotion = data.get("emotion", {})

    if img_b64:
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
        with camera_state["lock"]:
            camera_state["last_frame"]   = base64.b64decode(img_b64)
            camera_state["running"]      = True
            camera_state["last_status"]  = status
            camera_state["last_name"]    = name
            camera_state["last_emotion"] = emotion

    return jsonify({"ok": True})

@app.route("/api/camera/state")
def camera_state_api():
    if not check_auth(request): return auth_error()
    return jsonify({
        "running":  camera_state["running"],
        "status":   camera_state["last_status"],
        "name":     camera_state["last_name"],
        "emotion":  camera_state["last_emotion"],
    })


# ── Stats ─────────────────────────────────────────────────────────────────────
@app.route("/api/stats")
def get_stats():
    if not check_auth(request): return auth_error()
    conn   = sqlite3.connect(DB_PATH)
    total  = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    known  = conn.execute("SELECT COUNT(*) FROM logs WHERE status='known'").fetchone()[0]
    unk    = conn.execute("SELECT COUNT(*) FROM logs WHERE status='unknown'").fetchone()[0]
    spoof  = conn.execute("SELECT COUNT(*) FROM logs WHERE status='spoof'").fetchone()[0]
    people = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
    conn.close()
    return jsonify({
        "total_detections": total,
        "known":   known,
        "unknown": unk,
        "spoof":   spoof,
        "people":  people,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
