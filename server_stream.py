# server_stream.py
# MULTI-CAMERA ANPR BACKEND - Supports N cameras simultaneously
# AUTH: Flask session-based login added

import os
import cv2
import json
import time
import base64
import sqlite3
import threading
import secrets
from queue import Queue, Empty
from datetime import datetime
from functools import wraps
from flask import (Flask, Response, stream_with_context, jsonify,
                   request, send_file, session, redirect, url_for,
                   render_template_string)
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import easyocr
import torch
import re
import numpy as np

# ======================================================================
# CONFIG
# ======================================================================
MAX_CAMERA_PROBE = 8

CAMERA_LABELS = {
    0: "Entrance Gate",
    1: "Exit Gate",
    2: "Parking Lot A",
    3: "Parking Lot B",
}

FRAME_WIDTH        = 640
FRAME_HEIGHT       = 480
DB_PATH            = "anpr.db"
OUTPUT_DIR         = "outputs"
PLATE_DIR          = os.path.join(OUTPUT_DIR, "plate_crops")
FULL_DIR           = os.path.join(OUTPUT_DIR, "full_frames")

PLATE_MODEL_PATH   = "best.pt"
DETECT_CONF        = 0.35
OCR_CONF_THRESHOLD = 0.5
SMOOTH_CONFIRM     = 3
COOLDOWN_SECONDS   = 10

SKIP_FRAMES        = 2
JPEG_QUALITY       = 70
MAX_QUEUE_SIZE     = 100
MAX_READ_FAILURES  = 30

os.makedirs(PLATE_DIR, exist_ok=True)
os.makedirs(FULL_DIR,  exist_ok=True)

# ======================================================================
# DYNAMIC CAMERA DETECTION
# FIX: Added a release delay after probe so Windows DirectShow fully
#      frees the device before detection_loop tries to reopen it.
# ======================================================================
def probe_cameras(max_index: int = MAX_CAMERA_PROBE) -> list[int]:
    available = []
    backends = []
    if hasattr(cv2, 'CAP_DSHOW'):        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, 'CAP_V4L2'):         backends.append(cv2.CAP_V4L2)
    if hasattr(cv2, 'CAP_AVFOUNDATION'): backends.append(cv2.CAP_AVFOUNDATION)
    backends.append(None)

    print(f"[probe] Scanning camera indices 0–{max_index - 1} …")
    for idx in range(max_index):
        opened = False
        for backend in backends:
            try:
                cap = (cv2.VideoCapture(idx, backend)
                       if backend is not None
                       else cv2.VideoCapture(idx))
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    # FIX: give the OS time to release the device handle
                    time.sleep(0.3)
                    if ret:
                        available.append(idx)
                        opened = True
                        print(f"[probe]   ✓ Camera {idx} — available")
                        break
                    # release already called above
                else:
                    cap.release()
            except Exception:
                continue
        if not opened:
            print(f"[probe]   ✗ Camera {idx} — not available")

    if not available:
        print("[probe] WARNING: No cameras detected. Running in headless/demo mode.")
    else:
        print(f"[probe] Found {len(available)} camera(s): {available}")
    return available


CAMERA_INDICES: list[int] = probe_cameras()
print(f"[server] CAMERA_INDICES resolved to: {CAMERA_INDICES}")

latest_frames     = {idx: None              for idx in CAMERA_INDICES}
frame_locks       = {idx: threading.Lock()  for idx in CAMERA_INDICES}
detection_running = {idx: threading.Event() for idx in CAMERA_INDICES}
camera_stats      = {idx: {'detections': 0, 'fps': 0, 'connected': False}
                     for idx in CAMERA_INDICES}
stats_lock        = threading.Lock()
_smooth_dicts     = {idx: {}                for idx in CAMERA_INDICES}
_smooth_locks     = {idx: threading.Lock()  for idx in CAMERA_INDICES}
event_queue       = Queue(maxsize=MAX_QUEUE_SIZE)

# ======================================================================
# AUTH CONFIG
# ======================================================================
AUTH_USERNAME      = "admin"
AUTH_PASSWORD_HASH = generate_password_hash("admin123")

# ======================================================================
# FLASK
# ======================================================================
app = Flask(__name__)

_secret = os.environ.get("PLATEWISE_SECRET")
if not _secret:
    _secret = secrets.token_hex(32)
    print("[WARNING] PLATEWISE_SECRET env var not set. "
          "Sessions will be invalidated on every restart.")
app.secret_key = _secret

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5000")
CORS(app, supports_credentials=True, origins=[FRONTEND_ORIGIN, "http://localhost:5000", "http://127.0.0.1:5000"])

# ======================================================================
# CSRF
# ======================================================================
def generate_csrf_token():
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_hex(32)
    return session['_csrf_token']

def validate_csrf(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method == "POST":
            token = (request.form.get('_csrf_token')
                     or request.headers.get('X-CSRFToken', ''))
            session_token = session.get('_csrf_token', '')
            if not session_token or not secrets.compare_digest(token, session_token):
                return jsonify({"success": False, "error": "CSRF validation failed"}), 403
        return f(*args, **kwargs)
    return decorated

app.jinja_env.globals['csrf_token'] = generate_csrf_token

# ======================================================================
# AUTH DECORATOR
# ======================================================================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("logged_in"):
            return f(*args, **kwargs)
        path = request.path
        if path.startswith("/api/") or path.startswith("/stream") or path.startswith("/events"):
            return jsonify({"success": False, "error": "Unauthorized", "redirect": "/login"}), 401
        return redirect(url_for("login", next=request.path))
    return decorated

# ======================================================================
# LOGIN PAGE HTML
# ======================================================================
LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PlateWise — Login</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg:#080C14;--surface:#0D1422;--border:rgba(56,189,248,0.15);
            --border-hi:rgba(56,189,248,0.4);--accent:#38BDF8;--green:#10B981;
            --red:#EF4444;--text:#E2E8F0;--muted:#64748B;
            --mono:'JetBrains Mono',monospace;--sans:'Syne',sans-serif;
        }
        *{margin:0;padding:0;box-sizing:border-box;}
        body{font-family:var(--sans);background:var(--bg);color:var(--text);
             min-height:100vh;display:flex;align-items:center;justify-content:center;
             position:relative;overflow:hidden;}
        body::before{content:'';position:absolute;inset:0;
            background-image:linear-gradient(rgba(56,189,248,.04) 1px,transparent 1px),
            linear-gradient(90deg,rgba(56,189,248,.04) 1px,transparent 1px);
            background-size:40px 40px;animation:grid-drift 20s linear infinite;}
        @keyframes grid-drift{from{background-position:0 0}to{background-position:40px 40px}}
        body::after{content:'';position:absolute;inset:0;
            background:radial-gradient(ellipse at 50% 0%,rgba(56,189,248,.08) 0%,transparent 60%);
            pointer-events:none;}
        .login-wrap{position:relative;z-index:1;width:100%;max-width:400px;padding:20px;}
        .brand{text-align:center;margin-bottom:40px;}
        .brand-logo{font-size:36px;font-weight:800;color:var(--accent);letter-spacing:-1px;}
        .brand-logo span{color:var(--text);opacity:.5;}
        .brand-sub{font-family:var(--mono);font-size:10px;color:var(--muted);
                   letter-spacing:3px;margin-top:6px;text-transform:uppercase;}
        .card{background:var(--surface);border:1px solid var(--border);border-radius:16px;
              padding:36px 32px;box-shadow:0 0 60px rgba(56,189,248,.06),0 20px 60px rgba(0,0,0,.5);
              position:relative;overflow:hidden;}
        .card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
            background:linear-gradient(90deg,transparent,var(--accent),transparent);}
        .card-title{font-size:18px;font-weight:700;margin-bottom:6px;}
        .card-sub{font-size:13px;color:var(--muted);margin-bottom:28px;}
        .field{margin-bottom:18px;}
        label{display:block;font-size:11px;font-weight:700;color:var(--muted);
              text-transform:uppercase;letter-spacing:1.5px;margin-bottom:7px;font-family:var(--mono);}
        input[type="text"],input[type="password"]{width:100%;background:rgba(255,255,255,.04);
            border:1px solid var(--border);border-radius:8px;color:var(--text);
            font-family:var(--mono);font-size:14px;padding:12px 14px;outline:none;
            transition:border-color .2s,box-shadow .2s;}
        input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(56,189,248,.1);}
        input::placeholder{color:var(--muted);opacity:.5;}
        .btn-login{width:100%;padding:13px;background:var(--accent);color:#000;
            font-family:var(--sans);font-size:14px;font-weight:700;border:none;
            border-radius:8px;cursor:pointer;margin-top:8px;transition:filter .2s,transform .1s;
            display:flex;align-items:center;justify-content:center;gap:8px;}
        .btn-login:hover{filter:brightness(1.1);}
        .btn-login:active{transform:scale(.98);}
        .btn-login:disabled{opacity:.5;cursor:not-allowed;}
        .error-msg{display:flex;align-items:center;gap:8px;
            background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);
            border-radius:7px;padding:10px 14px;font-size:13px;color:#fca5a5;
            margin-bottom:20px;animation:shake .3s ease;}
        @keyframes shake{0%,100%{transform:translateX(0)}25%{transform:translateX(-6px)}75%{transform:translateX(6px)}}
        .error-dot{width:6px;height:6px;border-radius:50%;background:var(--red);flex-shrink:0;}
        .scan-bar{position:absolute;left:0;right:0;height:1px;
            background:linear-gradient(90deg,transparent,rgba(56,189,248,.4),transparent);
            animation:scan 4s linear infinite;pointer-events:none;}
        @keyframes scan{from{top:0}to{top:100%}}
        .footer-note{text-align:center;margin-top:20px;font-size:11px;color:var(--muted);font-family:var(--mono);}
        .cam-info{text-align:center;margin-top:16px;font-family:var(--mono);font-size:11px;color:var(--muted);}
        .cam-info span{color:var(--accent);font-weight:700;}
    </style>
</head>
<body>
<div class="login-wrap">
    <div class="brand">
        <div class="brand-logo">Plate<span>Wise</span></div>
        <div class="brand-sub">Secure ANPR System</div>
    </div>
    <div class="card">
        <div class="scan-bar"></div>
        <div class="card-title">Sign In</div>
        <div class="card-sub">Enter your credentials to access the dashboard</div>
        {% if error %}
        <div class="error-msg"><span class="error-dot"></span>{{ error }}</div>
        {% endif %}
        <form method="POST" action="/login" id="loginForm">
            <input type="hidden" name="_csrf_token" value="{{ csrf_token() }}">
            <input type="hidden" name="next" value="{{ next }}">
            <div class="field">
                <label for="username">Username</label>
                <input type="text" id="username" name="username"
                       placeholder="admin" autocomplete="username"
                       value="{{ username or '' }}" required>
            </div>
            <div class="field">
                <label for="password">Password</label>
                <input type="password" id="password" name="password"
                       placeholder="••••••••" autocomplete="current-password" required>
            </div>
            <button type="submit" class="btn-login" id="submitBtn">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/>
                    <polyline points="10 17 15 12 10 7"/><line x1="15" y1="12" x2="3" y2="12"/>
                </svg>
                Sign In
            </button>
        </form>
    </div>
    <div class="cam-info"><span>{{ cam_count }}</span> camera{{ 's' if cam_count != 1 else '' }} detected</div>
    <div class="footer-note">PlateWise ANPR &nbsp;|&nbsp; Secure Access Portal</div>
</div>
<script>
document.getElementById('loginForm').addEventListener('submit', function() {
    const btn = document.getElementById('submitBtn');
    btn.disabled = true;
    btn.innerHTML = 'Signing in…';
});
</script>
</body>
</html>"""

# ======================================================================
# AUTH ROUTES
# ======================================================================
@app.route("/login", methods=["GET", "POST"])
@validate_csrf
def login():
    if session.get("logged_in"):
        return redirect(url_for("index"))

    error    = None
    username = ""
    next_url = request.args.get("next", "/")

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        next_url = request.form.get("next", "/")

        if (username == AUTH_USERNAME and
                check_password_hash(AUTH_PASSWORD_HASH, password)):
            session.clear()
            session["logged_in"] = True
            session["username"]  = username
            session.permanent    = False
            safe_next = next_url if next_url.startswith("/") else "/"
            return redirect(safe_next)
        else:
            error = "Invalid username or password."

    return render_template_string(
        LOGIN_HTML,
        error=error,
        username=username,
        next=next_url,
        cam_count=len(CAMERA_INDICES)
    )

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ======================================================================
# DATABASE
# ======================================================================
def get_db():
    return sqlite3.connect(DB_PATH, timeout=10.0)

def init_database():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_text      TEXT    NOT NULL,
        plate_conf      REAL    NOT NULL,
        timestamp       TEXT    NOT NULL,
        camera_id       INTEGER NOT NULL DEFAULT 0,
        camera_label    TEXT    NOT NULL DEFAULT '',
        full_frame_path TEXT,
        plate_crop_path TEXT
    );""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS parking_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        plate       TEXT NOT NULL,
        entry_time  TEXT NOT NULL,
        exit_time   TEXT,
        status      TEXT NOT NULL,
        camera_id   INTEGER NOT NULL DEFAULT 0
    );""")
    for stmt in [
        "CREATE INDEX IF NOT EXISTS idx_plate_text  ON detections(plate_text)",
        "CREATE INDEX IF NOT EXISTS idx_timestamp   ON detections(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_camera      ON detections(camera_id)",
        "CREATE INDEX IF NOT EXISTS idx_pk_plate    ON parking_events(plate)",
        "CREATE INDEX IF NOT EXISTS idx_pk_status   ON parking_events(status, exit_time)",
    ]:
        c.execute(stmt)
    for alter in [
        "ALTER TABLE detections ADD COLUMN camera_id INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE detections ADD COLUMN camera_label TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE parking_events ADD COLUMN camera_id INTEGER NOT NULL DEFAULT 0",
    ]:
        try:
            c.execute(alter)
        except Exception:
            pass
    conn.commit()
    conn.close()

init_database()

# ======================================================================
# MODELS
# ======================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[server] device: {device}")

plate_model = YOLO(PLATE_MODEL_PATH)
if device == "cuda":
    plate_model.to(device)
print(f"[server] loaded plate model: {PLATE_MODEL_PATH}")

reader = easyocr.Reader(["en"], gpu=(device == "cuda"), verbose=False)
print("[server] EasyOCR initialized")

# ======================================================================
# OCR / VALIDATION
# ======================================================================
VALID_STATE_CODES = {
    'AP','AR','AS','BR','CH','DD','DL','DN','GA','GJ',
    'HP','HR','JH','JK','KA','KL','LA','LD','MH','ML',
    'MN','MP','MZ','NL','OD','PB','PY','RJ','SK','TN',
    'TR','TS','UK','UP','WB'
}
CHAR_MAP = {'O':'0','I':'1','Z':'2','S':'5','B':'8','G':'6','Q':'0','l':'1'}

def correct_plate(t):
    if not t: return ''
    t = t.upper().strip()
    t = re.sub(r'^IND\s*', '', t)
    t = t.replace(' ', '')
    t = ''.join(CHAR_MAP.get(ch, ch) for ch in t)
    t = re.sub(r'[^A-Z0-9]', '', t)
    return t

def validate_indian_plate(t):
    t = correct_plate(t)
    if len(t) < 7: return False, t
    if not any(c.isalpha() for c in t) or not any(c.isdigit() for c in t):
        return False, t
    found = any(t[i:i+2] in VALID_STATE_CODES for i in range(min(2, len(t))))
    if not found: return False, t
    first_digit = next((i for i, ch in enumerate(t) if ch.isdigit()), -1)
    if first_digit == -1: return False, t
    if sum(1 for ch in t[:first_digit] if ch.isalpha()) > 4: return False, t
    return True, t

def validate_crop(crop):
    if crop is None or crop.size == 0: return False
    h, w = crop.shape[:2]
    if h < 20 or w < 60: return False
    ar = w / h if h > 0 else 0
    return 1.8 <= ar <= 8.0

def preprocess(img):
    try:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        if np.std(g) > 30:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(g)
        return g
    except Exception:
        return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def perform_ocr(crop):
    try:
        if not validate_crop(crop): return '', 0.0
        h, w = crop.shape[:2]
        if h > 100 or w > 300:
            scale = min(100 / h, 300 / w)
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        enhanced = preprocess(crop)
        res = reader.readtext(enhanced, detail=1, paragraph=False, batch_size=1)
        if not res: return '', 0.0
        combined, best_conf = '', 0.0
        for _bbox, text, conf in res:
            combined += text + ' '
            best_conf = max(best_conf, float(conf))
        combined = combined.strip()
        if best_conf < 0.4: return '', 0.0
        is_valid, corrected = validate_indian_plate(combined)
        return (corrected, best_conf) if is_valid else ('', 0.0)
    except Exception:
        return '', 0.0

# ======================================================================
# UTILS
# ======================================================================
def encode_jpeg(frame, quality=JPEG_QUALITY):
    ok, jpg = cv2.imencode('.jpg', frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return jpg.tobytes() if ok else None

def crop_safe(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return None if x2 <= x1 or y2 <= y1 else frame[y1:y2, x1:x2]

def make_relative_path(abs_path):
    abs_output = os.path.realpath(OUTPUT_DIR)
    abs_target = os.path.realpath(abs_path)
    try:
        return os.path.relpath(abs_target, abs_output)
    except ValueError:
        return os.path.basename(abs_path)

# ======================================================================
# ASYNC DB WRITER
# ======================================================================
db_queue = Queue(maxsize=500)
_dropped_writes = 0
_dropped_lock   = threading.Lock()

def db_writer():
    global _dropped_writes
    while True:
        try:
            task = db_queue.get(timeout=1)
            if task is None:
                break
            task_type, data = task
            conn = get_db()
            c = conn.cursor()
            if task_type == 'detection':
                ts, plate, conf, full_p, crop_p, cam_id, cam_label = data
                c.execute("""INSERT INTO detections
                    (timestamp, plate_text, plate_conf,
                     full_frame_path, plate_crop_path, camera_id, camera_label)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (ts, plate, conf, full_p, crop_p, cam_id, cam_label))
                with stats_lock:
                    if cam_id in camera_stats:
                        camera_stats[cam_id]['detections'] += 1
            elif task_type == 'parking':
                plate, ts, cam_id = data
                c.execute("""SELECT id FROM parking_events
                             WHERE plate=? AND exit_time IS NULL
                             ORDER BY id DESC LIMIT 1""", (plate,))
                active = c.fetchone()
                if active:
                    c.execute("UPDATE parking_events SET exit_time=?, status='OUT' WHERE id=?",
                              (ts, active[0]))
                else:
                    c.execute("""INSERT INTO parking_events
                                 (plate, entry_time, exit_time, status, camera_id)
                                 VALUES (?, ?, NULL, 'IN', ?)""",
                              (plate, ts, cam_id))
            conn.commit()
            conn.close()
        except Empty:
            continue
        except Exception as e:
            print(f'[db_writer] error: {e}')

def enqueue_db(task_type, data):
    global _dropped_writes
    try:
        db_queue.put_nowait((task_type, data))
    except Exception:
        with _dropped_lock:
            _dropped_writes += 1
        print(f"[db_writer] WARNING: queue full — write dropped (total: {_dropped_writes})")

threading.Thread(target=db_writer, daemon=True).start()

# ======================================================================
# CAMERA OPEN
# FIX: Retry logic with delay so the OS has time to release handles
#      that probe_cameras() just freed.
# ======================================================================
def open_camera(cam_id, retries=3, delay=0.5):
    backends = []
    if hasattr(cv2, 'CAP_DSHOW'):        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, 'CAP_V4L2'):         backends.append(cv2.CAP_V4L2)
    if hasattr(cv2, 'CAP_AVFOUNDATION'): backends.append(cv2.CAP_AVFOUNDATION)
    backends.append(None)

    for attempt in range(retries):
        for backend in backends:
            try:
                cap = (cv2.VideoCapture(cam_id, backend)
                       if backend is not None
                       else cv2.VideoCapture(cam_id))
                if cap.isOpened():
                    print(f"[cam{cam_id}] Opened on attempt {attempt + 1}")
                    return cap
                cap.release()
            except Exception:
                continue
        print(f"[cam{cam_id}] Open attempt {attempt + 1} failed, retrying in {delay}s…")
        time.sleep(delay)

    return None

# ======================================================================
# PER-CAMERA DETECTION LOOP
# ======================================================================
def detection_loop(cam_id):
    label = CAMERA_LABELS.get(cam_id, f"Camera {cam_id}")
    print(f"[cam{cam_id}] Opening — {label}")

    cap = open_camera(cam_id)
    if cap is None or not cap.isOpened():
        print(f"[cam{cam_id}] ERROR: Cannot open camera {cam_id} after retries")
        detection_running[cam_id].clear()
        with stats_lock:
            camera_stats[cam_id]['connected'] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ret, _ = cap.read()
    if not ret:
        print(f"[cam{cam_id}] ERROR: Cannot read first frame")
        cap.release()
        detection_running[cam_id].clear()
        with stats_lock:
            camera_stats[cam_id]['connected'] = False
        return

    with stats_lock:
        camera_stats[cam_id]['connected'] = True
    print(f"[cam{cam_id}] Ready — {label}")

    frame_count   = 0
    fps_timer     = time.time()
    fps_frames    = 0
    read_failures = 0

    try:
        while detection_running[cam_id].is_set():
            ret, frame = cap.read()

            if not ret:
                read_failures += 1
                if read_failures >= MAX_READ_FAILURES:
                    print(f"[cam{cam_id}] Too many read failures — disconnecting")
                    break
                time.sleep(0.05)
                continue
            read_failures = 0

            frame_count += 1
            fps_frames  += 1

            # Store every other frame for the MJPEG stream
            if frame_count % 2 == 0:
                with frame_locks[cam_id]:
                    latest_frames[cam_id] = encode_jpeg(frame)

            if fps_frames >= 30:
                elapsed = time.time() - fps_timer
                with stats_lock:
                    camera_stats[cam_id]['fps'] = (
                        round(fps_frames / elapsed, 1) if elapsed > 0 else 0
                    )
                fps_timer  = time.time()
                fps_frames = 0

            if frame_count % SKIP_FRAMES != 0:
                continue

            try:
                results = list(plate_model.predict(
                    source=frame, conf=DETECT_CONF, verbose=False, stream=True))
            except Exception:
                continue

            candidates = []
            if results and getattr(results[0], 'boxes', None) is not None:
                for box in results[0].boxes:
                    try:
                        xy = box.xyxy[0]
                        x1, y1, x2, y2 = map(
                            int,
                            xy.cpu().numpy() if hasattr(xy, 'cpu') else xy
                        )
                        candidates.append((x1, y1, x2, y2, float(box.conf[0])))
                    except Exception:
                        continue

            now = time.time()

            with _smooth_locks[cam_id]:
                sm = _smooth_dicts[cam_id]

                for (x1, y1, x2, y2, _dconf) in candidates:
                    crop = crop_safe(frame, x1, y1, x2, y2)
                    if crop is None:
                        continue

                    ocr_text, ocr_conf = perform_ocr(crop)
                    if not ocr_text:
                        continue

                    entry = sm.get(ocr_text, {
                        'count': 0, 'conf': 0.0,
                        'last_seen': 0, 'last_processed': 0
                    })
                    if (now - entry['last_seen']) > 1.0:
                        entry['count'] = 0
                    entry['count']    += 1
                    entry['conf']      = max(entry['conf'], ocr_conf)
                    entry['last_seen'] = now
                    sm[ocr_text]       = entry

                    confirmed    = entry['count'] >= SMOOTH_CONFIRM
                    high_conf    = entry['conf']  >= OCR_CONF_THRESHOLD
                    off_cooldown = (now - entry['last_processed']) > COOLDOWN_SECONDS

                    if confirmed and high_conf and off_cooldown:
                        now_dt     = datetime.now()
                        ts_iso     = now_dt.isoformat()
                        ts_file    = now_dt.strftime('%Y%m%d_%H%M%S_%f')[:-3]
                        plate_name = f"cam{cam_id}_{ts_file}_{ocr_text}"
                        plate_abs  = os.path.join(PLATE_DIR, plate_name + ".jpg")
                        full_abs   = os.path.join(FULL_DIR,  plate_name + "_full.jpg")
                        plate_rel  = make_relative_path(plate_abs)
                        full_rel   = make_relative_path(full_abs)

                        crop_copy  = crop.copy()
                        frame_copy = frame.copy()
                        threading.Thread(
                            target=lambda c=crop_copy, f=frame_copy,
                                         pp=plate_abs, fp=full_abs: (
                                cv2.imwrite(pp, c), cv2.imwrite(fp, f)
                            ), daemon=True
                        ).start()

                        enqueue_db('detection', (
                            ts_iso, ocr_text, float(entry['conf']),
                            full_rel, plate_rel, cam_id, label
                        ))
                        enqueue_db('parking', (ocr_text, ts_iso, cam_id))

                        try:
                            _, buf = cv2.imencode(
                                '.jpg', crop,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                            )
                            plate_b64 = base64.b64encode(buf).decode()
                            event = {
                                'timestamp':    ts_iso,
                                'plate':        ocr_text,
                                'confidence':   float(entry['conf']),
                                'camera_id':    cam_id,
                                'camera_label': label,
                                'plate_image':  'data:image/jpeg;base64,' + plate_b64,
                                'plate_path':   plate_rel,
                                'full_path':    full_rel,
                            }
                            event_queue.put_nowait(json.dumps(event))
                        except Exception:
                            pass

                        print(f"[cam{cam_id}] Plate: {ocr_text}  conf={entry['conf']:.2f}")
                        entry['last_processed'] = now
                        entry['count']          = 0
                        sm[ocr_text]            = entry

                if len(sm) > 2000:
                    cutoff = now - 86400
                    _smooth_dicts[cam_id] = {
                        k: v for k, v in sm.items()
                        if v['last_seen'] >= cutoff or v['last_processed'] >= cutoff
                    }

    finally:
        cap.release()
        with stats_lock:
            camera_stats[cam_id]['connected'] = False
        detection_running[cam_id].clear()
        print(f"[cam{cam_id}] Thread exiting")

# ======================================================================
# MJPEG STREAM
# FIX: Generator no longer exits when detection is not running.
#      It waits (yielding nothing) so the browser connection stays open
#      and the feed appears as soon as Start is clicked.
# ======================================================================
def mjpeg_gen(cam_id):
    last = None
    while True:
        # If detection isn't running yet, wait — don't close the stream
        if not detection_running[cam_id].is_set():
            time.sleep(0.1)
            continue

        with frame_locks[cam_id]:
            fr = latest_frames[cam_id]

        if fr is None or fr is last:
            time.sleep(0.016)
            continue

        last = fr
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(fr)).encode() + b'\r\n\r\n'
               + fr + b'\r\n')

# ======================================================================
# SSE
# ======================================================================
def sse_gen():
    while True:
        try:
            d = event_queue.get(timeout=15)
            yield f"data: {d}\n\n"
        except Empty:
            yield ": keep-alive\n\n"

# ======================================================================
# ROUTES
# ======================================================================
@app.route('/')
@login_required
def index():
    csrf = generate_csrf_token()
    html_path = os.path.join(os.path.dirname(__file__), 'index.html')
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        html = html.replace(
            '<head>',
            f'<head>\n    <meta name="csrf-token" content="{csrf}">'
        )
        return Response(html, mimetype='text/html')
    except FileNotFoundError:
        return jsonify({
            'success': True,
            'message': 'PlateWise — place index.html next to server_stream.py',
            'cameras': CAMERA_INDICES,
            'camera_count': len(CAMERA_INDICES)
        })

@app.route('/stream/<int:cam_id>')
@login_required
def stream(cam_id):
    if cam_id not in CAMERA_INDICES:
        return jsonify({'error': 'Unknown camera'}), 404
    return Response(
        stream_with_context(mjpeg_gen(cam_id)),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/events')
@login_required
def events():
    return Response(stream_with_context(sse_gen()), mimetype='text/event-stream')

# ---- CSRF token endpoint ----
@app.route('/api/csrf-token', methods=['GET'])
@login_required
def api_csrf_token():
    return jsonify({'success': True, 'csrf_token': generate_csrf_token()})

# ---- Camera management ----
@app.route('/api/cameras', methods=['GET'])
@login_required
def api_cameras():
    with stats_lock:
        cams = [
            {
                'id':         idx,
                'label':      CAMERA_LABELS.get(idx, f"Camera {idx}"),
                'running':    detection_running[idx].is_set(),
                'connected':  camera_stats[idx]['connected'],
                'fps':        camera_stats[idx]['fps'],
                'detections': camera_stats[idx]['detections'],
            }
            for idx in CAMERA_INDICES
        ]
    return jsonify({'success': True, 'cameras': cams, 'total': len(cams)})

@app.route('/api/start', methods=['POST'])
@login_required
@validate_csrf
def api_start():
    try:
        data    = request.get_json(silent=True) or {}
        cam_ids = data.get('camera_ids', CAMERA_INDICES)
        cam_ids = [c for c in cam_ids if c in CAMERA_INDICES]
        if not cam_ids:
            return jsonify({'success': False,
                            'error': 'No valid camera IDs. Are cameras connected?'}), 400
        started = []
        for cam_id in cam_ids:
            if detection_running[cam_id].is_set():
                continue
            detection_running[cam_id].set()
            t = threading.Thread(target=detection_loop, args=(cam_id,), daemon=True)
            t.start()
            started.append(cam_id)
        print(f"[server] Started cameras: {started}")
        return jsonify({'success': True, 'started': started})
    except Exception as e:
        print(f"[api/start] ERROR: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
@login_required
@validate_csrf
def api_stop():
    try:
        data    = request.get_json(silent=True) or {}
        cam_ids = data.get('camera_ids', CAMERA_INDICES)
        cam_ids = [c for c in cam_ids if c in CAMERA_INDICES]
        for cam_id in cam_ids:
            detection_running[cam_id].clear()
            # Clear the last frame so the feed goes blank cleanly
            with frame_locks[cam_id]:
                latest_frames[cam_id] = None
        print(f"[server] Stopping cameras: {cam_ids}")
        return jsonify({'success': True})
    except Exception as e:
        print(f"[api/stop] ERROR: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
@login_required
def api_status():
    return jsonify({
        'success': True,
        'running': any(detection_running[i].is_set() for i in CAMERA_INDICES),
        'cameras': {i: detection_running[i].is_set() for i in CAMERA_INDICES},
        'total_cameras': len(CAMERA_INDICES)
    })

# ---- Records ----
@app.route('/api/records', methods=['GET'])
@login_required
def api_records():
    try:
        lim    = int(request.args.get('limit', 200))
        cam_id = request.args.get('camera_id', None)
        conn   = get_db()
        c      = conn.cursor()
        if cam_id is not None:
            c.execute("""SELECT * FROM detections WHERE camera_id=?
                         ORDER BY id DESC LIMIT ?""", (int(cam_id), lim))
        else:
            c.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (lim,))
        rows    = c.fetchall()
        columns = [d[0] for d in c.description]
        conn.close()
        return jsonify({'success': True,
                        'records': [dict(zip(columns, r)) for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/parking_all', methods=['GET'])
@login_required
def api_parking_all():
    try:
        conn = get_db()
        c    = conn.cursor()
        c.execute("""SELECT plate, entry_time, status, camera_id
                     FROM parking_events WHERE exit_time IS NULL
                     ORDER BY entry_time DESC""")
        inside = [{'plate': r[0], 'entry': r[1], 'status': r[2], 'camera_id': r[3]}
                  for r in c.fetchall()]
        c.execute("""SELECT plate, entry_time, exit_time, status, camera_id
                     FROM parking_events ORDER BY id DESC LIMIT 100""")
        history = [{'plate': r[0], 'entry': r[1], 'exit': r[2],
                    'status': r[3], 'camera_id': r[4]}
                   for r in c.fetchall()]
        conn.close()
        return jsonify({'success': True, 'inside': inside, 'history': history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/image/<path:filename>')
@login_required
def api_image(filename):
    try:
        abs_output = os.path.realpath(OUTPUT_DIR)
        abs_target = os.path.realpath(os.path.join(OUTPUT_DIR, filename))
        if not abs_target.startswith(abs_output + os.sep):
            return jsonify({'success': False, 'error': 'Access denied'}), 403
        if not os.path.isfile(abs_target):
            return jsonify({'success': False, 'error': 'Not found'}), 404
        return send_file(abs_target, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
@login_required
def api_stats():
    try:
        conn = get_db()
        c    = conn.cursor()
        c.execute("SELECT COUNT(*) FROM detections")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM parking_events WHERE exit_time IS NULL")
        inside = c.fetchone()[0]
        c.execute("SELECT camera_id, COUNT(*) FROM detections GROUP BY camera_id")
        per_cam = {str(r[0]): r[1] for r in c.fetchall()}
        conn.close()
        return jsonify({'success': True, 'stats': {
            'total_detections': total,
            'active_parking':   inside,
            'per_camera':       per_cam,
        }})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ======================================================================
# GLOBAL ERROR HANDLERS
# ======================================================================
@app.errorhandler(400)
def err_400(e): return jsonify({'success': False, 'error': str(e)}), 400

@app.errorhandler(401)
def err_401(e): return jsonify({'success': False, 'error': 'Unauthorized'}), 401

@app.errorhandler(403)
def err_403(e): return jsonify({'success': False, 'error': str(e)}), 403

@app.errorhandler(404)
def err_404(e): return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(500)
def err_500(e): return jsonify({'success': False, 'error': 'Internal server error', 'detail': str(e)}), 500

@app.errorhandler(Exception)
def err_unhandled(e):
    import traceback
    print(f"[server] Unhandled exception: {e}\n{traceback.format_exc()}")
    return jsonify({'success': False, 'error': str(e)}), 500

# ======================================================================
# MAIN
# ======================================================================
if __name__ == '__main__':
    print('[server] PlateWise Multi-Camera ANPR')
    print(f'[server] Detected cameras: {CAMERA_INDICES} ({len(CAMERA_INDICES)} total)')
    print(f'[server] Allowed frontend origin: {FRONTEND_ORIGIN}')
    print('[server] Open: http://127.0.0.1:5000')
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)