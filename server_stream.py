# server_stream.py
# OPTIMIZED ANPR BACKEND - Reduced lag, improved performance

import os
import cv2
import json
import time
import base64
import sqlite3
import threading
from queue import Queue, Empty
from datetime import datetime
from flask import Flask, Response, stream_with_context, jsonify, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
import easyocr
import torch
import re
import numpy as np

# ---------------- CONFIG ----------------
CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DB_PATH = "anpr.db"
OUTPUT_DIR = "outputs"
PLATE_DIR = os.path.join(OUTPUT_DIR, "plate_crops")
FULL_DIR = os.path.join(OUTPUT_DIR, "full_frames")

PLATE_MODEL_PATH = "best.pt"
DETECT_CONF = 0.35
OCR_CONF_THRESHOLD = 0.5
SMOOTH_CONFIRM = 3
COOLDOWN_SECONDS = 10

# Performance optimizations
SKIP_FRAMES = 2  # Process every Nth frame for detection
JPEG_QUALITY = 70  # Lower quality = faster encoding
MAX_QUEUE_SIZE = 50  # Prevent memory bloat

os.makedirs(PLATE_DIR, exist_ok=True)
os.makedirs(FULL_DIR, exist_ok=True)

# ---------------- FLASK ----------------
app = Flask(__name__)
CORS(app)
event_queue = Queue(maxsize=MAX_QUEUE_SIZE)
latest_frame_jpeg = None
latest_frame_lock = threading.Lock()
detection_running = threading.Event()

# ---------------- DATABASE ----------------
def get_db_connection():
    """Thread-safe database connections"""
    return sqlite3.connect(DB_PATH, timeout=10.0)

def init_database():
    """Initialize database with optimizations"""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_text TEXT NOT NULL,
        plate_conf REAL NOT NULL,
        timestamp TEXT NOT NULL,
        full_frame_path TEXT,
        plate_crop_path TEXT
    );
    """)
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS parking_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT NOT NULL,
        entry_time TEXT NOT NULL,
        exit_time TEXT,
        status TEXT NOT NULL
    );
    """)
    
    # Add indexes for faster queries
    c.execute("CREATE INDEX IF NOT EXISTS idx_plate_text ON detections(plate_text)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_parking_plate ON parking_events(plate)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_parking_status ON parking_events(status, exit_time)")
    
    conn.commit()
    conn.close()

init_database()

# ---------------- MODELS ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[server] device: {device}")

# Load model with optimizations
plate_model = YOLO(PLATE_MODEL_PATH)
if device == "cuda":
    plate_model.to(device)
print(f"[server] loaded plate model: {PLATE_MODEL_PATH}")

# Initialize OCR once (expensive operation)
reader = easyocr.Reader(["en"], gpu=(device == 'cuda'), verbose=False)
print("[server] EasyOCR initialized")

# ---------------- OCR UTILS (CORRECTED) ----------------

# Valid Indian state codes
VALID_STATE_CODES = {
    'AP', 'AR', 'AS', 'BR', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ', 
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 
    'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 
    'TR', 'TS', 'UK', 'UP', 'WB'
}

CHAR_MAP = {'O':'0','I':'1','Z':'2','S':'5','B':'8','G':'6','Q':'0','l':'1'}

def correct_plate(t):
    """Corrects OCR errors and removes only genuine IND prefix"""
    if not t:
        return ''
    
    t = t.upper().strip()
    
    # CRITICAL: Remove IND prefix BEFORE character mapping
    # Otherwise 'IND' becomes '1ND' after mapping and doesn't get removed
    t = re.sub(r'^IND\s*', '', t)
    
    # Remove spaces
    t = t.replace(' ', '')
    
    # Apply character corrections AFTER IND removal
    t = ''.join(CHAR_MAP.get(ch, ch) for ch in t)
    
    # Keep only alphanumeric characters
    t = re.sub(r'[^A-Z0-9]', '', t)
    
    return t

def validate_indian_plate(t):
    """
    More lenient Indian number plate validator.
    Returns: (is_valid: bool, cleaned_text: str)
    """
    t = correct_plate(t)
    
    # Minimum length check
    if len(t) < 7:
        return False, t
    
    # Must have both letters and digits
    has_alpha = any(c.isalpha() for c in t)
    has_digit = any(c.isdigit() for c in t)
    if not (has_alpha and has_digit):
        return False, t
    
    # More lenient: check if ANY valid state code appears in first 3 characters
    # This handles OCR errors in the first character
    found_valid_state = False
    for i in range(min(2, len(t))):
        if t[i:i+2] in VALID_STATE_CODES:
            found_valid_state = True
            break
    
    if not found_valid_state:
        return False, t
    
    # Relaxed pattern check: max 4 letters before digits
    # (was 2, now 4 to handle OCR reading errors)
    first_digit_pos = -1
    for i, ch in enumerate(t):
        if ch.isdigit():
            first_digit_pos = i
            break
    
    if first_digit_pos == -1:  # No digits found
        return False, t
    
    # Count letters before first digit
    letters_before_digit = sum(1 for ch in t[:first_digit_pos] if ch.isalpha())
    if letters_before_digit > 4:
        return False, t
    
    return True, t

def validate_crop_dimensions(crop):
    """
    Validates crop aspect ratio and minimum size.
    Returns: True if valid, False otherwise
    """
    if crop is None or crop.size == 0:
        return False
    
    h, w = crop.shape[:2]
    
    # Relaxed minimum size check (60x20 pixels)
    if h < 20 or w < 60:
        return False
    
    # Relaxed aspect ratio check (1.8 to 8.0) - allows more camera angles
    aspect_ratio = w / h if h > 0 else 0
    if aspect_ratio < 1.8 or aspect_ratio > 8.0:
        return False
    
    return True

def preprocess(img):
    """Optimized preprocessing - fewer operations"""
    try:
        if len(img.shape) == 3:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            g = img
        
        # Skip denoising if image is already decent quality
        if np.std(g) > 30:  # Has enough contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(g)
        return g
    except Exception:
        return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def perform_ocr(crop):
    """
    Optimized OCR with lenient Indian plate validation.
    Returns: (plate_text: str, confidence: float)
    """
    try:
        # Validate crop dimensions and aspect ratio
        if not validate_crop_dimensions(crop):
            return '', 0.0
        
        # Resize if too large (faster OCR)
        h, w = crop.shape[:2]
        if h > 100 or w > 300:
            scale = min(100/h, 300/w)
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # Preprocess and run OCR
        enhanced = preprocess(crop)
        res = reader.readtext(enhanced, detail=1, paragraph=False, batch_size=1)
        
        if not res:
            return '', 0.0

        # Combine all detected text
        combined = ''
        best_conf = 0.0
        for _bbox, text, conf in res:
            combined += text + ' '
            best_conf = max(best_conf, float(conf))
        
        combined = combined.strip()
        
        # Lowered confidence threshold for better detection (was 0.5)
        if best_conf < 0.4:
            return '', 0.0

        # Validate as Indian plate
        is_valid, corrected = validate_indian_plate(combined)
        
        if not is_valid:
            # Debug: print rejected plates to see what's being filtered
            # print(f"[OCR] Rejected: {combined} -> {corrected}")
            return '', 0.0
        
        return corrected, best_conf
        
    except Exception as e:
        return '', 0.0

# ---------------- UTILS ----------------
def encode_jpeg(frame):
    """Faster JPEG encoding with lower quality"""
    ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return jpg.tobytes() if ok else None

def crop_safe(frame, x1, y1, x2, y2):
    """Optimized cropping"""
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

# ---------------- ASYNC DB WRITER ----------------
db_write_queue = Queue(maxsize=100)

def db_writer_thread():
    """Dedicated thread for database writes to prevent blocking"""
    while True:
        try:
            task = db_write_queue.get(timeout=1)
            if task is None:  # Shutdown signal
                break
            
            task_type, data = task
            conn = get_db_connection()
            c = conn.cursor()
            
            if task_type == 'detection':
                ts_iso, plate, conf, full_path, plate_path = data
                c.execute("""INSERT INTO detections
                            (timestamp, plate_text, plate_conf, full_frame_path, plate_crop_path)
                            VALUES (?, ?, ?, ?, ?)""",
                        (ts_iso, plate, conf, full_path, plate_path))
            
            elif task_type == 'parking':
                plate, ts_iso = data
                c.execute("SELECT id FROM parking_events WHERE plate = ? AND exit_time IS NULL ORDER BY id DESC LIMIT 1", (plate,))
                active = c.fetchone()
                
                if active:
                    c.execute("UPDATE parking_events SET exit_time = ?, status = 'OUT' WHERE id = ?", (ts_iso, active[0]))
                else:
                    c.execute("INSERT INTO parking_events (plate, entry_time, exit_time, status) VALUES (?, ?, NULL, 'IN')", (plate, ts_iso))
            
            conn.commit()
            conn.close()
            
        except Empty:
            continue
        except Exception as e:
            print(f'[db_writer] error: {e}')

# Start DB writer thread
db_thread = threading.Thread(target=db_writer_thread, daemon=True)
db_thread.start()

# ---------------- DETECTION LOOP (OPTIMIZED) ----------------
def detection_loop(camera_index=CAMERA_INDEX):
    global latest_frame_jpeg

    print(f"[server] Opening camera {camera_index}")
    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(int(camera_index))

    if not cap.isOpened():
        for idx in range(5):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                camera_index = idx
                print(f'[server] Found camera at index {idx}')
                break

    if not cap.isOpened():
        print('[server] ERROR: Cannot open any camera')
        detection_running.clear()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag

    ret, test_frame = cap.read()
    if not ret:
        print('[server] ERROR: Cannot read frames')
        cap.release()
        detection_running.clear()
        return

    print('[server] Camera initialized successfully')

    # Smoothing dictionary
    if not hasattr(detection_loop, '_smooth'):
        detection_loop._smooth = {}
    sm = detection_loop._smooth

    frame_count = 0
    
    try:
        while detection_running.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1

            # Update MJPEG stream every frame (for smooth video)
            if frame_count % 2 == 0:  # Update stream every 2 frames
                with latest_frame_lock:
                    latest_frame_jpeg = encode_jpeg(frame)

            # Process detection only every Nth frame
            if frame_count % SKIP_FRAMES != 0:
                continue

            # Run plate detection
            try:
                results = plate_model.predict(source=frame, conf=DETECT_CONF, verbose=False, stream=True)
                results = list(results)
            except Exception as e:
                continue

            candidates = []
            if results and len(results) > 0 and getattr(results[0], 'boxes', None) is not None:
                for box in results[0].boxes:
                    try:
                        xy = box.xyxy[0]
                        if hasattr(xy, 'cpu'):
                            x1, y1, x2, y2 = map(int, xy.cpu().numpy())
                        else:
                            x1, y1, x2, y2 = map(int, xy)
                        conf = float(box.conf[0])
                        candidates.append((x1, y1, x2, y2, conf))
                    except Exception:
                        continue
            
            now = time.time()

            # Process candidates
            for (x1, y1, x2, y2, dconf) in candidates:
                crop = crop_safe(frame, x1, y1, x2, y2)
                if crop is None:
                    continue

                ocr_text, ocr_conf = perform_ocr(crop)
                if not ocr_text:
                    continue

                key = ocr_text
                entry = sm.get(key, {'count': 0, 'conf': 0.0, 'last_seen_ts': 0, 'last_processed_ts': 0})

                # Reset count if not seen recently
                if (now - entry['last_seen_ts']) > 1.0:
                    entry['count'] = 0
                
                entry['count'] += 1
                entry['conf'] = max(entry['conf'], ocr_conf)
                entry['last_seen_ts'] = now
                sm[key] = entry

                # Check for processing
                is_confirmed = entry['count'] >= SMOOTH_CONFIRM
                is_high_conf = entry['conf'] >= OCR_CONF_THRESHOLD
                is_off_cooldown = (now - entry['last_processed_ts']) > COOLDOWN_SECONDS

                if is_confirmed and is_high_conf and is_off_cooldown:
                    now_dt = datetime.now()
                    ts_iso = now_dt.isoformat()
                    ts_filename = now_dt.strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    
                    plate_name = f"{ts_filename}_{key}"
                    plate_path = os.path.join(PLATE_DIR, plate_name + ".jpg")
                    full_path = os.path.join(FULL_DIR, plate_name + "_full.jpg")

                    # Save images asynchronously
                    threading.Thread(target=lambda: (
                        cv2.imwrite(plate_path, crop),
                        cv2.imwrite(full_path, frame)
                    ), daemon=True).start()

                    # Queue database writes (non-blocking)
                    try:
                        db_write_queue.put_nowait(('detection', (ts_iso, key, float(entry['conf']), full_path, plate_path)))
                        db_write_queue.put_nowait(('parking', (key, ts_iso)))
                    except:
                        pass

                    # Send SSE event
                    try:
                        _, buf = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        plate_b64 = base64.b64encode(buf).decode('utf-8')
                        event = {
                            'timestamp': ts_iso,
                            'plate': key,
                            'confidence': float(entry['conf']),
                            'plate_image': 'data:image/jpeg;base64,' + plate_b64,
                            'plate_path': plate_path,
                            'full_path': full_path
                        }
                        event_queue.put_nowait(json.dumps(event))
                    except:
                        pass

                    print(f"[server] Processed plate {key}")
                    entry['last_processed_ts'] = now
                    entry['count'] = 0
                    sm[key] = entry

            # Cleanup old entries
            if len(sm) > 2000:
                cutoff = now - 86400
                sm = {k: v for k, v in sm.items() if v['last_seen_ts'] >= cutoff or v['last_processed_ts'] >= cutoff}
                detection_loop._smooth = sm

    finally:
        cap.release()
        print('[server] detection thread exiting')

# ---------------- STREAM ----------------
def mjpeg_stream():
    """Optimized MJPEG streaming"""
    global latest_frame_jpeg
    last_frame = None
    
    while True:
        with latest_frame_lock:
            fr = latest_frame_jpeg
        
        if fr is None or fr == last_frame:
            time.sleep(0.01)
            continue
        
        last_frame = fr
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(fr)).encode() + b'\r\n\r\n' + fr + b'\r\n')

# ---------------- SSE ----------------
def sse_events():
    while True:
        try:
            d = event_queue.get(timeout=15)
            yield f"data: {d}\n\n"
        except Empty:
            yield ": keep-alive\n\n"

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return jsonify({
        'success': True,
        'message': 'PlateWise ANPR Backend (Optimized)',
        'stream': '/stream',
        'events': '/events'
    })

@app.route('/stream')
def stream():
    return Response(stream_with_context(mjpeg_stream()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/events')
def events():
    return Response(stream_with_context(sse_events()), mimetype='text/event-stream')

@app.route('/api/start', methods=['POST'])
def api_start():
    if detection_running.is_set():
        return jsonify({'success': False, 'message': 'Already running'})

    detection_running.set()
    t = threading.Thread(target=detection_loop, args=(CAMERA_INDEX,), daemon=True)
    t.start()
    print('[server] Detection started')
    return jsonify({'success': True, 'message': 'Detection started'})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    detection_running.clear()
    print('[server] Detection stopping')
    return jsonify({'success': True, 'message': 'Stopping detection'})

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({'success': True, 'running': detection_running.is_set()})

@app.route('/api/records', methods=['GET'])
def api_records():
    try:
        lim = int(request.args.get('limit', 100))
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (lim,))
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description]
        conn.close()
        records = [dict(zip(columns, row)) for row in rows]
        return jsonify({'success': True, 'records': records})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/parking_all', methods=['GET'])
def api_parking_all():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT plate, entry_time, status FROM parking_events WHERE exit_time IS NULL ORDER BY entry_time DESC")
        inside_rows = cursor.fetchall()
        inside = [{'plate': row[0], 'entry': row[1], 'status': row[2]} for row in inside_rows]

        cursor.execute("SELECT plate, entry_time, exit_time, status FROM parking_events ORDER BY id DESC LIMIT 100")
        history_rows = cursor.fetchall()
        history = [{'plate': row[0], 'entry': row[1], 'exit': row[2], 'status': row[3]} for row in history_rows]
        
        conn.close()

        return jsonify({'success': True, 'inside': inside, 'history': history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/image/<path:filename>')
def api_image(filename):
    try:
        if os.path.exists(filename):
            return send_file(filename, mimetype='image/jpeg')
        return jsonify({'success': False, 'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM detections")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM parking_events WHERE exit_time IS NULL")
        inside = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {'total_detections': total, 'active_parking': inside}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ---------------- START SERVER ----------------
if __name__ == '__main__':
    print('[server] PlateWise ANPR System (Optimized)')
    print('[server] Press Ctrl+C to stop')
    print('[server] Frontend: http://localhost:5000')
    
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)