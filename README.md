# PlateWise ANPR Backend (YOLOv8 + EasyOCR)

This repository contains a fully optimized, stable, and production-ready ANPR backend
built using **YOLOv8** for number plate detection and **EasyOCR** for text extraction.

The system is optimized for **Indian number plates**, including:
- Strict OCR validation
- Indian state-code–based plate filtering (DL, MH, GJ, KA, etc.)
- IND-prefix removal (blue strip)
- Hallucination protection (“1N D6…”, “A8 0T…”, etc. blocked)
- Stable smoothing logic
- Parking IN/OUT system with timestamps
- Event streaming over SSE
- Live MJPEG camera stream

---

## 🚀 Features

### ✔ YOLOv8 plate detection  
Fast & accurate detection using your trained `best.pt`.

### ✔ EasyOCR for text recognition  
Strict filtering to avoid garbage OCR.

### ✔ Database (SQLite)
Stores detections and parking logs.

### ✔ Live video + events
- `/stream` — MJPEG feed  
- `/events` — Server-Sent Events for frontend

### ✔ Optimized performance
- Skip-frame logic  
- Fast JPEG encoding  
- Background DB writer thread  
- Cooldown + smoothing  

---

## 📦 Installation

### 1. Clone the repository

git clone https://github.com/AbhayCode-hub/car-number-plate-with-parking-system.git

cd car-number-plate-with-parking-system

2. Create a virtual environment

python -m venv venv

3. Activate it

For Windows

venv\Scripts\activate

For Linux/macOS

source venv/bin/activate

4. Install dependencies

pip install -r requirements.txt

▶️ Running the Backend
Make sure your best.pt model is in the root folder.

Start the server:

python server_stream.py

Backend will launch at:

http://localhost:5000

Endpoints:

/stream → Live video
/events → Plate detection events
/api/start → Start ANPR
/api/stop → Stop ANPR
/api/records → Recent detections
/api/parking_all → Parking log
/api/status → Status
/api/stats → Stats