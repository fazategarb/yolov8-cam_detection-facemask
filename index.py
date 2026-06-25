import base64
import os
import sys
import time
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# Inisialisasi FastAPI
app = FastAPI(title="YOLOv8 Mask Detection Backend")

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Buat folder 'sample' otomatis
if not os.path.exists("sample"):
    os.makedirs("sample")

# Variabel global anti-spam untuk endpoint HTTP POST
last_saved_time = 0.0

# Load model YOLOv8 kustom
MODEL_PATH = "model/best.pt"
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model YOLOv8 kustom berhasil dimuat!")
    print(f"📊 Kelas yang terdeteksi di model: {model.names}") 
    # Ini akan memunculkan {0: 'Incorrect_Mask', 1: 'with_Mask', 2: 'without_Mask'} di terminal
except Exception as e:
    print(f"❌ GAGAL MEMUAT MODEL! Periksa file di '{MODEL_PATH}'. Sistem dihentikan.\nDetail: {e}")
    sys.exit(1)


# Fungsi pembantu untuk memproses YOLO di background thread
def predict_frame(frame):
    results = model(frame, stream=True, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Mengambil nama asli dari model
            raw_label = model.names[cls]
            
            # Ubah ke huruf kecil dan format konsisten agar frontend mudah membaca
            # 'with_Mask' -> 'with_mask', 'without_Mask' -> 'without_mask', 'Incorrect_Mask' -> 'incorrect_mask'
            label = raw_label.lower().replace(" ", "_")

            detections.append(
                {"bbox": [x1, y1, x2, y2], "confidence": conf, "class": label}
            )
    return detections


# WEB-SOCKET ENDPOINT (REAL-TIME STREAM)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🤝 Frontend React-TS terhubung ke saluran WebSocket")

    try:
        while True:
            data = await websocket.receive_text()

            if not data:
                continue

            encoded_data = data.split(",")[1] if "," in data else data
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                detections = await run_in_threadpool(predict_frame, frame)
                await websocket.send_json({"status": "success", "data": detections})

    except WebSocketDisconnect:
        print("🛑 Koneksi WebSocket diputus oleh Frontend.")
    except Exception as e:
        print(f"⚠️ Terjadi kesalahan pada WebSocket: {e}")


# API ENDPOINT (HTTP POST SAVE SAMPLE)
class SnapshotRequest(BaseModel):
    image: str


@app.post("/api/sample")
async def save_sample(request: SnapshotRequest):
    global last_saved_time
    current_time = time.time()

    # Batasi penyimpanan maksimal 1 foto per 2 detik (Anti-Spam)
    if current_time - last_saved_time < 2.0:
        return {
            "status": "ignored",
            "message": "Request terlalu cepat (Anti-Spam aktif). Gambar diabaikan.",
        }

    try:
        encoded_data = request.image.split(",")[1] if "," in request.image else request.image
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            total_files = len(os.listdir("sample"))
            filename = f"sample/snapshot_{total_files + 1}.jpg"

            await run_in_threadpool(cv2.imwrite, filename, frame)
            last_saved_time = current_time

            return {
                "status": "success",
                "message": f"Gambar sampel berhasil disimpan sebagai {filename}!",
            }

        return {"status": "error", "message": "Format gambar tidak valid."}

    except Exception as e:
        return {"status": "error", "message": f"Gagal menyimpan: {str(e)}"}