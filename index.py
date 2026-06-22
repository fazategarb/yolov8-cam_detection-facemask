import base64
import os
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# Inisialisasi FastAPI
app = FastAPI(title="YOLOv8 Mask Detection Backend")

# Konfigurasi CORS
# Agar Frontend Vite (biasanya port 5173) diizinkan mengakses Backend FastAPI (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ],  # Mengizinkan frontend lokal
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Buat folder 'sample' otomatis jika belum ada di lokal
if not os.path.exists("sample"):
    os.makedirs("sample")

# Load model YOLOv8 kustom milik temanmu
try:
    model = YOLO("model/best.pt")
    print("Model YOLOv8 kustom berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model. Periksa jalur 'model/best.pt'. Detail: {e}")


# web-socket config
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Frontend React terhubung ke saluran WebSocket")

    try:
        while True:
            # Menerima string Base64 dari React
            data = await websocket.receive_text()

            # Prapemrosesan: Bersihkan header Base64 jika ada (e.g., 'data:image/jpeg;base64,')
            encoded_data = data.split(",")[1] if "," in data else data

            # Konversi string Base64 menjadi format matriks gambar OpenCV
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                # Jalankan inferensi YOLOv8 pada frame tersebut
                results = model(frame, stream=True)
                detections = []

                for r in results:
                    for box in r.boxes:
                        # Ambil koordinat boks [x1, y1, x2, y2]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])  # Skor akurasi (0.0 - 1.0)
                        cls = int(box.cls[0])  # Indeks kelas
                        label = model.names[cls]  # Nama kelas (cth: 'mask', 'no-mask')

                        # Bungkus ke dalam list dictionary
                        detections.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "confidence": conf,
                                "class": label,
                            }
                        )

                # E. Kirim kembali hasil koordinat dalam format JSON ke React
                await websocket.send_json({"status": "success", "data": detections})

    except WebSocketDisconnect:
        print("Koneksi WebSocket diputus oleh Frontend.")
    except Exception as e:
        print(f"Terjadi kesalahan pada WebSocket: {e}")

# API endpoint
# Skema data (pydantic) untuk memvalidasi data masuk dari HTTP POST
class SnapshotRequest(BaseModel):
    image: str  # Frontend wajib mengirim JSON berupa {"image": "string_base64"}


@app.post("/api/save-sample")
async def save_sample(request: SnapshotRequest):
    try:
        # Bersihkan string Base64
        encoded_data = (
            request.image.split(",")[1]
            if "," in request.image
            else request.image
        )

        # Konversi menjadi gambar OpenCV
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            # Beri nama file unik berdasarkan total file di folder sample
            total_files = len(os.listdir("sample"))
            filename = f"sample/snapshot_{total_files + 1}.jpg"

            # Simpan fisik file gambar ke folder lokal /sample
            cv2.imwrite(filename, frame)

            return {
                "status": "success",
                "message": f"Gambar sampel berhasil disimpan sebagai {filename}!",
            }

        return {"status": "error", "message": "Format gambar tidak valid."}

    except Exception as e:
        return {"status": "error", "message": f"Gagal menyimpan: {str(e)}"}