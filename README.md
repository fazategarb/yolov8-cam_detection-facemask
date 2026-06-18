# Real-Time Face Mask Detection using YOLOv8 Nano

Proyek ini merupakan aplikasi deteksi masker wajah secara *real-time* menggunakan webcam/kamera laptop. Aplikasi ini dibangun menggunakan pustaka **Ultralytics YOLOv8** dengan varian model terkecil (**YOLOv8 nano**) yang telah dilatih secara kustom (`best.pt`).

## 🚀 Fitur Utama
* **Deteksi Real-time:** Melakukan inferensi masker wajah langsung dari *feed* kamera laptop dengan performa ringan.
* **Robust Error Handling:** Program dilengkapi penanganan eror jika model eksternal tidak ditemukan atau kamera sedang sibuk/gagal diakses.
* **Snapshot Berdasarkan Permintaan (On-Demand Snapshot):** Mengambil sampel gambar *real-time* beserta hasil deteksinya dan menyimpannya langsung ke folder lokal.

---

## 📂 Struktur Proyek
```text
deteksi-masker-yolov8/
│
├── .venv/                # Virtual Environment Python (lokal)
├── model/
│   └── best.pt           # File bobot model kustom YOLOv8 Nano
├── sample/               # Folder tempat menyimpan hasil snapshot foto (auto-generated)
├── index.py              # File utama Python untuk mengeksekusi program
├── requirements.txt      # Daftar dependency/library yang dibutuhkan
└── README.md             # Dokumentasi teknis proyek
```

## 🙌 Penggunaan
Proyek ini menggunakan Virtual Environment (.venv) untuk menjaga isolasi dependency. Ikuti langkah-langkah berikut untuk menyiapkan lingkungan pengembangan kamu:

### 1. Klon atau Buka Folder Proyek
```bash
git clone https://github.com/fazategarb/yolov8-cam_detection-facemask.git
cd yolov8-cam_detection-facemask
```

### 2. Buat dan Aktifkan Virtual Environment
Buat environment baru bernama `.venv`:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependency `requirements.txt`
Pastikan `.venv` Anda sudah aktif (ditandai dengan teks `(.venv)` di awal baris terminal), lalu jalankan perintah instalasi berikut:
```bash
pip install -r requirements.txt
```
(Catatan: Isi requirements.txt minimal harus mencakup ultralytics dan opencv-python).
### 4. Setup Model YOLOv8nano
1. Pastikan file bobot model kustom Anda sudah berada di dalam folder `model/` dengan nama `best.pt`.
2. Aktifkan `.venv` Anda di terminal.
3. Jalankan file utama menggunakan perintah berikut:
```bash
python index.py
```
4. Jendela grafis (pop-up) OpenCV akan muncul menampilkan feed kamera laptop beserta kotak deteksi masker.

### 5. Control Keybinds
**Pastikan fokus kursor Anda sedang aktif mengklik jendela kamera OpenCV, kemudian gunakan tombol berikut:**
* Tekan `s`: Mengambil foto sampel saat itu juga. Foto hasil deteksi akan otomatis tersimpan di folder `sample/` dengan penamaan terurut (sample_mask_0.jpg, dst).
* Tekan `q`: Keluar dari aplikasi dan menutup jendela kamera dengan aman.
