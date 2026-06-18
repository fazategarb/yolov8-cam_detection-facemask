# Real-Time Face Mask Detection using YOLOv8 Nano

Proyek ini merupakan aplikasi deteksi masker wajah secara *real-time* menggunakan webcam/kamera laptop. Aplikasi ini dibangun di atas Jupyter Notebook menggunakan pustaka **Ultralytics YOLOv8** dengan varian model terkecil (**YOLOv8 nano**) yang telah dilatih secara kustom (`best.pt`).

## 🚀 Fitur Utama
* **Deteksi Real-time:** Melakukan inferensi masker wajah langsung dari *feed* kamera laptop dengan performa ringan.
* **Robust Error Handling:** Program dilengkapi penanganan eror jika model eksternal tidak ditemukan atau kamera sedang sibuk/gagal diakses.
* **Snapshot Berdasarkan Permintaan (On-Demand Snapshot):** Mengambil sampel gambar *real-time* beserta hasil deteksinya dan menyimpannya langsung ke folder lokal.

---

## 📂 Struktur Proyek
```text
├── sample/               # Folder tempat menyimpan hasil snapshot foto (auto-generated)
├── model/best.pt         # File bobot model kustom YOLOv8 Nano
├── deteksi_masker.ipynb  # Jupyter Notebook utama untuk eksekusi program
└── README.md             # Dokumentasi teknis proyek