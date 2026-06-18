import os
import cv2
from ultralytics import YOLO

# 1. Konfigurasi Folder Penyimpanan Sampel
# Menggunakan folder 'sample' di direktori yang sama dengan notebook
output_folder = "sample"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder '{output_folder}' berhasil dibuat.")

# 2. Load Model dengan Handle Error
model_path = "model/best.pt"
try:
    model = YOLO(model_path)
    print(f"Model {model_path} berhasil dimuat.")
except Exception as e:
    print(
        f"Error: Gagal memuat model '{model_path}'. Pastikan file berada di folder yang sama."
    )
    print(f"Detail Error: {e}")
    # Berhenti jika model tidak ditemukan
    raise SystemExit

# 3. Akses Kamera Laptop dengan Handle Error
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Kamera tidak merespon atau sedang digunakan aplikasi lain.")
    print("Kamera berhasil dibuka.")
    print("-" * 50)
    print("KONTROL KEYBOARD:")
    print("Tahan 's' -> Untuk mengambil foto & simpan sampel hasil deteksi")
    print("Tekan 'q' -> Untuk keluar dari program")
    print("-" * 50)
except Exception as e:
    print(f"Error Kamera: {e}")
    raise SystemExit

# Penghitung untuk nama file foto agar tidak duplikat
photo_counter = 0

# 4. Loop Utama Real-time
while cap.isOpened():
    try:
        success, frame = cap.read()

        if not success:
            print("Error: Gagal membaca frame dari kamera. Menghentikan...")
            break

        # Jalankan inferensi YOLOv8
        results = model(frame, stream=True)

        # Visualisasikan hasil deteksi ke frame asli
        annotated_frame = frame.copy()  # fallback jika plot gagal
        for r in results:
            annotated_frame = r.plot()

        # Tampilkan hasil ke jendela OpenCV
        cv2.imshow("Deteksi Masker Real-time - YOLOv8", annotated_frame)

        # Membaca input keyboard (menunggu 1 ms)
        key = cv2.waitKey(1) & 0xFF

        # --- FITUR 1: Keluar Program (Tekan 'q') ---
        if key == ord("q"):
            print("Menutup program atas permintaan pengguna...")
            break

        # --- FITUR 2: Ambil Foto Sampel (Tekan 's') ---
        elif key == ord("s"):
            # Membuat nama file unik berdasarkan urutan/counter
            # Kamu juga bisa menggantinya dengan format waktu (timestamp)
            filename = f"sample_mask_{photo_counter}.jpg"
            filepath = os.path.join(output_folder, filename)

            # Pastikan file tidak menimpa yang sudah ada
            while os.path.exists(filepath):
                photo_counter += 1
                filename = f"sample_mask_{photo_counter}.jpg"
                filepath = os.path.join(output_folder, filename)

            # Menyimpan bingkai yang sudah ada kotak deteksinya (annotated_frame)
            # Jika ingin menyimpan foto polos tanpa kotak, ganti menjadi cv2.imwrite(filepath, frame)
            cv2.imwrite(filepath, annotated_frame)
            print(f"[SUKSES] Sampel disimpan ke: {filepath}")
            photo_counter += 1

    except Exception as e:
        print(f"\nTerjadi error saat runtime: {e}")
        break

# 5. Bersihkan Proses
cap.release()
cv2.destroyAllWindows()
print("Kamera ditutup dan semua jendela OpenCV dibersihkan.")