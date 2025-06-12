# Sistem Rekomendasi Buku

Aplikasi web sederhana untuk merekomendasikan buku berdasarkan preferensi pengguna.

## Cara Menjalankan

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Generate model TF-IDF dan data yang diperlukan:
   ```
   python generate_models.py
   ```

3. Jalankan aplikasi Streamlit:
   ```
   streamlit run app.py
   ```

4. Buka browser dan akses `http://localhost:8501`

## Fitur

- Login dan registrasi pengguna
- Pengaturan preferensi kategori buku
- Rekomendasi buku personal
- Riwayat rekomendasi