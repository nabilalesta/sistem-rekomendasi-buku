import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import os

# Baca data buku
books = pd.read_csv('data_buku_bersih.csv')

# Gabungkan semua fitur teks untuk vectorizer
books['content'] = (
    books['judulBuku'] + ' ' +
    books['tipe'] + ' ' +
    books['penerbit'] + ' ' +
    books['tahun'].astype(str) + ' ' +
    books['deskripsi']
)

# Buat TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(books['content'])

# Buat dictionary untuk menyimpan profil pengguna
user_profiles = {}

# Muat data pengguna (jika ada)
if os.path.exists('user_data.json'):
    with open('user_data.json', 'r') as f:
        try:
            user_data = json.load(f)
            for username in user_data.keys():
                user_profiles[username] = np.zeros((1, tfidf_matrix.shape[1]))
            print(f"Profil dibuat untuk {len(user_profiles)} pengguna.")
        except json.JSONDecodeError:
            print("File user_data.json kosong atau rusak.")
else:
    print("File user_data.json tidak ditemukan.")

# Tambahkan profil default jika belum ada
if 'default' not in user_profiles:
    user_profiles['default'] = np.zeros((1, tfidf_matrix.shape[1]))
    print("Profil default ditambahkan.")

# Simpan semua model
models = {
    'vectorizer': vectorizer,
    'tfidf_matrix': tfidf_matrix,
    'user_profiles': user_profiles
}

with open('ModelRekomendasi.pkl', 'wb') as f:
    pickle.dump(models, f)

print("Model dan data berhasil disimpan.")
print(f"Jumlah profil pengguna: {len(user_profiles)}")
print(f"Ukuran vektor TF-IDF: {tfidf_matrix.shape[1]}")

