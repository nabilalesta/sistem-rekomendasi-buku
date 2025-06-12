import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import os

# Baca data buku
books = pd.read_csv('data_buku_bersih.csv')

# Gabungkan semua fitur teks untuk vectorizer
books['content'] = books['judulBuku'] + ' ' + books['tipe'] + ' ' + books['penerbit'] + ' ' + str(books['tahun']) + ' ' + books['deskripsi']

# Buat TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(books['content'])

# Buat dictionary untuk menyimpan profil pengguna
user_profiles = {}

# Muat data pengguna yang ada (jika ada)
if os.path.exists('user_data.json'):
    with open('user_data.json', 'r') as f:
        try:
            user_data = json.load(f)
            # Buat profil untuk setiap pengguna yang ada
            for username in user_data.keys():
                user_profiles[username] = np.zeros((1, tfidf_matrix.shape[1]))
            print(f"Profil dibuat untuk {len(user_profiles)} pengguna yang ada")
        except json.JSONDecodeError:
            print("File user_data.json kosong atau tidak valid, membuat dictionary kosong")
            user_profiles = {}
else:
    print("File user_data.json tidak ditemukan, membuat dictionary kosong")

# Tambahkan profil default untuk pengguna baru
user_profiles['default'] = np.zeros((1, tfidf_matrix.shape[1]))

# Buat dictionary untuk menyimpan semua model
models = {
    'vectorizer': vectorizer,
    'tfidf_matrix': tfidf_matrix,
    'user_profiles': user_profiles
}

# Simpan model dan data
with open('ModelRekomendasi.pkl', 'wb') as f:
    pickle.dump(models, f)

print("Model dan data berhasil disimpan!")
print(f"Jumlah profil pengguna: {len(user_profiles)}")
print(f"Ukuran vektor TF-IDF: {tfidf_matrix.shape[1]}")

if not load_all_models():
    # Jika gagal, coba load dari file terpisah
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open('user_profiles.pkl', 'rb') as f:
        user_profiles = pickle.load(f)
    
    # Simpan model dalam satu file untuk penggunaan selanjutnya
    save_all_models()

if st.session_state.username == "admin":
    st.markdown("""
    <div class='info-box'>
        <h3>Selamat Datang di Panel Admin</h3>
        <p>Untuk mengelola data buku dan pengguna, silakan gunakan menu Admin Panel di sidebar.</p>
        <p>Di panel admin Anda dapat:</p>
        <ul>
            <li>Melihat dan mengelola data buku</li>
            <li>Menambah, mengedit, dan menghapus buku</li>
            <li>Mengelola data pengguna</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    return

# Tombol Home hanya ditampilkan jika user BUKAN admin
if not (st.session_state.logged_in and st.session_state.username == "admin"):
    if st.button("Home", key="nav_home", 
                help="Kembali ke halaman utama",
                use_container_width=True,
                type="primary" if st.session_state.page == "home" else "secondary"):
        change_page("home")
        st.rerun()