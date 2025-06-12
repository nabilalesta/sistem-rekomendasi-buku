import streamlit as st
import pandas as pd
import pickle
import hashlib
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page config untuk layout yang lebih lebar
st.set_page_config(layout="wide", page_title="Sistem Rekomendasi Buku", page_icon="📚")

# Custom CSS untuk styling
st.markdown("""
<style>
    /* Styling untuk tombol navigasi di kiri atas */
    .nav-buttons {
        position: fixed;
        top: 0.5rem;
        left: 0.5rem;
        z-index: 999;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 0.5rem;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .nav-button {
        background-color: #1e3a8a;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s ease;
        width: 120px;
        text-align: center;
    }
    .nav-button:hover {
        background-color: #1e40af;
    }
    .active-nav-button {
        background-color: #1e40af;
    }
    /* Tambahkan margin untuk konten utama agar tidak tertutup tombol navigasi */
    .main-content {
        margin-left: 140px;
    }
    
    /* CSS yang sudah ada sebelumnya */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1.5rem;
        color: #1e3a8a;
        padding: 1rem;
        border-bottom: 2px solid #e5e7eb;
    }
    .book-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin-top: 1.5rem;
    }
    .book-card {
        width: 100%;
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        background-color: white;
        transition: transform 0.3s ease;
    }
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    .book-image {
        width: 150px;
        height: 220px;
        background-color: #f3f4f6;
        margin: 0 auto;
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6b7280;
        font-weight: bold;
    }
    .book-title {
        font-weight: bold;
        margin-top: 1rem;
        font-size: 1.1rem;
        color: #1e3a8a;
    }
    .book-description {
        font-size: 0.9rem;
        color: #4b5563;
        margin-top: 0.5rem;
        text-align: justify;
        padding: 0 0.5rem;
    }
    .search-container {
        max-width: 700px;
        margin: 0 auto;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background-color: #f9fafb;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        margin: 1.5rem 0;
        color: #1e3a8a;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e5e7eb;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        width: 100%;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1e40af;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f3f4f6;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a8a;
        color: white;
    }
    .info-box {
        padding: 1rem;
        background-color: #e0f2fe;
        border-left: 4px solid #0ea5e9;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #dcfce7;
        border-left: 4px solid #22c55e;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fef9c3;
        border-left: 4px solid #eab308;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .book-meta {
        display: flex;
        justify-content: space-between;
        margin: 0.5rem 0;
        font-size: 0.8rem;
        color: #6b7280;
    }
    .book-meta span {
        background-color: #f3f4f6;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
    }
    /* Penyejajaran tombol dengan input */
    div[data-testid="column"]:nth-of-type(2) .stButton {
        margin-top: 0px;
        height: 100%;
    }
    div[data-testid="column"]:nth-of-type(2) .stButton > button {
        height: 2.4rem;
        margin-top: 0px;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi session state
if 'page' not in st.session_state:
    st.session_state.page = "login"  # Ubah default page ke login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'admin_page_number' not in st.session_state:
    st.session_state.admin_page_number = 0

# Fungsi untuk load user data
def load_user_data():
    if os.path.exists('user_data.json'):
        with open('user_data.json', 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

# Fungsi untuk save user data
def save_user_data(data):
    with open('user_data.json', 'w') as f:
        json.dump(data, f)

# Fungsi untuk menyimpan semua model dalam satu file
def save_all_models():
    models = {
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'user_profiles': user_profiles
    }
    with open('ModelRekomendasi.pkl', 'wb') as f:
        pickle.dump(models, f)

# Fungsi untuk memuat semua model dari satu file
def load_all_models():
    global vectorizer, tfidf_matrix, user_profiles
    try:
        with open('ModelRekomendasi.pkl', 'rb') as f:
            models = pickle.load(f)
            vectorizer = models['vectorizer']
            tfidf_matrix = models['tfidf_matrix']
            user_profiles = models['user_profiles']
        return True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return False

# Load data
try:
    books = pd.read_csv('data_buku_bersih.csv')
    
    # Bersihkan kolom tahun - hapus karakter non-numerik
    books['tahun'] = books['tahun'].astype(str).str.extract('(\d+)', expand=False)
    books['tahun'] = pd.to_numeric(books['tahun'], errors='coerce')
    books['tahun'] = books['tahun'].fillna(2000)  # Isi nilai NaN dengan default
    
    # Coba load model dari file tunggal
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
    
    # Load atau buat user data
    user_data = load_user_data()
    
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

# Fungsi untuk memperbarui profil pengguna
def update_user_profile(username, book_index, rating=1.0):
    """
    Memperbarui profil pengguna berdasarkan buku yang direkomendasikan
    username: nama pengguna
    book_index: indeks buku dalam dataframe
    rating: nilai rating (default 1.0 untuk rekomendasi)
    """
    # Pastikan pengguna ada dalam dictionary
    if username not in user_profiles:
        user_profiles[username] = np.zeros((1, tfidf_matrix.shape[1]))
    
    # Perbarui profil dengan menambahkan vektor buku yang direkomendasikan
    book_vector = tfidf_matrix[book_index]
    user_profiles[username] += rating * book_vector.toarray()
    
    # Simpan profil yang diperbarui
    with open('user_profiles.pkl', 'wb') as f:
        pickle.dump(user_profiles, f)

# Fungsi untuk mendapatkan rekomendasi berdasarkan profil pengguna
def get_recommendations(username, filtered_indices=None, top_n=5):
    """
    Mendapatkan rekomendasi buku untuk pengguna
    username: nama pengguna
    filtered_indices: indeks buku yang sudah difilter berdasarkan preferensi
    top_n: jumlah rekomendasi yang diinginkan
    """
    # Gunakan profil pengguna jika ada, jika tidak gunakan profil default
    if username in user_profiles:
        user_profile = user_profiles[username]
    else:
        user_profile = user_profiles['default']
    
    # Hitung similarity
    similarity = cosine_similarity(user_profile, tfidf_matrix)
    scores = list(enumerate(similarity[0]))
    
    # Filter berdasarkan indeks jika ada
    if filtered_indices:
        scores = [score for score in scores if score[0] in filtered_indices]
    
    # Urutkan berdasarkan similarity
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    
    return ranked[:top_n]

# Fungsi untuk halaman Home
def show_home_page():
    st.markdown("<h1 class='main-header'>📚 Sistem Rekomendasi Buku</h1>", unsafe_allow_html=True)
    
    # Jika user sudah login
    if st.session_state.logged_in:
        # Jika user adalah admin, tampilkan pesan khusus
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
        
        # Untuk user biasa, tampilkan konten normal
        st.markdown(f"<h2 class='section-header'>Selamat Datang, {st.session_state.username}!</h2>", unsafe_allow_html=True)
        
        # Cek apakah user memiliki history buku
        if st.session_state.username in user_data and "history" in user_data[st.session_state.username]:
            history = user_data[st.session_state.username]["history"]
            
            if history:
                st.markdown("<h3 class='section-header'>Buku-buku dalam Profil Anda</h3>", unsafe_allow_html=True)
                
                # Cari buku-buku yang ada di history
                profile_books = []
                for title in history:
                    # Cari buku berdasarkan judul
                    book_indices = books[books['judulBuku'] == title].index.tolist()
                    if book_indices:
                        book_index = book_indices[0]
                        book = books.iloc[book_index]
                        profile_books.append({
                            "title": book['judulBuku'],
                            "type": book['tipe'],
                            "year": book['tahun'],
                            "publisher": book['penerbit'],
                            "description": book['deskripsi'],
                            "index": book_index
                        })
                
                if profile_books:
                    # Tampilkan buku dalam format kartu
                    # Buat baris-baris dengan 5 kolom per baris
                    for i in range(0, len(profile_books), 5):
                        cols = st.columns(5)
                        for j in range(5):
                            if i+j < len(profile_books):
                                book = profile_books[i+j]
                                with cols[j]:
                                    st.markdown(f"""
                                    <div class='book-card'>
                                        <div class='book-image'>📚</div>
                                        <div class='book-title'>{book['title']}</div>
                                        <div class='book-meta'>
                                            <span>{book['type']}</span>
                                            <span>{int(book['year'])}</span>
                                        </div>
                                        <div class='book-description'>{book['description'][:100]}...</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Tampilkan detail dengan expander
                                    with st.expander("Detail Buku"):
                                        st.markdown(f"**Judul:** {book['title']}")
                                        st.markdown(f"**Tipe:** {book['type']}")
                                        st.markdown(f"**Tahun:** {int(book['year'])}")
                                        st.markdown(f"**Penerbit:** {book['publisher']}")
                                        st.markdown(f"**Deskripsi:**")
                                        st.markdown(f"{book['description']}")
                                        
                                        # Tambahkan tombol untuk menghapus dari profil
                                        if st.button(f"Hapus dari Profil", key=f"remove_{book['index']}"):
                                            user_data[st.session_state.username]["history"].remove(book['title'])
                                            save_user_data(user_data)
                                            st.success(f"Buku '{book['title']}' dihapus dari profil Anda!")
                                            st.rerun()
                else:
                    st.info("Belum ada buku yang ditambahkan ke profil Anda.")
            else:
                st.info("Belum ada buku yang ditambahkan ke profil Anda.")
        else:
            st.info("Belum ada buku yang ditambahkan ke profil Anda.")
        
        # Cek apakah user sudah mengatur preferensi
        has_preferences = False
        if st.session_state.username in user_data and "preferences" in user_data[st.session_state.username]:
            if "keywords" in user_data[st.session_state.username]["preferences"]:
                if user_data[st.session_state.username]["preferences"]["keywords"]:
                    has_preferences = True
        
        if has_preferences:
            # Tampilkan rekomendasi berdasarkan profil pengguna
            st.markdown("<h3 class='section-header'>Rekomendasi Berdasarkan Profil Anda</h3>", unsafe_allow_html=True)
            
            # Dapatkan rekomendasi berdasarkan profil pengguna
            if st.session_state.username in user_profiles:
                # Gunakan fungsi get_recommendations untuk mendapatkan rekomendasi
                ranked = get_recommendations(st.session_state.username, top_n=5)
                
                if ranked and len(ranked) > 0:
                    # Tampilkan buku dalam format kartu
                    cols = st.columns(5)  # 5 kolom
                    for i, (book_idx, score) in enumerate(ranked):
                        if i < 5:  # Tampilkan 5 buku teratas
                            book = books.iloc[book_idx]
                            with cols[i]:
                                st.markdown(f"""
                                <div class='book-card'>
                                    <div class='book-image'>📚</div>
                                    <div class='book-title'>{book['judulBuku']}</div>
                                    <div class='book-meta'>
                                        <span>{book['tipe']}</span>
                                        <span>{int(book['tahun'])}</span>
                                    </div>
                                    <div class='book-description'>{book['deskripsi'][:100]}...</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Tampilkan detail dengan expander
                                with st.expander("Detail Buku"):
                                    st.markdown(f"**Judul:** {book['judulBuku']}")
                                    st.markdown(f"**Tipe:** {book['tipe']}")
                                    st.markdown(f"**Tahun:** {int(book['tahun'])}")
                                    st.markdown(f"**Penerbit:** {book['penerbit']}")
                                    st.markdown(f"**Deskripsi:**")
                                    st.markdown(f"{book['deskripsi']}")
                                    st.markdown(f"**Kemiripan:** {score:.2f}")
                                
                                # Tambahkan tombol untuk menambahkan ke profil
                                if st.button(f"Tambahkan ke Profil", key=f"add_rec_home_{book_idx}"):
                                    update_user_profile(st.session_state.username, book_idx)
                                    if book['judulBuku'] not in user_data[st.session_state.username]["history"]:
                                        user_data[st.session_state.username]["history"].append(book['judulBuku'])
                                        save_user_data(user_data)
                                    st.success(f"Buku '{book['judulBuku']}' ditambahkan ke profil Anda!")
                                    st.rerun()
                else:
                    st.info("Belum ada rekomendasi yang tersedia. Tambahkan lebih banyak buku ke profil Anda atau atur preferensi Anda.")
            else:
                st.info("Belum ada rekomendasi yang tersedia. Tambahkan buku ke profil Anda atau atur preferensi Anda.")
        else:
            # Tampilkan pesan untuk user baru
            st.markdown("""
            <div class='info-box'>
                <h3>Selamat Datang di Sistem Rekomendasi Buku!</h3>
                <p>Untuk mendapatkan rekomendasi buku yang sesuai dengan minat Anda, silakan:</p>
                <ol>
                    <li>Klik menu "Profile" di sidebar</li>
                    <li>Atur preferensi Anda dengan memasukkan kata kunci yang Anda minati</li>
                    <li>Simpan preferensi Anda</li>
                </ol>
                <p>Setelah mengatur preferensi, Anda akan mendapatkan rekomendasi buku yang sesuai dengan minat Anda.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tambahkan bagian pencarian
        st.markdown("<div class='search-container'>", unsafe_allow_html=True)
        st.markdown("<h2 class='section-header'>Cari Rekomendasi Buku Baru</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            search_keyword = st.text_input("", placeholder="Masukkan judul buku atau kata kunci", key="search_home", label_visibility="collapsed")
        with col2:
            search_btn = st.button("Rekomendasikan", key="btn_search_home")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if search_btn and search_keyword:
            # Cari buku berdasarkan kata kunci
            keyword_vector = vectorizer.transform([search_keyword])
            similarity = cosine_similarity(keyword_vector, tfidf_matrix)
            scores = list(enumerate(similarity[0]))
            ranked = sorted(scores, key=lambda x: x[1], reverse=True)
            
            # Tampilkan hasil dalam format kartu
            book_data = []
            count = 0
            for i in ranked:
                if count >= 5:  # Tampilkan 5 buku teratas
                    break
                if i[1] > 0:  # Hanya tampilkan jika ada kemiripan
                    book = books.iloc[i[0]]
                    book_data.append({
                        "title": book['judulBuku'],
                        "type": book['tipe'],
                        "year": book['tahun'],
                        "publisher": book['penerbit'],
                        "description": book['deskripsi'],
                        "index": i[0],
                        "similarity": i[1]
                    })
                    count += 1
            
            if count > 0:
                st.markdown("<h2 class='section-header'>Hasil Rekomendasi</h2>", unsafe_allow_html=True)
                
                # Tampilkan buku dalam format kartu
                cols = st.columns(5)  # 5 kolom
                for i, book in enumerate(book_data):
                    with cols[i]:
                        st.markdown(f"""
                        <div class='book-card'>
                            <div class='book-image'>📚</div>
                            <div class='book-title'>{book['title']}</div>
                            <div class='book-meta'>
                                <span>{book['type']}</span>
                                <span>{int(book['year'])}</span>
                            </div>
                            <div class='book-description'>{book['description'][:100]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Tampilkan detail dengan expander
                        with st.expander("Detail Buku"):
                            st.markdown(f"**Judul:** {book['title']}")
                            st.markdown(f"**Tipe:** {book['type']}")
                            st.markdown(f"**Tahun:** {int(book['year'])}")
                            st.markdown(f"**Penerbit:** {book['publisher']}")
                            st.markdown(f"**Deskripsi:**")
                            st.markdown(f"{book['description']}")
                            st.markdown(f"**Kemiripan:** {book['similarity']:.2f}")
                        
                        # Tambahkan tombol untuk menambahkan ke profil
                        if st.button(f"Tambahkan ke Profil", key=f"add_home_{book['index']}"):
                            update_user_profile(st.session_state.username, book['index'])
                            if book['title'] not in user_data[st.session_state.username]["history"]:
                                user_data[st.session_state.username]["history"].append(book['title'])
                                save_user_data(user_data)
                            st.success(f"Buku '{book['title']}' ditambahkan ke profil Anda!")
                            st.rerun()
            else:
                st.warning("Tidak ditemukan buku yang sesuai dengan kata kunci tersebut.")
    else:
        # Tampilkan halaman home untuk user yang belum login
        # Search container
        st.markdown("<div class='search-container'>", unsafe_allow_html=True)
        st.markdown("<h2 class='section-header'>Cari Rekomendasi Buku</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            search_keyword = st.text_input("", placeholder="Masukkan judul buku atau kata kunci", key="search_home", label_visibility="collapsed")
        with col2:
            search_btn = st.button("Rekomendasikan", key="btn_search_home")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if search_btn and search_keyword:
            # Cari buku berdasarkan kata kunci
            keyword_vector = vectorizer.transform([search_keyword])
            similarity = cosine_similarity(keyword_vector, tfidf_matrix)
            scores = list(enumerate(similarity[0]))
            ranked = sorted(scores, key=lambda x: x[1], reverse=True)
            
            # Tampilkan hasil dalam format kartu
            book_data = []
            count = 0
            for i in ranked:
                if count >= 5:  # Tampilkan 5 buku teratas
                    break
                if i[1] > 0:  # Hanya tampilkan jika ada kemiripan
                    book = books.iloc[i[0]]
                    book_data.append({
                        "title": book['judulBuku'],
                        "type": book['tipe'],
                        "year": book['tahun'],
                        "publisher": book['penerbit'],
                        "description": book['deskripsi'],
                        "index": i[0],
                        "similarity": i[1]
                    })
                    count += 1
            
            if count > 0:
                st.markdown("<h2 class='section-header'>Hasil Rekomendasi</h2>", unsafe_allow_html=True)
                
                # Tampilkan buku dalam format kartu
                cols = st.columns(5)  # 5 kolom
                for i, book in enumerate(book_data):
                    with cols[i]:
                        st.markdown(f"""
                        <div class='book-card'>
                            <div class='book-image'>📚</div>
                            <div class='book-title'>{book['title']}</div>
                            <div class='book-meta'>
                                <span>{book['type']}</span>
                                <span>{int(book['year'])}</span>
                            </div>
                            <div class='book-description'>{book['description'][:100]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Tampilkan detail dengan expander
                        with st.expander("Detail Buku"):
                            st.markdown(f"**Judul:** {book['title']}")
                            st.markdown(f"**Tipe:** {book['type']}")
                            st.markdown(f"**Tahun:** {int(book['year'])}")
                            st.markdown(f"**Penerbit:** {book['publisher']}")
                            st.markdown(f"**Deskripsi:**")
                            st.markdown(f"{book['description']}")
                            st.markdown(f"**Kemiripan:** {book['similarity']:.2f}")
                        
                        # Tambahkan tombol untuk login
                        if st.button(f"Login untuk Menambahkan", key=f"login_add_{book['index']}"):
                            st.session_state.page = "login"
                            st.rerun()
            else:
                st.warning("Tidak ditemukan buku yang sesuai dengan kata kunci tersebut.")
        else:
            # Tampilkan pesan informasi
            st.markdown("""
            <div class='info-box'>
                <h3>Selamat Datang di Sistem Rekomendasi Buku</h3>
                <p>Masukkan judul buku atau kata kunci di kotak pencarian di atas untuk mendapatkan rekomendasi buku.</p>
                <p>Anda juga dapat membuat akun untuk menyimpan preferensi dan mendapatkan rekomendasi yang lebih personal.</p>
            </div>
            """, unsafe_allow_html=True)

# Fungsi untuk halaman Login
def show_login_page():
    st.markdown("<h1 class='main-header'>Login / Register</h1>", unsafe_allow_html=True)
    
    # Buat tab untuk login dan daftar
    login_tab, register_tab = st.tabs(["Login", "Daftar"])
    
    # Tab Login
    with login_tab:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="btn_login"):
            if login_username and login_password:
                # Verifikasi login
                hashed_pw = hashlib.sha256(login_password.encode()).hexdigest()
                if login_username in user_data and user_data[login_username]["password"] == hashed_pw:
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    
                    # Jika username adalah admin, arahkan ke halaman admin
                    if login_username == "admin":
                        st.session_state.page = "admin"
                        st.markdown("<div class='success-box'>Login berhasil! Anda akan dialihkan ke panel admin.</div>", unsafe_allow_html=True)
                    else:
                        # Untuk pengguna biasa, arahkan ke halaman utama
                        st.session_state.page = "home"
                        st.markdown("<div class='success-box'>Login berhasil! Anda akan dialihkan ke halaman utama.</div>", unsafe_allow_html=True)
                    
                    st.rerun()
                else:
                    st.markdown("<div class='error-box'>Username atau password salah</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='warning-box'>Username dan password harus diisi</div>", unsafe_allow_html=True)
    
    # Tab Daftar
    with register_tab:
        st.subheader("Daftar Akun Baru")
        reg_username = st.text_input("Username", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_confirm_password = st.text_input("Konfirmasi Password", type="password", key="reg_confirm_password")
        
        if st.button("Daftar", key="btn_register"):
            if reg_username and reg_password:
                if reg_password != reg_confirm_password:
                    st.error("Password dan konfirmasi password tidak sama")
                elif reg_username in user_data:
                    st.error("Username sudah digunakan")
                else:
                    # Buat profil user baru
                    hashed_pw = hashlib.sha256(reg_password.encode()).hexdigest()
                    user_data[reg_username] = {
                        "password": hashed_pw,
                        "preferences": {},
                        "history": []
                    }
                    save_user_data(user_data)
                    
                    # Inisialisasi profil pengguna jika belum ada
                    if reg_username not in user_profiles:
                        user_profiles[reg_username] = np.zeros((1, tfidf_matrix.shape[1]))
                        with open('user_profiles.pkl', 'wb') as f:
                            pickle.dump(user_profiles, f)
                    
                    # Set session state untuk login otomatis
                    st.session_state.logged_in = True
                    st.session_state.username = reg_username
                    
                    # Jika username adalah admin, arahkan ke halaman admin
                    if reg_username == "admin":
                        st.session_state.page = "admin"
                        st.success("Pendaftaran berhasil! Anda akan dialihkan ke panel admin.")
                    else:
                        # Untuk pengguna baru, arahkan ke halaman profil untuk mengatur preferensi
                        st.session_state.page = "profile"
                        st.success("Pendaftaran berhasil! Silakan atur preferensi Anda terlebih dahulu.")
                    
                    st.rerun()
            else:
                st.warning("Username dan password harus diisi")

# Fungsi untuk halaman Profile
def show_profile_page():
    st.markdown(f"<h1 class='main-header'>Profile: {st.session_state.username}</h1>", unsafe_allow_html=True)
    
    # Tab untuk preferensi dan rekomendasi
    tab1, tab2 = st.tabs(["Preferensi", "Rekomendasi"])
    
    with tab1:
        st.subheader("Atur Preferensi Anda")
        st.info("Masukkan kata kunci untuk mendapatkan rekomendasi yang lebih akurat.")
        
        # Ambil preferensi yang sudah ada jika ada
        existing_keywords = ""
        if st.session_state.username in user_data and "preferences" in user_data[st.session_state.username]:
            if "keywords" in user_data[st.session_state.username]["preferences"]:
                existing_keywords = user_data[st.session_state.username]["preferences"]["keywords"]
        
        # Form untuk kata kunci
        keywords = st.text_area("Masukkan kata kunci yang Anda minati (pisahkan dengan koma):", 
                               value=existing_keywords,
                               help="Contoh: novel, sejarah, pendidikan, teknologi")
        
        # Simpan preferensi
        if st.button("Simpan Preferensi"):
            if st.session_state.username in user_data:
                user_data[st.session_state.username]["preferences"] = {
                    "keywords": keywords
                }
                save_user_data(user_data)
                
                # Update profil pengguna berdasarkan kata kunci
                if keywords:
                    # Buat vektor TF-IDF dari kata kunci
                    keyword_vector = vectorizer.transform([keywords])
                    
                    # Reset profil pengguna
                    user_profiles[st.session_state.username] = np.zeros((1, tfidf_matrix.shape[1]))
                    
                    # Update profil dengan kata kunci baru
                    user_profiles[st.session_state.username] += keyword_vector.toarray()
                    
                    # Simpan profil yang diperbarui
                    with open('user_profiles.pkl', 'wb') as f:
                        pickle.dump(user_profiles, f)
                
                st.success("Preferensi berhasil disimpan!")
    
    with tab2:
        st.subheader("Rekomendasi Buku")
        
        if st.button("Tampilkan Rekomendasi"):
            # Gunakan fungsi get_recommendations untuk mendapatkan rekomendasi
            ranked = get_recommendations(st.session_state.username, top_n=10)
            
            if ranked:
                st.subheader("Rekomendasi Personal untuk Anda:")
                
                # Tampilkan buku dalam format kartu
                cols = st.columns(5)  # 5 kolom
                count = 0
                for i, score in ranked:
                    if count >= 5:  # Tampilkan 5 buku teratas
                        break
                    book = books.iloc[i]
                    with cols[count]:
                        st.markdown(f"""
                        <div class='book-card'>
                            <div class='book-image'>📚</div>
                            <div class='book-title'>{book['judulBuku']}</div>
                            <div class='book-meta'>
                                <span>{book['tipe']}</span>
                                <span>{int(book['tahun'])}</span>
                            </div>
                            <div class='book-description'>{book['deskripsi'][:100]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Tampilkan detail dengan expander
                        with st.expander("Detail Buku"):
                            st.markdown(f"**Judul:** {book['judulBuku']}")
                            st.markdown(f"**Tipe:** {book['tipe']}")
                            st.markdown(f"**Tahun:** {int(book['tahun'])}")
                            st.markdown(f"**Penerbit:** {book['penerbit']}")
                            st.markdown(f"**Deskripsi:**")
                            st.markdown(f"{book['deskripsi']}")
                            st.markdown(f"**Kemiripan:** {score:.2f}")
                        
                        # Tambahkan tombol untuk menambahkan ke profil
                        if st.button(f"Tambahkan ke Profil", key=f"add_rec_{i}"):
                            update_user_profile(st.session_state.username, i)
                            st.success(f"Buku '{book['judulBuku']}' ditambahkan ke profil Anda!")
                    count += 1
            else:
                st.warning("Tidak dapat menemukan rekomendasi yang sesuai. Coba ubah preferensi Anda.")

# Fungsi untuk mengubah halaman
def change_page(page):
    st.session_state.page = page

# Fungsi untuk logout
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.page = "home"

# Tampilkan tombol navigasi di sidebar
def show_navigation():
    with st.sidebar:
        st.title("Navigation")
        
        # Tombol Home hanya ditampilkan jika user BUKAN admin
        if not (st.session_state.logged_in and st.session_state.username == "admin"):
            if st.button("Home", key="nav_home", 
                        help="Kembali ke halaman utama",
                        use_container_width=True,
                        type="primary" if st.session_state.page == "home" else "secondary"):
                change_page("home")
                st.rerun()
        
        # Tombol Profile hanya ditampilkan jika user sudah login dan BUKAN admin
        if st.session_state.logged_in and st.session_state.username != "admin":
            if st.button("Profile", key="nav_profile", 
                        help="Lihat profil Anda",
                        use_container_width=True,
                        type="primary" if st.session_state.page == "profile" else "secondary"):
                change_page("profile")
                st.rerun()
        
        # Tombol Admin hanya untuk user admin
        if st.session_state.logged_in and st.session_state.username == "admin":
            if st.button("Admin Panel", key="nav_admin", 
                        help="Panel administrasi",
                        use_container_width=True,
                        type="primary" if st.session_state.page == "admin" else "secondary"):
                change_page("admin")
                st.rerun()
            
        # Tombol Logout untuk semua user yang sudah login
        if st.session_state.logged_in:
            if st.button("Logout", key="nav_logout", 
                        help="Keluar dari akun",
                        use_container_width=True):
                logout()
                st.rerun()
        else:
            if st.button("Login", key="nav_login", 
                        help="Masuk ke akun Anda",
                        use_container_width=True,
                        type="primary" if st.session_state.page == "login" else "secondary"):
                change_page("login")
                st.rerun()

# Fungsi untuk menambah buku baru
def add_book(judul, tipe, tahun, penerbit, deskripsi):
    global books, vectorizer, tfidf_matrix
    
    # Tambahkan buku baru ke DataFrame
    new_book = pd.DataFrame({
        'judulBuku': [judul],
        'tipe': [tipe],
        'tahun': [tahun],
        'penerbit': [penerbit],
        'deskripsi': [deskripsi]
    })
    
    books = pd.concat([books, new_book], ignore_index=True)
    
    # Simpan ke file CSV
    books.to_csv('data_buku_bersih.csv', index=False)
    
    # Perbarui model TF-IDF
    update_tfidf_models()

# Fungsi untuk menghapus buku
def delete_book(book_index):
    global books, vectorizer, tfidf_matrix
    
    # Hapus buku dari DataFrame
    books = books.drop(book_index)
    
    # Simpan ke file CSV
    books.to_csv('data_buku_bersih.csv', index=False)
    
    # Perbarui model TF-IDF
    update_tfidf_models()

# Fungsi untuk menghapus user
def delete_user(username):
    global user_data, user_profiles
    
    # Hapus dari user_data
    if username in user_data:
        del user_data[username]
        save_user_data(user_data)
    
    # Hapus dari user_profiles
    if username in user_profiles:
        del user_profiles[username]
        with open('user_profiles.pkl', 'wb') as f:
            pickle.dump(user_profiles, f)

# Fungsi untuk halaman Admin
def show_admin_page():
    st.markdown("<h1 class='main-header'>Panel Admin</h1>", unsafe_allow_html=True)
    
    # Cek apakah user adalah admin
    if not st.session_state.logged_in or st.session_state.username != "admin":
        st.error("Anda tidak memiliki akses ke halaman ini.")
        st.button("Kembali ke Home", on_click=lambda: change_page("home"))
        return
    
    # Tab untuk manajemen buku dan user
    tab1, tab2, tab3, tab4 = st.tabs(["Lihat Data Buku", "Tambah Buku", "Edit/Hapus Buku", "Kelola User"])
    
    with tab1:
        st.subheader("Data Buku")
        
        # Filter dan pencarian
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Cari Buku", placeholder="Masukkan judul, tipe, atau penerbit")
        with col2:
            sort_by = st.selectbox("Urutkan Berdasarkan", ["judulBuku", "tipe", "tahun", "penerbit"])
        
        # Filter data berdasarkan pencarian
        filtered_books = books
        if search_term:
            filtered_books = books[
                books['judulBuku'].str.contains(search_term, case=False, na=False) |
                books['tipe'].str.contains(search_term, case=False, na=False) |
                books['penerbit'].str.contains(search_term, case=False, na=False)
            ]
        
        # Urutkan data
        filtered_books = filtered_books.sort_values(by=sort_by)
        
        # Tampilkan jumlah data
        st.write(f"Menampilkan {len(filtered_books)} dari {len(books)} buku")
        
        # Tampilkan data dalam bentuk tabel dengan pagination
        PAGE_SIZE = 10
        total_pages = (len(filtered_books) + PAGE_SIZE - 1) // PAGE_SIZE
        
        if "admin_page_number" not in st.session_state:
            st.session_state.admin_page_number = 0
            
        # Navigasi halaman
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("← Sebelumnya", disabled=st.session_state.admin_page_number <= 0):
                st.session_state.admin_page_number -= 1
                st.rerun()
        with col2:
            st.write(f"Halaman {st.session_state.admin_page_number + 1} dari {max(1, total_pages)}")
        with col3:
            if st.button("Selanjutnya →", disabled=st.session_state.admin_page_number >= total_pages - 1):
                st.session_state.admin_page_number += 1
                st.rerun()
        
        # Tampilkan data untuk halaman saat ini
        start_idx = st.session_state.admin_page_number * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, len(filtered_books))
        
        current_books = filtered_books.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # Tampilkan tabel
        st.dataframe(
            current_books[['judulBuku', 'tipe', 'tahun', 'penerbit']],
            use_container_width=True,
            column_config={
                "judulBuku": "Judul Buku",
                "tipe": "Tipe",
                "tahun": "Tahun",
                "penerbit": "Penerbit"
            }
        )
        
        # Tampilkan detail buku yang dipilih
        selected_title = st.selectbox("Pilih buku untuk melihat detail", [""] + current_books['judulBuku'].tolist())
        if selected_title:
            selected_book = books[books['judulBuku'] == selected_title].iloc[0]
            
            st.subheader("Detail Buku")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("""
                <div style="width: 150px; height: 220px; background-color: #f3f4f6; 
                            border-radius: 5px; display: flex; align-items: center; 
                            justify-content: center; color: #6b7280; font-weight: bold;">
                    📚
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**Judul:** {selected_book['judulBuku']}")
                st.markdown(f"**Tipe:** {selected_book['tipe']}")
                st.markdown(f"**Tahun:** {int(selected_book['tahun'])}")
                st.markdown(f"**Penerbit:** {selected_book['penerbit']}")
                st.markdown("**Deskripsi:**")
                st.markdown(f"{selected_book['deskripsi']}")
    
    with tab2:
        st.subheader("Tambah Buku Baru")
        
        # Form untuk menambah buku
        with st.form("add_book_form"):
            judul = st.text_input("Judul Buku", key="add_judul")
            
            col1, col2 = st.columns(2)
            with col1:
                tipe = st.selectbox("Tipe", ["Buku", "Jurnal", "Majalah", "Skripsi", "Tesis", "Disertasi", "Lainnya"], key="add_tipe")
            with col2:
                tahun = st.number_input("Tahun", min_value=1900, max_value=2100, value=2023, key="add_tahun")
            
            penerbit = st.text_input("Penerbit", key="add_penerbit")
            deskripsi = st.text_area("Deskripsi", key="add_deskripsi", height=150)
            
            submitted = st.form_submit_button("Tambah Buku")
            
            if submitted:
                if not judul or not penerbit or not deskripsi:
                    st.error("Semua field harus diisi!")
                else:
                    # Cek apakah judul sudah ada
                    if judul in books['judulBuku'].values:
                        st.error(f"Buku dengan judul '{judul}' sudah ada!")
                    else:
                        try:
                            # Tambahkan buku baru
                            add_book(judul, tipe, tahun, penerbit, deskripsi)
                            st.success(f"Buku '{judul}' berhasil ditambahkan!")
                            
                            # Reset form menggunakan callback
                            st.rerun()
                        except Exception as e:
                            st.error(f"Terjadi kesalahan: {str(e)}")
    
    with tab3:
        st.subheader("Edit atau Hapus Buku")
        
        # Pilih buku untuk diedit/dihapus
        selected_book_title = st.selectbox("Pilih Buku", [""] + books['judulBuku'].tolist(), key="edit_book_select")
        
        if selected_book_title:
            selected_book = books[books['judulBuku'] == selected_book_title].iloc[0]
            selected_index = books[books['judulBuku'] == selected_book_title].index[0]
            
            # Form untuk mengedit buku
            with st.form("edit_book_form"):
                judul = st.text_input("Judul Buku", value=selected_book['judulBuku'], key="edit_judul")
                
                col1, col2 = st.columns(2)
                with col1:
                    tipe = st.selectbox("Tipe", ["Buku", "Jurnal", "Majalah", "Skripsi", "Tesis", "Disertasi", "Lainnya"], 
                                       index=["Buku", "Jurnal", "Majalah", "Skripsi", "Tesis", "Disertasi", "Lainnya"].index(selected_book['tipe']) 
                                       if selected_book['tipe'] in ["Buku", "Jurnal", "Majalah", "Skripsi", "Tesis", "Disertasi", "Lainnya"] else 0,
                                       key="edit_tipe")
                with col2:
                    tahun = st.number_input("Tahun", min_value=1900, max_value=2100, value=int(selected_book['tahun']), key="edit_tahun")
                
                penerbit = st.text_input("Penerbit", value=selected_book['penerbit'], key="edit_penerbit")
                deskripsi = st.text_area("Deskripsi", value=selected_book['deskripsi'], key="edit_deskripsi", height=150)
                
                col1, col2 = st.columns(2)
                with col1:
                    update_button = st.form_submit_button("Update Buku")
                with col2:
                    delete_button = st.form_submit_button("Hapus Buku", type="primary", help="Hapus buku ini dari database")
                
                if update_button:
                    if not judul or not penerbit or not deskripsi:
                        st.error("Semua field harus diisi!")
                    else:
                        try:
                            # Update buku
                            books.at[selected_index, 'judulBuku'] = judul
                            books.at[selected_index, 'tipe'] = tipe
                            books.at[selected_index, 'tahun'] = tahun
                            books.at[selected_index, 'penerbit'] = penerbit
                            books.at[selected_index, 'deskripsi'] = deskripsi
                            
                            # Simpan ke file CSV
                            books.to_csv('data_buku_bersih.csv', index=False)
                            
                            # Perbarui model TF-IDF
                            update_tfidf_models()
                            
                            st.success(f"Buku '{judul}' berhasil diperbarui!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Terjadi kesalahan: {str(e)}")
                
                if delete_button:
                    try:
                        # Konfirmasi penghapusan
                        if st.checkbox("Saya yakin ingin menghapus buku ini", key="confirm_delete_book"):
                            # Hapus buku
                            delete_book(selected_index)
                            st.success(f"Buku '{selected_book_title}' berhasil dihapus!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {str(e)}")
    
    with tab4:
        st.subheader("Kelola Data Pengguna")
        
        # Load user data
        user_data = load_user_data()
        
        if not user_data:
            st.info("Belum ada data pengguna.")
        else:
            # Tampilkan daftar pengguna
            st.write(f"Total pengguna: {len(user_data)}")
            
            # Buat DataFrame untuk tabel
            user_list = []
            for username, data in user_data.items():
                preferences = data.get("preferences", {}).get("keywords", "Belum ada")
                history_count = len(data.get("history", []))
                user_list.append({
                    "Username": username,
                    "Preferensi": preferences,
                    "Jumlah Buku": history_count
                })
            
            user_df = pd.DataFrame(user_list)
            
            # Tampilkan header tabel
            col1, col2, col3, col4 = st.columns([3, 3, 1, 1])
            with col1:
                st.markdown("**Username**")
            with col2:
                st.markdown("**Preferensi**")
            with col3:
                st.markdown("**Jumlah Buku**")
            with col4:
                st.markdown("**Aksi**")
            
            # Garis pemisah header
            st.markdown("---")
            
            # Tampilkan tabel dengan tombol aksi
            for i, row in user_df.iterrows():
                col1, col2, col3, col4 = st.columns([3, 3, 1, 1])
                
                with col1:
                    st.write(row['Username'])
                with col2:
                    st.write(row['Preferensi'])
                with col3:
                    st.write(row['Jumlah Buku'])
                with col4:
                    # Tombol aksi
                    if st.button("Hapus", key=f"del_{row['Username']}"):
                        st.session_state.user_to_delete = row['Username']
                
                # Tampilkan detail pengguna dalam expander
                with st.expander("Detail Pengguna"):
                    user_info = user_data[row['Username']]
                    
                    # Reset password
                    if st.button("Reset Password", key=f"reset_{row['Username']}"):
                        try:
                            # Generate random password
                            import random
                            import string
                            new_password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                            
                            # Update password
                            hashed_pw = hashlib.sha256(new_password.encode()).hexdigest()
                            user_info["password"] = hashed_pw
                            save_user_data(user_data)
                            
                            st.success(f"Password untuk {row['Username']} telah direset ke: {new_password}")
                        except Exception as e:
                            st.error(f"Terjadi kesalahan: {str(e)}")
                    
                    # Tampilkan riwayat buku
                    if "history" in user_info and user_info["history"]:
                        st.write("Riwayat Buku:")
                        for i, book_title in enumerate(user_info["history"]):
                            st.write(f"{i+1}. {book_title}")
                    else:
                        st.info("Pengguna belum memiliki riwayat buku.")
                
                # Garis pemisah
                st.markdown("---")
            
            # Konfirmasi penghapusan
            if 'user_to_delete' in st.session_state:
                username = st.session_state.user_to_delete
                st.warning(f"Apakah Anda yakin ingin menghapus pengguna {username}?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Ya, Hapus", key="confirm_delete"):
                        try:
                            delete_user(username)
                            st.success(f"Pengguna {username} berhasil dihapus.")
                            del st.session_state.user_to_delete
                            st.rerun()
                        except Exception as e:
                            st.error(f"Terjadi kesalahan: {str(e)}")
                with col2:
                    if st.button("Batal", key="cancel_delete"):
                        del st.session_state.user_to_delete
                        st.rerun()

# Fungsi untuk memperbarui model TF-IDF setelah perubahan data
def update_tfidf_models():
    global vectorizer, tfidf_matrix
    
    # Gabungkan semua fitur teks untuk vectorizer
    books['content'] = books['judulBuku'] + ' ' + books['tipe'] + ' ' + books['penerbit'] + ' ' + books['tahun'].astype(str) + ' ' + books['deskripsi']
    
    # Buat TF-IDF Vectorizer baru
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(books['content'])
    
    # Perbarui profil pengguna jika diperlukan
    update_user_profiles_after_data_change()
    
    # Simpan semua model dalam satu file
    save_all_models()

# Fungsi untuk memperbarui profil pengguna setelah perubahan data
def update_user_profiles_after_data_change():
    global user_profiles
    
    # Inisialisasi ulang profil pengguna dengan dimensi yang benar
    for username in user_profiles:
        user_profiles[username] = np.zeros((1, tfidf_matrix.shape[1]))
    
    # Perbarui profil berdasarkan preferensi dan riwayat
    for username, data in user_data.items():
        if username in user_profiles:
            # Update dari preferensi
            if "preferences" in data and "keywords" in data["preferences"]:
                keywords = data["preferences"]["keywords"]
                if keywords:
                    keyword_vector = vectorizer.transform([keywords])
                    user_profiles[username] += keyword_vector.toarray()
            
            # Update dari riwayat buku
            if "history" in data:
                for title in data["history"]:
                    book_indices = books[books['judulBuku'] == title].index.tolist()
                    if book_indices:
                        book_idx = book_indices[0]
                        update_user_profile(username, book_idx)

# Main function
def main():
    # Tampilkan navigasi
    show_navigation()
    
    # Tampilkan halaman yang sesuai
    if st.session_state.page == "home":
        # Jika user belum login, arahkan ke halaman login
        if not st.session_state.logged_in:
            st.session_state.page = "login"
            st.rerun()
        # Jika admin mencoba mengakses home, arahkan ke panel admin
        elif st.session_state.username == "admin":
            st.session_state.page = "admin"
            st.rerun()
        else:
            show_home_page()
    elif st.session_state.page == "login":
        show_login_page()
    elif st.session_state.page == "profile":
        # Jika user belum login, arahkan ke halaman login
        if not st.session_state.logged_in:
            st.session_state.page = "login"
            st.rerun()
        # Jika admin mencoba mengakses halaman profil, arahkan ke panel admin
        elif st.session_state.username == "admin":
            st.session_state.page = "admin"
            st.rerun()
        else:
            show_profile_page()
    elif st.session_state.page == "admin":
        # Jika bukan admin mencoba mengakses panel admin, arahkan ke login
        if not st.session_state.logged_in or st.session_state.username != "admin":
            st.session_state.page = "login"
            st.rerun()
        else:
            show_admin_page()

# Jalankan aplikasi
if __name__ == "__main__":
    main()
