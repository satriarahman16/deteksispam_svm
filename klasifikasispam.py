import streamlit as st
import joblib

# ===============================
# 🔧 Load Model dan Vectorizer
# ===============================
try:
    model = joblib.load("model_klasifikasi_spam.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
except:
    st.error("❌ Pastikan file model_klasifikasi_spam.joblib dan vectorizer.joblib ada di folder yang sama!")
    st.stop()

# ===============================
# 🎨 Judul Aplikasi
# ===============================
st.title("📩 Aplikasi Deteksi Pesan Spam")
st.write("**Masukkan pesan SMS dan periksa apakah itu **spam** atau bukan menggunakan model SVM yang telah dilatih.**")

# ===============================
# 💬 Input Teks Pengguna
# ===============================
input_text = st.text_area("Ketik atau tempel pesan SMS di bawah ini:", height=150)
accuracy =0.9441
# ===============================
# 🔍 Tombol untuk Prediksi
# ===============================
if st.button("Periksa Pesan"):
    if input_text.strip() == "":
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        # Transformasi teks dengan vectorizer
        text_vec = vectorizer.transform([input_text])
        prediction = model.predict(text_vec)[0]

        # ===============================
        # 🎯 Hasil Prediksi
        # ===============================
        if prediction == 0:
            st.success("✅ Pesan ini **bukan spam.** Aman dibaca.")
        else:
            st.error("⚠️ Pesan ini **SPAM!** Harap berhati-hati dengan tautan atau informasi di dalamnya.")

# ===============================
# 📊 Info Model (Opsional)
# ===============================
with st.expander("ℹ️ Tentang Model"):
    st.markdown(f"""
    - **Algoritma:** Support Vector Machine (SVM)  
    - **Kernel:** Linear  
    - **Vectorizer:** TfidfVectorizer (Unigram + Bigram)  
    - **Tujuan:** Mendeteksi pesan SMS spam dalam bahasa Indonesia 
    - **Akurasi Model:** {accuracy*100:.2f}% 
    """)

st.markdown("---")
st.caption("🧠 Developed in October – © 2025")
