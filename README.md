# REDDIT EMOTION CLASSIFICATION

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Sentiment_analysis.svg/1200px-Sentiment_analysis.svg.png" alt="Sentiment Analysis" width="500" height="300" />
</p>
<p align="center"><small>Sumber Image: <a href="https://commons.wikimedia.org/wiki/File:Sentiment_analysis.svg">Wikipedia Commons</a></small></p>

---

## 📑 Table of Content

1. [Deskripsi Project](#-deskripsi-project)

   * [Latar Belakang](#latar-belakang)
   * [Tujuan Pengembangan](#tujuan-pengembangan)
2. [Sumber Dataset](#-sumber-dataset)
3. [Preprocessing dan Pemodelan](#-preprocessing-dan-pemodelan)

   * [Pemilihan Atribut](#pemilihan-atribut)
   * [Preprocessing Data](#preprocessing-data)
   * [Pemodelan](#pemodelan)
4. [Langkah Instalasi](#-langkah-instalasi)

   * [Software Utama](#software-utama)
   * [Dependensi](#dependensi)
   * [Menjalankan Sistem Prediksi](#menjalankan-sistem-prediksi)
   * [Pelatihan Model](#pelatihan-model)
5. [Hasil dan Analisis](#-hasil-dan-analisis)

   * [Evaluasi Model](#evaluasi-model)
6. [Sistem Sederhana Streamlit](#-sistem-sederhana-streamlit)
7. [Biodata](#-biodata)

---

## 📚 Deskripsi Proyek



Proyek ini bertujuan untuk **mengembangkan sistem klasifikasi emosi otomatis** pada teks curhatan (*confessions*) di platform Reddit. Sistem ini membandingkan kinerja antara model **Deep Learning Tradisional** dan model **Transformer Modern** untuk memprediksi nuansa emosi pengguna secara akurat.

---

### Latar Belakang

Analisis emosi pada teks media sosial penting untuk memahami kondisi psikologis dan ekspresi pengguna. Pada proyek ini, emosi diklasifikasikan ke dalam **6 emosi dasar (Ekman)**:

* 🤬 **Anger** (Marah)
* 😱 **Fear** (Takut)
* 😂 **Joy** (Senang)
* 🥰 **Love** (Cinta)
* 😭 **Sadness** (Sedih)
* 😲 **Surprise** (Terkejut)

---

### Tujuan Pengembangan

1. Membangun **model klasifikasi teks** untuk memprediksi emosi berdasarkan curhatan pengguna Reddit.
2. Melakukan **evaluasi performa model**, khususnya perbandingan **Bi-LSTM** dengan **BERT** dan **RoBERTa** dalam menangani bahasa informal dan slang.
3. Mengembangkan **dashboard interaktif berbasis Streamlit** untuk pengujian model secara real-time.

---

## 📊 Penjelasan Dataset dan Preprocessing



### Sumber Dataset

Dataset yang digunakan merupakan dataset publik:

### One Million Reddit Confessions

* **Deskripsi**: Kumpulan teks postingan dari subreddit `r/confession`.
* **Labeling**: Dataset asli tidak memiliki label. Oleh karena itu dilakukan **pseudo-labeling** menggunakan model pre-trained `DistilBERT-Emotion`.
* **Jumlah Data**: 10.000 sampel terpilih.
* **Struktur Data**:

  * Teks gabungan (`Title` + `Selftext`)
  * Label emosi (0–5)

#### 📥 Link Dataset

Dataset yang digunakan pada proyek ini dapat diakses melalui Google Drive:

🔗 **[Download Dataset Reddit Emotion](https://www.kaggle.com/datasets/pavellexyr/one-million-reddit-confessions/versions/1?resource=download)**

> **Catatan**: Dataset tidak disertakan langsung di repository GitHub untuk menjaga ukuran repository tetap ringan.

---

## 🧑‍💻 Penjelasan Model dan Preprocessing



### Pemilihan Atribut

| Kolom      | Tipe    | Deskripsi                                                                    |
| ---------- | ------- | ---------------------------------------------------------------------------- |
| clean_text | String  | Gabungan judul dan isi postingan yang telah dibersihkan                      |
| label      | Integer | Kategori emosi (0: Anger, 1: Fear, 2: Joy, 3: Love, 4: Sadness, 5: Surprise) |

---

### Preprocessing Data

1. **Cleaning**: Menghapus baris dengan isi `[removed]` dan `[deleted]`.
2. **Balancing Data**: Menggunakan **Random Oversampling** sehingga setiap kelas memiliki ±1.600 data.
3. **Tokenisasi**:

   * **Bi-LSTM**: Tokenizer Keras (max length = 150).
   * **Transformer**: AutoTokenizer (WordPiece/BPE, max length = 128).

---

### Pemodelan

Model yang digunakan:

1. **Bi-LSTM (Bidirectional LSTM)**

   * Dua arah pemrosesan sekuens (forward & backward).
   * Embedding layer dilatih dari awal.

2. **BERT (Bidirectional Encoder Representations from Transformers)**

   * Menggunakan `bert-base-uncased`.
   * Fine-tuning pada data Reddit.

3. **RoBERTa (Robustly Optimized BERT Approach)**

   * Varian BERT dengan optimasi hyperparameter yang lebih kuat.

---

## 🔧 Langkah Instalasi

### Software Utama

* Python **3.10+**
* Editor: **VS Code** atau **Google Colab**

---

### Dependensi

Seluruh dependensi tersedia pada file `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

### Menjalankan Sistem Prediksi

Untuk menjalankan dashboard Streamlit:

```bash
streamlit run app.py
```

---

### Pelatihan Model

Karena ukuran model **Transformer (BERT & RoBERTa)** sangat besar (>400MB), file model **tidak disimpan langsung di GitHub**.

Sebagai gantinya, seluruh folder **Models** disimpan di **Google Drive** dan dapat diunduh secara manual.

#### 📥 Cara Download Folder Models dari Google Drive

1. Buka link Google Drive berikut:

   🔗 **[Download Folder Models](https://drive.google.com/drive/folders/1paaNLPDH6uoVDun1yo_rCaYWAwQ-uuXl?usp=sharing)**

2. Setelah halaman Google Drive terbuka:

   * Klik kanan pada folder **Models**
   * Pilih **Download**

3. Google Drive akan mengompres folder menjadi file `.zip`. Tunggu hingga proses download selesai.

4. Ekstrak file `.zip` tersebut.

5. Letakkan folder **Models** hasil ekstraksi ke dalam direktori utama proyek, sehingga struktur folder menjadi:

```
Dashboard UAP/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│
└── Models/
    │── bert_model/
    │── roberta_model/
    │── lstm_model/
```

6. Setelah folder **Models** berada di lokasi yang benar, aplikasi Streamlit dapat dijalankan tanpa error.

---

## 🔍 Hasil Evaluasi dan Analisis Perbandingan



### Evaluasi Model

Evaluasi dilakukan menggunakan **20% data uji** dengan metrik:

* Accuracy
* Precision
* Recall
* F1-Score (Macro)

#### Tabel Perbandingan Performa Model

| Model   | Akurasi | Macro F1-Score | Status      |
| ------- | ------- | -------------- | ----------- |
| Bi-LSTM | 70%     | 0.71           | Baseline    |
| RoBERTa | 77%     | 0.77           | Sangat Baik |
| BERT    | 81%     | 0.81           | Terbaik 🏆  |

#### Analisis

* **Dominasi Transformer**: BERT unggul signifikan dibanding Bi-LSTM.
* **Pemahaman Konteks**: Transformer lebih efektif dalam memahami konteks kalimat kompleks dan slang.
* **Keterbatasan LSTM**: Kesulitan membedakan emosi yang mirip seperti *Joy* dan *Love*.

---

## 🎓 Panduan Menjalankan Sistem Website Secara Lokal

Bagian ini menjelaskan **langkah-langkah lengkap dan berurutan** untuk menjalankan aplikasi klasifikasi emosi berbasis **Streamlit** secara lokal.

---

### 1️⃣ Persiapan Lingkungan

Pastikan perangkat telah memenuhi prasyarat berikut:

* Python **versi 3.10 atau lebih baru**
* Git sudah terinstal
* Koneksi internet (untuk download model & dataset)

Cek versi Python:

```bash
python --version
```

---

### 2️⃣ Clone Repository

Clone repository project ke komputer lokal:

```bash
git clone <URL_REPOSITORY_GITHUB>
cd Dashboard\ UAP
```

---

### 3️⃣ Instalasi Dependensi

Install seluruh library yang dibutuhkan menggunakan file `requirements.txt`:

```bash
pip install -r requirements.txt
```

Tunggu hingga seluruh proses instalasi selesai tanpa error.

---

### 4️⃣ Download Model dan Dataset

Karena ukuran file besar, **model dan dataset tidak tersedia langsung di GitHub**.

#### 🔹 Download Model

* Unduh folder **Models** dari Google Drive (lihat bagian *Pelatihan Model*)
* Ekstrak dan letakkan folder `Models` di direktori utama project

#### 🔹 Download Dataset

* Unduh dataset dari Google Drive (lihat bagian *Penjelasan Dataset*)
* Dataset tidak wajib diletakkan di repo jika hanya menjalankan prediksi

Struktur folder yang benar:

```
Dashboard UAP/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│
└── Models/
    │── bert_model/
    │── roberta_model/
    │── lstm_model/
```

---

### 5️⃣ Menjalankan Aplikasi Streamlit

Jalankan perintah berikut dari direktori utama project:

```bash
streamlit run app.py
```

Jika berhasil, terminal akan menampilkan pesan seperti:

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Buka browser dan akses:

👉 **[http://localhost:8501](http://localhost:8501)**

---

### 6️⃣ Cara Menggunakan Aplikasi

1. Masukkan teks curhatan (bahasa Inggris) pada kolom input.
2. Pilih mode prediksi:

   * **Single Model** (Bi-LSTM / BERT / RoBERTa)
   * **Komparasi Model**
3. Klik tombol **Predict**.
4. Sistem akan menampilkan:

   * Prediksi emosi utama
   * Confidence score
   * Grafik probabilitas

---

### ⚠️ Troubleshooting Umum

* **Error model tidak ditemukan**: Pastikan folder `Models` berada di direktori utama.
* **ModuleNotFoundError**: Jalankan ulang `pip install -r requirements.txt`.
* **Port 8501 digunakan**: Gunakan `streamlit run app.py --server.port 8502`.

---

## 👤 Biodata

**Muhammad Daffa Alprianda Raihan**
📘 NIM: 202210370311039
🎓 Kelas: Machine Learning A?D
🏛️ Universitas Muhammadiyah Malang
