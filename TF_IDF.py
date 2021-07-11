import numpy as np
import pandas as pd
from collections import Counter


# method preprocessing
def preproccessing(corpus):
    import re
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    # create stemmer (sesuai penggunaan library Sastrawi)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # create stopword remover (sesuai penggunaan library Sastrawi)
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()

    preprocessed_corpus = []                                                                                    # buat corpus baru (kosong)
    for text in corpus:                                                                                         # untuk setiap kalimat pada corpus:
        if text is not np.nan:
            text = re.sub("\.|\,|\)|\(|\:", "", text)                                                               # hapus "." "," "(" ")"
            text = re.sub("\s+[0-9]{1,}\s+|\-", " ",text)                                                           # ganti bilangan (*hanya yang terpisah) dan tanda penghubung (-) menjadi spasi
            text = stemmer.stem(text)                                                                               # mengubah kata ke bentuk kata dasar + mengganti huruf kapital ke huruf kecil
            text = stopword.remove(text)                                                                            # menghapus stopword
            preprocessed_corpus.append(text)                                                                        # tambahkan kalimat ke corpus baru

    return preprocessed_corpus                                                                                  # corpus yang telah dilakukan preprocessing


# method tokenisasi corpus
def tokenize(corpus):
    grouped_corpus = ""                                                                                         # buat corpus baru (kosong)
    for text in corpus:                                                                                         # untuk setiap kalimat pada corpus:
        grouped_corpus += text + " "                                                                            # menambahkan setiap kalimat ke corpus baru dengan spasi di akhir kalimat

    words_list = grouped_corpus.split(" ")                                                                       # tokenisasi corpus baru (dengan pengulangan)
    words_list.pop()                                                                                             # menghapus kata/token terakhir (*spasi)
    word_list = list(Counter(words_list))                                                                        # tokenisasi (tanpa pengulangan)
    joined_corpus = dict(Counter(words_list))

    # print("token_len:\n",len(word_list))
    return word_list, joined_corpus                                                                                            # token dari corpus (dimensi: kolom -> fitur -> kata)


# method untuk menghitung df (document frequency)
def get_df(corpus, word_list):
    list_df = []                                                                                                # list untuk menyimpan nilai df dari fitur/kata
    for word in word_list:                                                                                      # untuk setiap kata hasil preprocessing+tokenisasi: (looping pertama)
        df = 0                                                                                                  # inisialisasi nilai df = 0
        for text in corpus:                                                                                     # untuk setiap kalimat pada corpus: (looping kedua)
            Words = list(Counter(text.split(" ")))                                                              # mengindeks kata pada kalimat (tanpa pengulangan)
            if word in Words:                                                                                   # jika kata (pada looping pertama) terdapat pada list kata (pada looping kedua), maka:
                df += 1                                                                                         # increment nilai df dari fitur/kata
                continue                                                                                        # lanjut ke kalimat/dokumen berikutnya (looping kedua)

        list_df.append(df)                                                                                      # masukkan nilai df tiap fitur/kata pada list

    # print("df_len: ", len(list_df))
    return list_df


# method untuk melakukan pemilihan fitur (feature/term selection) dengan treshold df>1
def term_selection(df, wordlist, dfTresh):
    new_df = []
    new_wordlist = []
    for i in range(len(df)):
        if df[i] >= dfTresh:
            new_df.append(df[i])
            new_wordlist.append(wordlist[i])

    return new_df, new_wordlist


# method untuk menghitung tf (log - term frequency)
def get_tf(corpus, word_list, freq):
    tf = []                                                                                                     # list untuk menyimpan bobot TF semua dokumen9888
    for text in corpus:                                                                                         # untuk setiap Kalimat pada corpus:
        Words = dict(Counter(text.split(" ")))                                                                  # mengindeks kata beserta frekuensinya dari kalimat
        weight = [round(1+(np.log10(Words[word]/freq[word])),3) if word in Words else 0 for word in word_list]             # pembobotan log-tf tiap kalimat/dokumen (dimensi: kolom -> fitur -> bobot tf dari tiap kata pada kalimat)
        tf.append(weight)                                                                                       # memasukkan hasil bobot tiap kalimat/dokumen ke dalam list yang telah dibuat

    # print("\ntf_len: ",len(tf))
    return tf                                                                                                   # bobot log-tf (dimensi: baris*kolom -> dokumen*fitur -> kalimat*bobot tiap kata pada kalimat)


# method untuk menghitung idf (inverse document frequency)
def get_idf(corpus, df):
    N = len(corpus)
    idf = [round(np.log10(N / df), 3) for df in df]                                                        # pembobotan idf tiap fitur/kata

    return idf


# method untuk menghitung tf-idf
def get_tf_idf(corpus, dfTresh):
    ori_corpus = corpus
    corpus = preproccessing(corpus)                                                                             # Preprocessing corpus
    word_list, joined_corpus = tokenize(corpus)                                                                                # Tokenisasi corpus
    df = get_df(corpus, word_list)
    df, word_list = term_selection(df,word_list,dfTresh)
    # print("df_len: ", len(df), "\tterm_len: ", len(word_list))
    tf = get_tf(corpus, word_list, joined_corpus)                                                                              # perhitungan tf (Log-Term Frequency)
    idf = get_idf(corpus, df)                                                                                   # perhitungan idf (Inverse-Document Frequency)

    tf_idf = np.array(tf, copy=True)                                                                            # membuat matrix tf-idf dengan dimensi yang sama seperti tf
    for i in range(len(tf_idf)):                                                                                # looping baris:
        for j in range(len(tf_idf[i])):                                                                         # looping kolom:
            tf_idf[i][j] = round((tf[i][j] * idf[j]), 3)                                                        # pembobotan tf-idf tiap fitur/kata

    tf_idf_df = pd.DataFrame(tf_idf, columns=word_list)                                                         # dataframe
    tf_idf_df.insert(0, 'judul skripsi', ori_corpus)

    return tf_idf_df                                                                                            # bobot tf-idf (dimensi: baris*kolom -> dokumen*fitur -> kalimat*bobot tiap kata pada kalimat)


# maingf 3
if __name__ == '__main__':

    corpus = [
            "Klasifikasi Arah Gerak Diagonal Mata dan Normalisasi Sinyal Electrooculography dengan Metode Diferensiasi",
            "Evaluasi Manajemen Resiko Teknologi Informasi Menggunakan COBIT 5 dengan Domain EDM03 dan APO12 (Studi Kasus pada UPT-TIK Universitas Brawijaya)",
            "Sistem Biometrik Gerakan Tanda Tangan Menggunakan Sensor MPU6050 dengan Metode Backpropagation",
            "Perancangan Desain Interaksi Aplikasi Malang Sehat Modul Pendataan dan Monitoring Kesehatan Masyarakat Kota Malang dengan menggunakan Metode Human-Centered Design",
            "Penggunaan Jalur dalam Multipath Routing secara Proporsional Berdasarkan Kebutuhan Bandwidth Pengguna pada Software Defined Networking",
            "Pengembangan Aplikasi Rekomendasi Musik berdasarkan Detak Jantung pada Platform Android",
            "Perancangan dan Evaluasi User Experience Aplikasi Mobile Pembelajaran Online di Jurusan Sistem Informasi menggunakan Pendekatan Human-Centered Design dan TUXEL",
            "Pengaruh Implementasi Metode Pembelajaran Teams Games Tournaments terhadap Variabel Motivasi, Minat Belajar, dan Hasil Belajar Peserta Didik pada Mata Pelajaran Desain Grafis Percetakan di SMK Negeri 12 Malang",
            "Pengembangan Aplikasi Wedding.co berbasis Web (Studi Kasus: Kota Malang)",
            "Implementasi Saving Energy Protokol Dynamic Source Routing (DSR) pada Mobile Ad Hoc Network (MANET) dengan Metode Sleep Mode",
            "Pengembangan Sistem Manajemen Pemeliharaan Preventif Mesin berbasis Web (Studi Kasus: PT Valeo AC Indonesia)",
            "Pengembangan Sistem Manajemen Pemeliharaan Preventif Mesin berbasis Web (Studi Kasus: PT Valeo AC Indonesia)",
            "Evaluasi dan Perbaikan Antarmuka Aplikasi Web Daily Plan LPMI Al-Izzah Kota Batu menggunakan Pendekatan Human Centered Design",
            "Pemodelan dan Rekomendasi Proses Bisnis menggunakan Metode Business Process Improvement (BPI) (Studi Kasus CV Wisa Tunggal Perkasa)",
            "Sistem Pemantauan Daya pada Wireless Sensor Network menggunakan Algoritma Priority Scheduling",
            "Pengembangan Sistem Informasi Pemesanan Paket Wisata (Studi Kasus : Agen Wisata Liburan Sekolah)",
            "Pengembangan Sistem Informasi Monitoring Pengelolaan Dana Bantuan Operasional Sekolah berbasis Website Studi Kasus : Korwilcam Bidang Pendidikan Kecamatan Sempor",
            "Pengembangan Sistem Informasi Kurban Pada Proses Pendukung dan Pendaftaran Kurban (Studi Kasus: Masjid Ibnu Sina Kota Malang)",
            "Evaluasi dan Perbaikan Proses Bisnis Menggunakan Business Process Improvement (BPI) (Studi Kasus: Bidang Mutasi, Badan Kepegawaian dan Pengembangan Sumber Daya Manusia Kota Batu)",
            "Analisis Segmentasi Pelanggan Kartu Prabayar Kabupaten Malang dengan RFM Model Menggunakan Metode Fuzzy C-Means Clustering (Studi Kasus : PT. XYZ)",
            "Implementasi High Performance Computing Cluster Menggunakan Rocks Cluster",
            "Evaluasi dan Perbaikan Alur dan Navigasi Website Event Surabaya Menggunakan Pendekatan Human Centered Design (HCD)",
            "Sistem Pendeteksi Kualitas Tanah Tanaman Kedelai Menggunakan Metode K-Nearest Neighbor (K-NN) dengan Arduino Nano",
            "Pembangunan Sistem Aplikasi Kolaborasi Peneliti berbasis Website menggunakan Metode Rapid Application Development",
            "Pengembangan Aplikasi Cashless Payment menggunakan Teknologi QR Code berbasis Android pada Kantin Dharma Wanita Filkom UB",
            "Pengaruh Energi Terhadap Pengiriman Data Pada Protokol Fisheye State Routing (FSR) Dalam Mobile Ad Hoc Network (Manet)",
            "Evaluasi dan Perbaikan Antarmuka Aplikasi Web Daily Plan LPMI Al-Izzah Kota Batu Menggunakan Pendekatan Human Centered Design",
            "Pengembangan Sistem Monitoring Kanal Air sebagai Sarana Penanggulangan Banjir",
            "Sistem Monitoring Lahan Parkir berbasis Bluetooth Low Energy (BLE)",
            "Penerapan REST API dalam Pengembangan Aplikasi Pemesanan Rental Mobil berbasis Web dan Mobile (Studi Kasus: CV. Dwi Cipta Rent Car)",
            "Rekomendasi Produk UMKM Kabupaten Malang menggunakan Metode Analytical Hierarchy Process (AHP) dan Simple Additive Weighting (SAW) (Studi Kasus: Rumah Kreatif BUMN Telkom Kabupaten Malang)",
            "Analisis Penerimaan Penggunaan Aplikasi Antrian Online pada Mal Pelayanan Publik Sidoarjo berdasarkan Unified Theory of Accepptance and Use of Technology (UTAUT)",
            "Implementasi Representational State Transfer pada Sistem Administrasi Laporan Bulanan (Studi Kasus : Navila Resto)",
            "Evaluasi Usability Aplikasi Golife berbasis Android dari Perspektif Pengguna dengan menggunakan Metode Use Questionnaire dan Pendekatan Human-Centered Design",
            "Analisis dan Perancangan Sistem Informasi Akademik dan Keuangan TK Tunas Bangsa",
            "Pengembangan Sistem Informasi Sewa Mobil dan Paket Wisata berbasis Web menggunakan Teknologi Framework Laravel (Studi Kasus: Mobil Kampus)",
            "Implementasi Teknologi AWS Cloud Dalam Pengembangan Aplikasi Ujian Online Berbasis Website Menggunakan Framework Codeigniter (Studi Kasus: SMAN 1 Jombang dan MAN 9 Jombang)",
            "Rancang Bangun Sistem Monitoring Cuaca Low Power Berbasis Mikrokontroler",
            "Pengembangan Aplikasi Ojek Daring untuk Motor Ramah Difabel (Toradi) berbasis Android",
            "Pengembangan Aplikasi Berbasis Android untuk Pemantauan Masalah dalam Kegiatan Praktik Kerja Industri Siswa di SMK Negeri 2 Malang Berdasarkan Model Extreme Programming (LAPAN)",
            "Prediksi Pertumbuhan Jumlah Penduduk Kota Malang menggunakan Metode Average-based Fuzzy Time Series",
            "Implementasi Time Redundancy pada Sistem Monitoring Sungai yang berbasis Mikrokontroler NodeMCU dan LabVIEW",
            "Pengaruh Model Pergerakan Node terhadap Konsumsi Energi Protokol Routing Location Aided Routing (LAR) pada Mobile Ad Hoc Network (MANET)",
            "Object Following Robot berbasis Pembacaan Jarak menggunakan Metode PID Controller",
            "Implementasi Prototype Kapal sebagai Sistem Monitoring Kualitas Air menggunakan Algoritme Naïve Bayes",
            "Navigasi Robot Beroda menggunakan Algoritma SVM (Support Vector Mechines)",
            "Pengembangan Media Pembelajaran Interaktif berbasis Website pada Mata Pelajaran Administrasi Infrastruktur Jaringan dengan Model Pengembangan Four-D di SMKN 3 Malang",
            "Perancangan Sistem Plug and Play pada Otomasi Lampu menggunakan nRF24L01 dan Protokol MQTT melalui Smartphone",
            "Implementasi Protokol Routing Directed Diffusion pada WSN dengan Modul Komunikasi LoRa",
            "Perancangan Enterprise Architecture Menggunakan TOGAF ADM pada PT. Hafintech Prima Mandiri",
            "Implementasi Algoritma K-Nearest Neighbor Pada Database Menggunakan Bahasa SQL",
            "Pengembangan Aplikasi Perangkat Bergerak “Tahu Kediri” Informasi Wisata Kediri Dengan Metode Human Centered Design",
            "Rancang Bangun Sistem Klasifikasi Rasa Permen Karet Berdasarkan Warna Dengan Metode K-Nearest Neighbor (KNN)",
            "Pemanfaatan API Youtube dalam Pengembangan Aplikasi Portal Video Penangkaran Kenari untuk Peternak Kenari Berbasis Android",
            "Sistem Navigasi Waypoint Pada Robot Beroda Berdasarkan Global Positioning System Dan Filter Kalman",
            "Pengenalan Jenis Kelamin dan Rentang Umur berdasarkan Suara menggunakan Metode Backpropagation Neural Network",
            "Implementasi Mekanisme Publish-Subscribe pada Pemantauan Kehadiran Beacon menggunakan Protokol Bluetooth Low Energy",
            "Pengembangan Platform Pengolahan Data Sensor Internet of Things Berjenis Streaming dengan Komputasi Terdistribusi Menggunakan Spark Streaming",
            "Implementasi untuk Prediksi Jumlah Kedatangan Wisatawan Domestik Pulau Bali menggunakan Algoritme Performance Improved Holt winters",
            "Pengembangan Modul Inventori Pada Supply Chain Management PT Sampoerna Pagi",
            "Desain Antarmuka Pengguna Sistem Informasi Pengelolaan Penyimpanan Bukti Fisik Penilaian Akreditasi Sekolah Menggunakan Goal Directed Design (Studi Kasus: Sekolah Menengah Pertama Islam Sabilurrosyad Malang)",
            "Optimasi Kebutuhan Gizi Menggunakan Algoritme Evolution Strategies Pada Balita Dan Ibu Menyusui",
            "Pengujian Usability Untuk Aplikasi Silsilah Keluarga Myheritage Dalam Memenuhi Kebutuhan Umat Islam Dengan Pendekatan Kuantitatif",
            "Pengembangan Chatbot Yanies Cookies Untuk Pemesanan Kue Kering Berbasis Dialogflow",
            "Implementasi Fuzzy Analytical Hierarchy Process Untuk Menentukan Berita Utama (Headline News) di Kavling 10",
            "Evaluasi Sistem E-rapor Direktorat PSMA Terhadap Aspek Usability dan Utility (Studi Kasus: SMAN 1 Tuban)",
            "Implementasi Komunikasi Multi-Hop Menggunakan Metode Controlled Flooding Pada Wireless Sensor Network Berbasis LoRa",
            "Pengembangan Sistem Informasi Manajemen Ternak Burung Kenari Berbasis Web",
            "Pengembangan Data Warehouse untuk Evaluasi Pembelajaran Matakuliah Berdasarkan Data Kuesioner Mahasiswa di SIAM dan Rekapitulasi Presensi Dosen (Studi Kasus Teknologi Informasi Fakultas Ilmu Komputer)",
            "Pengaruh Kemampuan Berpikir Kritis dan Berpikir Logis Siswa Terhadap Kemampuan Belajar Secara Kolaboratif Pada Jurusan Teknik Komputer dan Jaringan di SMK Negeri 2 Malang",
            "Evaluasi User Experience Pada Aplikasi Programming HUB Menggunakan Indikator UX Honeycomb",
            "Analisis Sentimen pada Ulasan Pengguna MRT Jakarta Menggunakan Metode Neighbor-Weighted K-Nearest Neighbor dengan Seleksi Fitur Information Gain",
            "Pengembangan E-Modul Berbasis Electronic Publication (EPUB) Menggunakan Model Pengembangan ADDIE Pada Mata Pelajaran Pemrograman Dasar di SMK Negeri 4 Malang",
            "Implementasi RFID untuk Mengatasi Untraceable Book Pada Perpustakaan",
            "Evaluasi Tata Kelola Teknologi Informasi Pada Diskominfosantik Kabupaten Bekasi Menggunakan Kerangka Kerja COBIT 5 Subdomain EDM04 dan APO07",
            "Pengujian Usability untuk Aplikasi Silsilah Keluarga FamilySearch Tree dengan Pendekatan Kuantitatif",
            "Prediksi Persentase Penyelesaian Permohonan Hak Milik menggunakan Metode Extreme Learning Machine (ELM) (Studi Kasus: Badan Pertanahan Nasional Kabupaten Malang)",
            "Pengembangan Aplikasi Penentu Kadar Hidrokuinon berbasis Android (Studi Kasus: Laboratorium Kimia Analitik Fakultas MIPA UB)",
            "Evaluasi Aplikasi Silsilah Keluarga FamilySearch Dengan Pengujian Usability",
            "Analisis dan Perbaikan Usability Pada Aplikasi Ker Menggunakan Metode Usability Testing dan System Usability Scale (SUS)",
            "Pengembangan Plugin QGIS Untuk Mengakses Peta Geologis Seluruh Indonesia",
            "Implementasi Plugin Notifikasi Sebagai Media Integrasi Antara E-Learning Moodle dengan BOT Telegram (Studi Kasus : Bimbingan Belajar The Second School)",
            "Pengembangan Aplikasi Monitoring Kartu Menuju Sehat (KMS) Terintegrasi berbasis Mobile",
            "Pengembangan Sistem Informasi Manajemen Barang (Studi Pada Toko Kertas MBC)",
            "Klasifikasi Jenis Kelamin Berdasarkan Suara Menggunakan Metode Learning Vector Quantization",
            "Sistem Pendukung Keputusan Untuk Pengelompokan Barang Terjual Pada PT Dasema Digi Persada Dengan Metode K-Means Clustering",
            "Desain Datapath Arsitektur Komputer MIC-1 8 bit Menggunakan IC74XX",
            "Implementasi Fault Tolerant System Menggunakan Metode Self-Purging Redundancy Pada Sistem Pendeteksi Kebakaran",
            "Implementasi Sistem Rekomendasi Tempat Pembelian Oleh-Oleh Khas Malang berbasis Perangkat Bergerak",
            "Pengembangan Aplikasi Web Reservasi Paket Wisata menggunakan MERN Stack (Studi Kasus: Zona Tamasya Tour Organizer)",
            "Implementasi Algoritme Fuzzy C-Means dengan Particle Swarm Optimization (FCMPSO) untuk Pengelompokan Proses Berpikir Siswa dalam Proses Belajar",
            "Pengembangan Aplikasi Pelatihan Pendamping Pusat Studi dan Layanan Disabilitas Universitas Brawijaya (PSLD UB) Berbasis Android",
            "Implementasi Sistem Transmisi Data Sensor Healthcare Berbasis Zigbee Dengan Protokol Ad-hoc On Demand Distance Vector",
            "Klasifikasi Kategori Buku Ilmu Agama Islam Menggunakan Metode Naive Bayes Dan Seleksi Fitur Information Gain",
            "Pengembangan Aplikasi Android Rekomendasi Tempat Pembelian Kuliner Korea dengan GDSS dan LBS (Studi Kasus: Kota Malang)",
            "Prediksi Permintaan Keripik Buah dengan Metode Jaringan Syaraf Tiruan Backpropagation (Studi Kasus: CV. Arjuna 999)",
            "Analisis Sentimen Penggunaan Tol Trans Jawa Periode Mudik Lebaran 2019 dengan Metode K-Nearest Neighbor dan Seleksi Fitur Information Gain",
            "Penerapan Particle Swarm Optimization Pada Algoritme K-Means Untuk Pengelompokan Proses Berpikir Siswa Dalam Belajar",
            "Rancang Bangun Sistem Monitoring dan Klasifikasi Lingkungan Hidup Larva Lalat Tentara Hitam (Hermetia Illucens) dengan Metode K-Nearest Neighbor (K-NN)",
            "Klasifikasi Hoaks Kesehatan di Media Sosial menggunakan Support Vector Machine"
    ]

    tf_idf = get_tf_idf(corpus, dfTresh=1)
    print("\njumlah data:", len(tf_idf), "\njumlah fitur:", len(tf_idf.columns)-1)
    print("\nTF-IDF:\n",tf_idf)
    print()
    print(tf_idf.iloc[:,1:])
    # print("\nterm:\n", fitur)

    # tf_idf_df = pd.DataFrame(tf_idf, columns=fitur)
    # print("\ntf_idf_dataframe:")
    # print(tf_idf_df)