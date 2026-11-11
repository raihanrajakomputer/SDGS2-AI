# Klasifikasi Gangguan Kesehatan Daun Singkong Menggunakan CNN dan EfficientNetB0 dengan Visualisasi Grad-CAM untuk Mendukung SDG 2

Tugas Besar Kecerdasan Buatan. Model mengenali 5 kelas daun singkong
(CBB, CBSD, CGM, CMD, Healthy) dari foto, memakai CNN dan EfficientNetB0.

## Dataset
TensorFlow Datasets `cassava` (iCassava 2019, Mwebaze et al., 2019).
Pembagian resmi: train 5.656, validation 1.889, test 1.885.

## Hasil akhir (data test)
- Model terpilih: efficientnet_frozen
- Accuracy: 0.7210
- Macro-F1: 0.6360
- Balanced accuracy: 0.6653

Model dipilih memakai validation macro-F1. Detail per kelas ada di
`reports/classification_report_test.txt`.

## Menjalankan aplikasi
    pip install -r requirements.txt
    python app.py

## Keterbatasan
- Dataset berasal dari Uganda, domain Indonesia belum tervalidasi.
- Satu gambar hanya punya satu label.
- Prediksi berbasis foto, bukan diagnosis laboratorium.
- Skor softmax bukan kepastian. Foto non-daun tetap diberi salah satu kelas.

## Anggota
- Raihan Al Rajab
- Fatimah Alya Sabhira
- Anggun Dian Mutiara
