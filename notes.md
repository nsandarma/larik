TERMINOLOGY:
1. Layout : Menentukan format penyimpanan matriks.
  - Row-major
  - Col-major


# Blas:
- Pastikan dimensi sesuai: A(m x k), B(k x n), C(m x n).
- lda & ldb: 
  Menentukan jarak (dalam jumlah elemen) antar baris matriks.
  -- Kegunaan :
  Digunakan untuk menangani matriks yang mungkin memiliki padding (ruang tambahan) atau merupakan bagian dari matriks yang lebih besar (submatriks).
  -- Contoh:
  jika matriks A berukuran m x k dan disimpan dalam row-major tanpa padding (dense matrix). Maka lda = k (jumlah kolom A).
  jika ada padding atau A adalah bagian dari matriks lebih besar, lda bisa lebih besar dari k.

- Mengapa Leading Dimension penting ?
  Tidak semua matriks disimpan secara "Padat" di memori. kadang-kadang, matriks memiliki
  ruang tambahan (padding). untuk alasan optimasi atau karena merupakan bagian dari matriks lebih besar.
  lda,ldb,ldc memberi tahu fungsi blas cara mengakses elemen matriks dengan benar.






