# Tugas Kecil 2 IF2211 Strategi Algoritma

Voxelisasi mesh 3D `.obj` menggunakan pendekatan **Octree (divide and conquer)** dengan GUI berbasis **Raylib**.

## Anggota Kelompok

- **Fazri Arrashyi Putra** - **13524127**
- **Dika Pramudya Nugraha** - **13524132**

## Deskripsi Singkat

Program membaca file OBJ segitiga, membangun octree hingga kedalaman tertentu, lalu mengekstraksi voxel (kubus) pada leaf node yang terisi.  
Hasil voxel diekspor kembali ke file OBJ (`*-voxelized.obj`) dan dapat dipreview di viewer 3D.

## Strategi Algoritma

Implementasi utama menggunakan:

- **Divide and Conquer (Octree)**  
  Ruang 3D dibagi rekursif menjadi 8 child box per node.
- **Pruning kandidat segitiga**  
  Tiap node hanya memproses kandidat face yang masih relevan.
- **Uji interseksi Triangle-AABB (SAT)**  
  Menentukan apakah segitiga benar-benar berpotongan dengan voxel box.
- **Ekstraksi voxel dari leaf occupied**  
  Leaf yang terisi diubah menjadi kubus (8 vertex, 12 face).

## Fitur Program

- GUI viewer interaktif berbasis Raylib.
- Pilih file OBJ dari daftar dan preview model asli.
- Atur depth octree lalu jalankan voxelisasi.
- Tombol **Cancel Voxel** untuk membatalkan proses voxelisasi yang sedang berjalan.
- Tampilkan model **Original** / **Voxel**.
- Statistik hasil:
  - Banyak voxel
  - Banyak vertex
  - Banyak faces
  - Statistik node octree terbentuk dan dipruning per depth
  - Waktu proses
  - Path output file

## Dependensi

- C++17
- CMake >= 3.14
- Compiler C++ (MSVC/MinGW/Clang)
- Git (untuk fetch raylib)

> `raylib` didownload otomatis melalui `FetchContent` pada CMake.

## Cara Build

Di root project:

```bash
cmake -S . -B build
cmake --build build
```

Output executable berada di folder `bin/`.

## Cara Menjalankan

Contoh run:

```bash
./bin/Tucil2_13524127_13524132
```

Atau dengan opsi awal:

```bash
./bin/Tucil2_13524127_13524132 test/teapot.obj --depth 5 --smooth
```

### Opsi CLI

- `--depth <n>`: depth awal octree
- `--smooth`: mode normal smooth saat viewer dibuka
- `[input.obj]`: path file OBJ opsional untuk preselect

## Kontrol Viewer

- **LMB / RMB drag**: orbit kamera
- **Mouse wheel**: zoom
- **Q / E**: zoom alternatif
- **Arrow keys**: orbit alternatif
- **R**: reset kamera
- **F11**: fullscreen
- **W**: toggle wireframe

## Struktur File Utama

- `src/main.cpp` - entry point dan parsing argumen
- `src/obj_io.*` - parsing / ekspor OBJ
- `src/octree.*` - algoritma octree dan interseksi segitiga-box
- `src/voxelizer.*` - pipeline voxelisasi
- `src/viewer.*` - GUI dan visualisasi Raylib