Proyek ini bertujuan untuk mengklasifikasikan serangan jaringan (khususnya serangan DDoS) menggunakan algoritma klasifikasi Decision Tree, dan mengevaluasi akurasinya terhadap dataset gabungan dari beberapa jenis serangan.
```phyton
import pandas as pd
import zipfile
```

pandas: Digunakan untuk manipulasi dan analisis data berbasis tabel (DataFrame).
zipfile: Digunakan untuk membuka dan mengekstrak file ZIP.

```phyton
with zipfile.ZipFile('drive-download-20250506T042545Z-1-001.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')
```
Membuka file ZIP yang berisi dataset.
Mengekstrak seluruh isinya ke direktori /content/.

```phyton
dataset = pd.read_csv("DDoS ICMP Flood.csv")
dataset2 = pd.read_csv("DDoS UDP Flood.csv")
dataset3 = pd.read_csv("DoS ICMP Flood.csv")
```
Membaca tiga file CSV yang berisi data serangan jaringan.
Masing-masing dataset merepresentasikan jenis serangan yang berbeda.

```phyton
hasilgabung = pd.concat([dataset, dataset2, dataset3], ignore_index=True)
```
Menggabungkan ketiga dataset menjadi satu.
ignore_index=True memastikan indeks ulang disusun dari 0 hingga akhir.

```phyton
hasilgabung.columns.values
```
Menampilkan daftar nama kolom dari dataset gabungan.

```phyton
x = hasilgabung.iloc[:, 7:76]
y = hasilgabung.iloc[:, 83:84]
```
x: Mengambil fitur dari kolom ke-7 sampai ke-75 (69 kolom).
y: Mengambil target/label dari kolom ke-83.

```phyton
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
Membagi data menjadi 80% data pelatihan dan 20% data pengujian.
random_state=42 untuk menjamin hasil pembagian tetap jika dijalankan ulang.

```phyton
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
alya = DecisionTreeClassifier(criterion='entropy', splitter='random')
alya.fit(x_train, y_train)
```
Membuat model decision tree dengan entropy sebagai kriteria pemisahan dan random sebagai metode pemilihan split.
Melatih model dengan data pelatihan.

```phyton
y_pred = alya.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
```
Melakukan prediksi terhadap data uji.
Mengukur akurasi prediksi menggunakan accuracy_score.

```phyton
import matplotlib.pyplot as plt
import numpy as np\
```
matplotlib.pyplot: Digunakan untuk membuat berbagai macam visualisasi seperti grafik, heatmap, dan sebagainya.
numpy: Digunakan untuk manipulasi array dan operasi numerik. Dalam bagian ini, dipakai untuk membentuk array label kelas.

```phyton
plt.figure(figsize=(10, 7))
tree.plot_tree(alya,
               feature_names=x.columns.values,
               class_names=np.array(['DDoS ICMP Flood.csv', 'DDoS UDP Flood.csv', 'DoS ICMP Flood.csv']),
               filled=True)
plt.show()
```
plt.figure(figsize=(10, 7)): Mengatur ukuran gambar.
tree.plot_tree(...): Membuat visualisasi dari model pohon keputusan (alya).
feature_names: Menyediakan nama fitur (dari kolom x) untuk ditampilkan di tiap node.
class_names: Menyediakan label kelas untuk klasifikasi. Di sini dituliskan langsung sebagai nama file CSV, yang sebaiknya digantikan dengan label aktual.
filled=True: Mewarnai node berdasarkan proporsi kelas.

```phyton
import seaborn as lol
from sklearn import metrics
label = np.array(['DDoS ICMP Flood.csv', 'DDoS UDP Flood.csv', 'DoS ICMP Flood.csv'])
```
seaborn diimport sebagai lol, yang tidak umum. Konvensi standar biasanya import seaborn as sns.
metrics: Modul dari sklearn yang menyediakan fungsi evaluasi seperti confusion matrix.
label: Array yang digunakan sebagai label untuk baris dan kolom heatmap.

```phyton
import matplotlib.pyplot as plt
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
lol.heatmap(conf_matrix,
            annot=True,
            xticklabels=label,
            yticklabels=label)
plt.xlabel('Prediksi')
plt.ylabel('Fakta')
plt.show()
```
confusion_matrix(y_test, y_pred): Menghasilkan matriks kebingungan yang menunjukkan jumlah prediksi benar dan salah dari model.
lol.heatmap(...): Membuat visualisasi matriks kebingungan dengan nilai anotasi.
xticklabels dan yticklabels: Mengatur label sumbu X dan Y berdasarkan array label.
plt.xlabel dan plt.ylabel: Memberi label pada sumbu.
plt.show(): Menampilkan hasil plot.















