# Laporan Proyek Machine Learning - Felicia Salim

## Domain Proyek

Diabetes mellitus merupakan salah satu penyakit kronis yang umum di dunia dan berpotensi untuk menyebabkan penyakit seperti sirosis hati, gagal ginjal, kebutaan, bahkan amputasi [1]. Menurut  World Health Organization (WHO), jumlah penderita diabetes terus meningkat drastis dari 200 juta hingga 800 juta dalam waktu 30 tahun, dengan angka kematian yang mencapai 2 juta jiwa pada tahun 2021 [2]. Banyak kasus diabetes baru terdiagnosis ketika penyakit sudah mencapai tahap komplikasi dam ini disebabkan oleh gejala awal yang sering kali tidak disadari. Oleh karena itu, penting untuk mengidentifikasi diabetes secara dini agar dapat dilakukan penanganan yang lebih cepat dan efektif.

Penelitian oleh Antar et al. (2023) menunjukkan bahwa sejumlah faktor seperti kadar glukosa darah, indeks massa tubuh (BMI), tekanan darah, usia, jumlah kehamilan, serta riwayat keluarga memiliki kaitan kuat dengan kemungkinan seseorang terkena diabetes [1]. Namun, masih belum sepenuhnya jelas faktor-faktor yang paling berkontribusi terhadap risiko tersebut. Masalah ini harus diselesaikan agar masyarakat mendapatkan pemahaman yang lebih akurat mengenai faktor risiko utama, sehingga mereka dapat melakukan strategi pencegahan diabetes, terutama pada orang yang memiliki keluarga dengan riwayat penyakit diabetes. Untuk itu, dengan adanya perkembangan teknologi pembelajaran mesin, kita dapat menganalisis faktor yang paling berpengaruh terhadap penyakit diabetes dan membangun model prediktif yang mampu mendeteksi penyakit diabetes dengan lebih akurat.

**Refrences**

[1] S. A. Antar et al., “Diabetes mellitus: Classification, mediators, and complications; a gate to identify potential targets for the development of new effective treatments,” Biomedicine &amp; Pharmacotherapy, vol. 168, p. 115734, Oct. 2023. doi:10.1016/j.biopha.2023.115734 

[2] World Health Organization, “Diabetes,” World Health Organization, https://www.who.int/news-room/fact-sheets/detail/diabetes (Accessed May 15, 2025). 

## Business Understanding

### Problem Statements
- Meskipun terdapat banyak faktor yang diketahui terkait dengan diabetes seperti kadar glukosa darah, tekanan darah, BMI, usia, dan riwayat kehamilan, masih belum sepenuhnya jelas faktor mana yang paling berkontribusi terhadap risiko terkena diabetes. Hal ini dapat mengakibatkan kurangnya strategi pencegahan yang fokus pada indikator paling kritis saat penanganan pasien.
- Terdapat berbagai algoritma machine learning seperti Logistic Regression ataupun Random Forest yang bisa digunakan untuk klasifikasi diabetes. Namun, performa setiap model pasti menghasilkan hasil yang berbeda tergantung pada karakteristik dan fitur dari data, sehingga masih belum diketahui model mana yang paling efektif dalam mengklasifikasikan penyakit diabetes.


### Goals
- Mengidentifikasi faktor yang paling berpengaruh terhadap risiko diabetes melalui analisis fitur pada data medis, sehingga dapat membantu dalam perumusan strategi pencegahan yang lebih fokus dan efektif.
- Mengevaluasi dan membandingkan performa berbagai algoritma klasifikasi untuk menentukan model machine learning yang paling akurat dalam mendeteksi penyakit diabetes berdasarkan data yang tersedia.


### Solution Statements
- Mengimplementasikan dan membandingkan beberapa algoritma klasifikasi, seperti Logistic Regression, Random Forest, dan K-Nearest Neighbor, untuk memprediksi apakah seorang individu menderita diabetes berdasarkan fitur medis seperti kadar glukosa, tekanan darah, BMI, usia, dan lainnya.
- Melakukan evaluasi performa masing-masing model menggunakan metrik evaluasi seperti akurasi, presisi, recall, F1-score, dan Confusion Matrix guna menentukan model yang paling efektif dalam melakukan klasifikasi penyakit diabetes.
- Melakukan analisis feature importance pada model seperti Random Forest untuk mengidentifikasi fitur-fitur yang paling berpengaruh terhadap prediksi diabetes.


## Data Understanding
Dataset yang digunakan dalam proyek ini adalah [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set) yang diambil dari Kaggle. Dataset tersebut memiliki 768 baris dan 9 kolom.

### Variabel-variabel pada Dataset Diabetes adalah sebagai berikut:
- Pregnancies: Jumlah kehamilan
- Glucose: Konsentrasi glukosa plasma dua jam setelah tes toleransi glukosa oral
- BloodPressure: Tekanan darah diastolik (mm Hg)
- SkinThickness: Ketebalan kulit triseps (mm)
- Insulin: Kadar insulin (mu U/ml)
- BMI: Indeks massa tubuh
- DiabetesPedigreeFunction: Fungsi silsilah diabetes (faktor keturunan dan riwayat diabetes dalam keluarga)
- Age: Usia (tahun)
- Outcome: Target variabel kelas (0: Tidak diabetes atau 1: Diabetes)

### Exploratory Data Analysis
Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan. Beberapa tahapan EDA yang akan dilakukan adalah pemahaman terhadap struktur data, analisis setiap fitur, pengecekan data duplikat dan nilai null, analisis univariate variabel target, serta analisis distribusi, outlier, dan korelasi data. 

```
diabetes_df.info()
```
Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64  
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
```
Terdapat 768 row data dan 9 fitur pada dataset yang digunakan. Ada 7 fitur dengan tipe data integer dan 2 fitur dengan tipe data float. Dataset ini berisi data numerikal semuanya.



### Eksplor Parameter Statistik
```
diabetes_df.describe()
```
Output:

|           | Pregnancies | Glucose   | BloodPressure | SkinThickness | Insulin   | BMI      | DiabetesPedigreeFunction | Age       | Outcome   |
|-----------|-------------|-----------|----------------|----------------|-----------|----------|---------------------------|-----------|-----------|
| count     | 768.000000  | 768.000000| 768.000000     | 768.000000     | 768.000000| 768.000000| 768.000000                | 768.000000| 768.000000|
| mean      | 3.845052    | 120.894531| 69.105469      | 20.536458      | 79.799479 | 31.992578| 0.471876                  | 33.240885 | 0.348958  |
| std       | 3.369578    | 31.972618 | 19.355807      | 15.952218      |115.244002 | 7.884160 | 0.331329                  | 11.760232 | 0.476951  |
| min       | 0.000000    | 0.000000  | 0.000000       | 0.000000       | 0.000000  | 0.000000 | 0.078000                  | 21.000000 | 0.000000  |
| 25%       | 1.000000    | 99.000000 | 62.000000      | 0.000000       | 0.000000  | 27.300000| 0.243750                  | 24.000000 | 0.000000  |
| 50%       | 3.000000    | 117.000000| 72.000000      | 23.000000      | 30.500000 | 32.000000| 0.372500                  | 29.000000 | 0.000000  |
| 75%       | 6.000000    | 140.250000| 80.000000      | 32.000000      |127.250000 | 36.600000| 0.626250                  | 41.000000 | 1.000000  |
| max       |17.000000    |199.000000 |122.000000      | 99.000000      |846.000000 | 67.100000| 2.420000                  | 81.000000 | 1.000000  |

- Terdapat nilai 0 di beberapa kolom penting seperti Glucose, BloodPressure, SkinThickness, Insulin, padahal kolom tersebut tidak mungkin bernilai 0. Ini berarti ada data yang mungkin perlu ditangani nantinya.
- Rata-rata 'Glucose' mencapai sekitar 120 dengan variasi cukup besar. Ini dapat menandakan bahwa tingginya kadar gula sangat berpengaruh pada 'Outcome'.
- Insulin memiliki nilai max yang sangat besar yaitu 846. Ada kemungkinan disini terdapat outlier.
- BMI rata-rata mencapai 32 dan ini menandakan bahwa terdapat kelebihan berat badan atau obesitas.
- Usia rata-rata 33 tahun dari rentang umur 21 sampai 81 tahun. Jadi faktor usia cukup bervariasi.

### Univariate Analysis Terhadap Target 'Outcome'
```
diabetes_df.groupby(by='Outcome').agg({
    "Pregnancies": ["mean","std"],
    "Glucose": ["mean","std"],
    "BloodPressure": ["mean","std"],
    "SkinThickness": ["mean","std"],
    "Insulin": ["mean","std"],
    "BMI": ["mean","std"],
    "DiabetesPedigreeFunction": ["mean","std"],
    "Age": ["mean","std"]
})
```
Output:
![image](https://github.com/user-attachments/assets/5f8866bf-f4c3-4b40-b4fc-ba10e38d0612)
Pasien diabetes umumnya memiliki lebih banyak kehamilan, kadar glukosa, insulin, dan BMI yang lebih tinggi, serta usia dan riwayat keluarga yang lebih kuat terkait diabetes. Sementara itu, tekanan darah memiliki pengaruh yang kecil terhadap risiko diabetes.

### Pengecekan Data Nil, Data Duplikat, dan Class Imbalance
```
#Cek null values
diabetes_df.isna().sum()
```
Output:
```
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
dtype: int64
```
Tidak ada data nil.

```
#Cek Data Duplikat
print("Terdapat ",diabetes_df.duplicated().sum(), "data duplikat.")
```
Output:
```
Terdapat 0 data duplikat.
```
Tidak ada data duplikat.

```
#Cek apakah ada class imbalance
diabetes_df["Outcome"].value_counts()
```
Output:
```
Outcome
0    500
1    268
Name: count, dtype: int64
```
Terdapat class imbalance dimana kelas tidak diabetes memiliki 500 data, sedangkan kelas diabetes hanya memiliki 268 data. Ini akan ditangani nanti pada tahap preprocessing.

### Pendeteksian Outlier Dengan Boxplot
```
plt.figure(figsize=(15,10))
for i, col in enumerate (diabetes_df.columns, 1):
    plt.subplot(3,4,i)
    sns.boxplot(x = diabetes_df[col])
    plt.title(col)

plt.tight_layout()
plt.show()
```
Output:

![image](https://github.com/user-attachments/assets/7a8694e9-4b3a-4b4d-a57a-1945dbe9bf85)

Terdapat cukup banyak outlier pada beberapa kolom seperti Insulin, DiabetesPedigreeFunction, SkinThickness, BMI, Age, dan BloodPressure. Outlier ini menandakan bahwa terdapat beberapa pasien dengan nilai yang jauh lebih tinggi dari mayoritas, seperti kadar insulin yang sangat tinggi, riwayat keturunan diabetes yang kuat, atau nilai tekanan darah dan BMI yang ekstrim. Kondisi ini menunjukkan adanya data yang perlu ditangani saat preprocessing.

### Visualisasi Distribusi Dengan Pairplot
```
sns.pairplot(diabetes_df, hue='Outcome', palette='coolwarm')
plt.show()
```
Output:

![image](https://github.com/user-attachments/assets/bdb3195a-dc50-47cd-b5a1-1074fef79914)

Distribusi antara pasien diabetes dan non-diabetes menunjukkan perbedaan yang cukup jelas pada fitur Glucose, BMI, dan Age, di mana pasien diabetes cenderung memiliki nilai yang lebih tinggi pada ketiga fitur tersebut. Selain itu, fitur Pregnancies juga menunjukkan kecenderungan lebih tinggi pada pasien diabetes. Perbedaan distribusi ini mengindikasikan bahwa faktor-faktor seperti kadar glukosa darah, indeks massa tubuh, usia, dan jumlah kehamilan berpotensi menjadi indikator penting dalam mendeteksi diabetes.

### Visualisasi Skewness pada Setiap Fitur
```
cols = diabetes_df.columns
plt.figure(figsize=(40, 25))

for i, cols in enumerate (cols, 1):
    plt.subplot(3,5,i)
    sns.histplot(data=diabetes_df[cols], kde=True)
    plt.title(cols)

plt.show()
```
Output:

![image](https://github.com/user-attachments/assets/f5127f2b-11ac-4ed2-a503-7a9fd56d0699)

```
diabetes_df.skew()
```
Output:
```
Pregnancies                 0.901674
Glucose                     0.173754
BloodPressure              -1.843608
SkinThickness               0.109372
Insulin                     2.272251
BMI                        -0.428982
DiabetesPedigreeFunction    1.919911
Age                         1.129597
Outcome                     0.635017
dtype: float64
```
Sebagian besar fitur memiliki distribusi miring ke kanan (positif), seperti Pregnancies, Glucose, SkinThickness, Insulin, DiabetesPedigreeFunction, dan Age, yang menunjukkan banyak data berada di nilai rendah dengan beberapa nilai ekstrim tinggi. BloodPressure dan BMI justru cenderung miring ke kiri (negatif) dan memiliki nilai abnormal (BMI = 0). Glucose dan SkinThickness hampir simetris, sementara Insulin dan DiabetesPedigreeFunction sangat skewed, yang bisa memengaruhi performa model. Kelas Outcome juga tidak seimbang, dengan mayoritas data berasal dari kelas non-diabetes.

### Visualisasi Korelasi antar Fitur
```
plt.figure(figsize=(8, 8))
sns.heatmap(diabetes_df.corr(method="spearman"), annot=True, fmt='.2f', cmap='Blues',annot_kws={"size": 10})
plt.title('Feature Correlation', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
```
Output:

![image](https://github.com/user-attachments/assets/380c55f0-9570-4926-9924-9b87103ae83e)

- Glucose memiliki korelasi tinggi dengan Outcome sebesar 0.48 yang menunjukkan hubungan kuat dengan kelas diabetes.
- Age dan BMI memiliki korelasi sedang dengan Outcome (sekitar 0.31) yang berarti keduanya cukup berpengaruh terhadap kelas diabetes (Outcome).
- Pregnancies, DiabetesPedigreeFunction, BloodPressure, SkinThickness, dan Insulin memiliki korelasi rendah dengan Outcome. Ini menunjukkan pengaruh yang lebih kecil terhadap kelas diabetes.


## Data Preparation
Tahap ini adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning. Data mentah sering kali mengandung nilai kosong, duplikat, atau rentang nilai yang tidak konsisten, yang dapat memengaruhi kinerja model. Oleh karena itu, proses ini bertujuan untuk membersihkan dan mempersiapkan data. Tahap preparasi yang adalah dilakukan adalah penanganan terhadap data abnormal, penanganan data outlier, reset index, split data, standarisasi data dan oversampling data. Disini tidak dilakukan penghapusan data null dan duplikat karena dari tahap 'Data Understanding' tidak ditemukan data null maupun duplikat.

### - Penanganan Terhadap Data Abnormal 
Dari Data Understanding, terdapat beberapa fitur data yang bernilai abnormal. Data abnormal 'BMI = 0' harus di drop karena index massa tubuh tidak mungkin bernilai 0 dan tidak masalah apabila di drop (dikarenakan jumlahnya tidak terlalu signifikan). Sedangkan untuk kolom 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin' yang bernilai 0 (yang abnormal) harus diganti nilainya menjadi median dikarenakan jumlahnya terlalu banyak. 

### - Penanganan Outlier
Outlier akan diganti menggunakan IQR agar mengurangi overfitting model terhadap data ekstrim dan meningkatkan akurasi pada model yang akan dibangun nantinya. Hal ini dilakukan karena outlier sangat mempengaruhi kualitas data yang digunakan.

### - Split Data
Split data menjadi train dan test set dengan rasio 80:20 menggunakan library sklearn train_test_split agar model dapat dilatih dan dievaluasi dengan baik.

### - Standarisasi Data
Jika tidak melakukan standarisasi, fitur dengan nilai yang jauh lebih besar bisa mendominasi proses perhitungan dan menyebabkan model memberikan bobot yang tidak adil pada fitur tertentu, contohnya fitur insulin yang dapat mencapai nilai 100 dibandingkan dengan fitur pregnancies yang bernilai puluhan. Dengan standarisasi, semua fitur pada data disamakan skala dan distribusinya sehingga model dapat belajar pola secara lebih seimbang dan akurat. Standarisasi akan dilakukan dengan StandardScaler.

### - Oversampling
Oversampling harus dilakukan karena terdapat class imbalance pada target variable (Outcome) yang cukup signifikan. Oversampling akan dilakukan dengan SMOTE agar memastikan bahwa kelas yang mendominasi tidak terlalu dominan pada model. 

## Modeling
Terdapat 3 model klasifikasi yang digunakan untuk klasifikasi penyakit diabetes, yaitu:
### - Logistic Regression
Logistic Regression merupakan model yang cocok untuk masalah klasifikasi seperti binary classification (0 = tidak, 1 = iya). Model ini bekerja dengan memperkirakan probabilitas suatu kelas.
- Parameter: "**max_iter=100**" berarti jumlah maksimum iterasi untuk mencari konvergensi pas training. 
- Kelebihan:
  - Cepat dan simpel untuk baseline model klasifikasi.
  - Hasilnya mudah diinterpretasi karena bisa lihat koefisien tiap fitur.
  - Cukup bagus di dataset yang nggak terlalu besar.
- Kekurangan:
  - Kurang cocok untuk data yang punya pola non-linear.
  - Bisa sensitif terhadap fitur yang punya skala berbeda.

### - Random Forest
Random Forest adalah algoritma berbasis ensemble yang terdiri dari banyak decision tree. Saat memprediksi, tiap pohon memberi “vote”, dan hasil akhirnya ditentukan dari mayoritas voting tersebut. 
- Parameter: "**n_estimators=100**" berarti jumlah pohon dalam hutan, "**max_depth=10**" yang berarti kedalaman maksimal pada setiap pohon, "**random_state=42**" yang berarti nilai acuan untuk pengacakan data supaya tiap proses training itu konsisten.
- Kelebihan:
  - Cocok buat data numerik dan kategorikal.
  - Tidak mudah overfitting dan dapat menangani data non-linier.
  - Bisa menemukan fitur mana yang paling penting (feature importance).
- Kekurangan:
  - Lebih lambat dibanding model sederhana (karena banyak pohon).
  - Kurang interpretasi dibanding model seperti Logistic Regression.
 
### - K Nearest Neighbor
KNN merupakan model yang bekerja dengan mencari beberapa data yang paling dekat (dar jarak) ke data baru yang mau diprediksi. KNN cocok kalau data antar kelas punya jarak yang cukup jelas dan sudah diskalakan. 
- Parameter: "**n_neighbors=10**" yang berarti model akan mempertimbangkan 10 tetangga terdekat saat menentukan kelas.
- Kelebihan:
  - Simpel dan mudah dipahami.
  - Cocok kalau data antar kelas cukup terpisah.
- Kekurangan:
  - Prediksi bisa lambat jika terdapat banyak data.
  - Performanya bisa jelek kalau datanya belum diskala atau banyak outlier/noise.

#### Ketiga model tersebut dilatih dengan data train dan test yang sudah distandarisasi dan oversampling. Model kemudian akan dievaluasi untuk mencari yang terbaik di antara ketiga model tersebut.

## Evaluation
Disini, model akan dievaluasi dengan 4 metrik berbeda yaitu:
- Accuracy: metrik yang mengukur seberapa sering model memberikan prediksi yang benar, baik positif maupun negatif, dari seluruh data. Ini menunjukkan performa keseluruhan model secara umum. Rumus:
  
    ![image](https://github.com/user-attachments/assets/8c28351f-5474-43bf-ae1a-6ac5e5cb236d)


- Precision: menunjukkan hasil yang benar-benar positif dari semua kasus yang diprediksi positif oleh model. Metrik ini penting untuk mengukur seberapa akurat prediksi positif model, terutama jika false positive harus diminimalkan. Precision tinggi artinya sedikit prediksi positif yang salah. Rumus:

    ![image](https://github.com/user-attachments/assets/1d96b072-6934-4271-b852-3035757e45cc)

- Recall: mengukur banyaknya hasil yang berhasil dideteksi oleh model dari semua data yang sebenarnya positif. Recall penting ketika kita tidak ingin melewatkan kasus positif yang sebenarnya. Recall tinggi berarti model jarang gagal mendeteksi kasus positif. Rumus:
  
    ![image](https://github.com/user-attachments/assets/16602179-aacc-4ee7-b850-d789caf23b84)

- F1 Score: rata-rata dari precision dan recall, yang memberikan nilai tunggal untuk menilai keseimbangan antara keduanya. F1 Score berguna saat kita ingin memastikan model tidak hanya akurat dalam menemukan positif, tapi juga meminimalkan kesalahan prediksi. Rumus:

    ![image](https://github.com/user-attachments/assets/3555860c-43fe-4919-9576-06cdf7e651a4)


Dimana:

      - TP = True Positive (benar positif)
      - TN = True Negative (benar negatif)
      - FP = False Positive (salah positif)
      - FN = False Negative (salah negatif)

### Hasil Evaluasi
```
Logistic Regression:
  Accuracy:  0.7951
  Precision: 0.6275
  Recall:    0.8421
  F1 Score:  0.7191

Random Forest:
  Accuracy:  0.8443
  Precision: 0.6939
  Recall:    0.8947
  F1 Score:  0.7816

KNN:
  Accuracy:  0.7377
  Precision: 0.5500
  Recall:    0.8684
  F1 Score:  0.6735
```
Confusion Matrix Random Forest (karena memiliki hasil akurasi tertinggi) untuk melihat hasil prediksi salah & benar:

![image](https://github.com/user-attachments/assets/c93c3816-786b-422a-acfa-c08862fd6186)

Feature Importance untuk melihat fitur paling signifikan pada klasifikasi penyakit diabetes:

![image](https://github.com/user-attachments/assets/6aefc9f4-00b7-4bec-b57d-91cff1b95797)


Model Random Forest menunjukkan performa terbaik dibandingkan model lain, dengan akurasi sebesar 84.4%, presisi 69.3%, recall 89.4%, dan F1 Score tertinggi (78%). Hal ini menunjukkan bahwa model ini tidak hanya akurat secara keseluruhan, tetapi juga sangat baik dalam mengenali pasien yang benar-benar menderita diabetes (dari recall yang tinggi). Dari confusion matrix Random Forest juga dapat dilihat bahwa ada 103 label terprediksi dengan benar dan 19 label terprediksi dengan salah. Sedangkan untuk fitur yang paling signifikan pada penyakit diabetes adalah Glucose (kadar gula darah), BMI (indeks massa tubuh), dan Age (usia).




