Berikut penjelasan lengkap per bagian dari kode Python Anda yang digunakan untuk **sentiment analysis** menggunakan **SVM (Support Vector Machine)**:

---

### **Bagian 1: Load Data**

```python
import pandas as pd

# train Data
trainData = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/train.csv")
# test Data
testData = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/test.csv")
```

* **Tujuan**: Mengimpor dataset pelatihan (`train.csv`) dan pengujian (`test.csv`) dari GitHub.
* **trainData** dan **testData** akan berisi dua kolom:

  * `Content`: Teks ulasan/konten.
  * `Label`: Label sentimen (`pos` atau `neg`).

---

###  **Bagian 2: Vektorisasi Teks dengan TF-IDF**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
train_vectors = vectorizer.fit_transform(trainData['Content'])
test_vectors = vectorizer.transform(testData['Content'])
```

* **TfidfVectorizer** mengubah teks menjadi representasi numerik (vektor).
* `min_df=5`: Abaikan kata yang muncul di kurang dari 5 dokumen.
* `max_df=0.8`: Abaikan kata yang muncul di lebih dari 80% dokumen (umum).
* `sublinear_tf=True`: Gunakan log(1 + tf) daripada tf mentah.
* `use_idf=True`: Gunakan IDF (Inverse Document Frequency) untuk bobot kata.
* `fit_transform()`: Melatih vectorizer di data training dan mengubahnya.
* `transform()`: Mengubah data test menggunakan vectorizer yang sama.

---

### **Bagian 3: Training dan Prediksi dengan SVM**

```python
import time
from sklearn import svm
from sklearn.metrics import classification_report

classifier_linear = svm.SVC(kernel='linear')

t0 = time.time()
classifier_linear.fit(train_vectors, trainData['Label'])  # Melatih model
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)  # Prediksi data test
t2 = time.time()
```

* `SVC(kernel='linear')`: Support Vector Classifier dengan kernel linear, cocok untuk data teks.
* `fit()`: Melatih model dengan data training.
* `predict()`: Memprediksi label dari data test.

Waktu dievaluasi:

```python
time_linear_train = t1 - t0
time_linear_predict = t2 - t1
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
```

---

### **Bagian 4: Evaluasi Hasil**

```python
report = classification_report(testData['Label'], prediction_linear, output_dict=True)
print('positive: ', report['pos'])
print('negative: ', report['neg'])
```

* `classification_report`: Menghitung precision, recall, f1-score, dan support untuk setiap kelas.
* Dicetak khusus untuk kelas `'pos'` dan `'neg'`.

---

### **Bagian 5: Menyimpan Model & Vectorizer**

```python
import pickle
pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))
pickle.dump(classifier_linear, open('classifier.sav', 'wb'))
```

* `pickle`: Digunakan untuk menyimpan (serialize) objek Python.
* File `vectorizer.sav`: Menyimpan vektorizer TF-IDF.
* File `classifier.sav`: Menyimpan model klasifikasi SVM.

---

### **Bagian 6: Load Model & Prediksi Manual**

```python
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
classifier = pickle.load(open('classifier.sav', 'rb'))

test_sentences = [
    "I really love this product! It's amazing.",
    "This is the worst service I’ve ever experienced."
]
test_vector = vectorizer.transform(test_sentences)
predictions = classifier.predict(test_vector)
```

* **Meload kembali model dan vectorizer** dari file.
* **Uji coba dua kalimat secara manual** untuk melihat hasil sentimennya.

Cetak hasil prediksi:

```python
for sentence, label in zip(test_sentences, predictions):
    print(f"Kalimat: {sentence} => Sentimen: {label}")
```

Output:

```
Kalimat: I really love this product! It's amazing. => Sentimen: pos
Kalimat: This is the worst service I’ve ever experienced. => Sentimen: neg
```

gin saya bantu membuat versi web-based dari sistem ini menggunakan Flask atau Streamlit?
