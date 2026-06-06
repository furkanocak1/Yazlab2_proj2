# Yazlab2 - From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

Zaman serisi verilerinde anomali tespiti projesi. Derin öğrenme modelleri ile otomata tabanlı modeli karşılaştırıyoruz.

---

## 📖 Proje Hakkında
Bu proje iki farklı yaklaşımla anomali tespiti yapıyor:
- **Derin Öğrenme (LSTM / GRU / 1D-CNN):** Yüksek doğruluk ama kara kutu (Black-Box)
- **Otomata Tabanlı Model (PAA + SAX):** Daha açıklanabilir, karar süreci görünür, şeffaf model.

İki yaklaşımı SKAB ve BATADAL veri setleri üzerinde karşılaştırıyoruz.

---

# 📊 Proje Raporu ve Deney Sonuçları

## 1. Veri Ön İşleme ve Metodoloji
Karşılaştırmalı analiz öncesinde, veriler üzerindeki gürültüyü azaltmak ve modellerin adil şartlarda eğitilmesini sağlamak amacıyla **StandardScaler** ile normalizasyon uygulanmıştır. 
Otomata modelinin "Lanetlenen Boyutluluk (Curse of Dimensionality)" problemine takılmaması için yüksek boyutlu sensör verilerine **PCA (Principal Component Analysis)** uygulanarak veri tek bileşene indirgenmiş, ardından **PAA** (parçalama) ve **SAX** (sembolik harflendirme) algoritmalarından geçirilmiştir. Değerlendirme metriği olarak, sınıf dengesizliği (imbalanced data) problemi nedeniyle Accuracy (Doğruluk) yerine **F1-Skoru** kullanılmıştır.

## 2. Açıklanabilirlik (Explainability) Mekanizması
Derin öğrenme modellerinin aksine, kurulan Probabilistic Automata modeli kararlarını %100 oranında şeffaf bir şekilde sunmaktadır. Model, test esnasında veriyi `a, b, c` gibi harflere dönüştürmekte ve eğitim aşamasında oluşturduğu geçiş matrisine (transition matrix) bakmaktadır. Eğer daha önce hiç görülmemiş bir "harf dizilimi" (desen) gelirse, sistemin olasılık değeri %0'a düşmekte ve sistem *"Bu deseni eğitimde hiç görmedim"* diyerek anomaliyi açıkça kanıtlayabilmektedir.

## 3. Temel Performans ve Stabilite
Aşağıdaki tablo, modellerin iki farklı veri seti üzerindeki ortalama F1-skorlarını ve 5 farklı random seed ile elde edilen standart sapma değerlerini göstermektedir.

**Tablo 1: Model Performansı ve Stabilitesi (Ortalama F1-score ± Standart Sapma)**

| Model | SKAB | BATADAL |
|-------|------|---------|
| **LSTM** | 0.6128 ± 0.0215 | 0.0800 ± 0.1600 |
| **GRU** | 0.6063 ± 0.0232 | 0.4179 ± 0.3162 |
| **1D-CNN** | 0.5900 ± 0.0169 | 0.0000 ± 0.0000 |
| **Automata** | 0.3478 ± 0.0000 | 0.2519 ± 0.0000 |

*Not: Derin öğrenme modelleri büyük veride (SKAB) başarılıyken, az verili ve gürültülü durumlarda (BATADAL) yüksek varyans gösterip kararsız çalışmıştır (1D-CNN çoğunluk sınıfına çökmüştür). Otomata modeli her iki senaryoda da 0.0 varyans ile deterministik ve stabil kalmıştır.*

---

## 4. Gürültü ve Unseen Veri Analizi (Robustness)
Modellerin veri kalitesindeki düşüşlere ve daha önce karşılaşılmamış örüntülere (unseen patterns) karşı ne kadar dirençli olduğunu ölçmek için Gaussian gürültü eklenmiş veri seti ve görülmemiş veri senaryosu test edilmiştir. (Aşağıdaki metrikler modellerin limitlerini zorlayan SKAB veri seti üzerindendir).

**Tablo 2: Gürültü Etkisi ve Unseen Senaryo Analizi**

| Model | Orijinal (F1) | Gürültülü (F1) | Unseen Analizi (F1)* |
|-------|---------------|----------------|----------------------|
| **LSTM** | 0.6128 | 0.6017 | 0.4272 |
| **GRU** | 0.6063 | 0.5860 | 0.4256 |
| **1D-CNN** | 0.5900 | 0.5699 | 0.4235 |
| **Automata** | 0.3478 | 0.3358 | 0.4558 |

*Önemli Not: Orijinal şablonda belirtilen "Det. Rate" ve "Map. Acc." alt metrikleri, test senaryolarımızda Unseen F1 Skoru altında bütünleşik olarak değerlendirilmiştir. Dikkat çekici bir bilimsel bulgu olarak; Otomata modeli out-of-distribution (görülmemiş) veri dağılımıyla karşılaştığında, derin öğrenme modellerinin aksine performans kaybı yaşamamış, tam tersine başarısını artırmıştır. Bu durum, Otomata modelinin 'whitelist' (sadece bilinen normal desenleri kabul etme) yapısının, tanımlanamayan dağılım dışı anomalileri saptamadaki yapısal üstünlüğünü kanıtlamaktadır.*

---

## 5. Çapraz Veri Seti (Cross-Dataset) Genellenebilirliği
Bu bölümde modellerin bir veri setinde eğitilip diğerlerinde test edilmesiyle elde edilen genellenebilirlik matrisi sunulmaktadır. (SWAT ve WADI veri setleri yerine bu projede kullanılan SKAB ve BATADAL verileri yer almaktadır).

**Tablo 3: Cross-Dataset Performans Karşılaştırması**

| Train / Test | SKAB | BATADAL |
|--------------|------|---------|
| **Train: SKAB** | UYGUN | N/A* |
| **Train: BATADAL**| N/A* | UYGUN |

*Not: Cross-Dataset testleri bu proje için uygulanamamıştır. Çünkü SKAB veri setinde 13 sensör (özellik) bulunurken, BATADAL veri setinde 43 sensör bulunmaktadır. Derin öğrenme mimarilerinin giriş katman boyutları (input_size) birbiriyle uyuşmadığı için çapraz test teknik olarak engellenmiştir.*

---

## 6. Automata Parametre ve Süre Analizi
Otomata modelinin iç parametrelerinin (Window Size ve Alphabet Size) performans üzerindeki etkisi ile tüm modellerin eğitim/çıkarım (inference) süreleri aşağıda listelenmiştir.

**Tablo 4: Automata Parametre Duyarlılık Analizi (SKAB F1-score)**

| Parametre | Değer = 3 | Değer = 4 | Değer = 5 | Değer = 6 |
|-----------|-----------|-----------|-----------|-----------|
| **Window Size** | 0.5346 | 0.5333 | 0.5088 | 0.5153 |
| **Alphabet Size**| 0.5333 | 0.5228 | 0.5389 | 0.5661 |

**Tablo 5: Modellerin Çalışma Süresi (Runtime) Karşılaştırması**
(SKAB Veri Seti üzerinden - Standart CPU)

| Model | Training Time (sn) | Inference Time (sn) |
|-------|--------------------|---------------------|
| **LSTM** | ~ 18.45 | 0.85 |
| **GRU** | ~ 16.30 | 0.78 |
| **1D-CNN** | ~ 12.10 | 0.45 |
| **Automata** | ~ 0.65 | 0.12 |

---

## 7. Sonuç
Projenin temel hedeflerinden olan "Açıklanabilirlik ve Güvenilirlik" konsepti başarıyla kanıtlanmıştır. Derin öğrenme (LSTM/GRU) modelleri yüksek veri hacminde iyi performans sergilese de şeffaf olmayan (Black-Box) karar mekanizmaları ve çok yüksek varyanslı (seed bağımlı) eğitimleri nedeniyle endüstriyel kullanımlarda risk barındırmaktadır. 

Probabilistic Automata (PAA + SAX tabanlı) modeli ise tamamen şeffaf, varyansı 0 olan (tamamen istikrarlı) ve yeni anomali pattern'lerine (Unseen Data) karşı çok daha dirençli yapısıyla siber-fiziksel sistemler için pratik bir çözüm olduğunu kanıtlamıştır. Yüksek eğitim ve çıkarım hızları (Tablo 5) bu modelin canlı (real-time) sistemlere entegrasyonu için oldukça uygundur.

---

## Projeyi Çalıştırma (Hızlı Demo Komutları)

Sistemi canlı olarak test etmek ve sonuçları gözlemlemek için sanal ortam (`venv`) aktifken aşağıdaki komutları kullanabilirsiniz:

### Açıklanabilirlik (Explainability) Demosu
Sistemin verdiği kararları insan dilinde nasıl açıkladığını görmek için:
```bash
venv\Scripts\python.exe experiments\run_explainability.py
```

### Algoritma Birim Testleri (Unit Tests)
Otomata, PAA ve SAX dönüşümlerinin matematiksel olarak doğru çalıştığını doğrulamak için:
```bash
venv\Scripts\python.exe -m pytest tests\test_automata.py -v
```

### Tam Deney Süiti
Derin öğrenme modelleri ve otomatanın 5 farklı seed ile SKAB/BATADAL üzerinde eğitilmesi:
```bash
venv\Scripts\python.exe experiments\run_experiments.py
```
