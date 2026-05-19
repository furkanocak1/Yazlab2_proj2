# Yazlab2 - From Black-Box to Explainability

Zaman serisi verilerinde anomali tespiti projesi.  
Derin öğrenme modelleri ile otomata tabanlı modeli karşılaştırıyoruz.

---

## Proje Hakkında

Bu proje iki farklı yaklaşımla anomali tespiti yapıyor:

- **Derin Öğrenme (LSTM / GRU / 1D-CNN):** Yüksek doğruluk ama kara kutu
- **Otomata Tabanlı Model (PAA + SAX):** Daha açıklanabilir, karar süreci görünür

İki yaklaşımı iki farklı veri seti üzerinde karşılaştırıyoruz.

---

## Kullanılan Veri Setleri

### SKAB
Laboratuvar ortamındaki bir boru/vana sisteminin sensör verileri.  
Kaynak: https://github.com/waico/SKAB  
Kullanılan klasörler: `valve1`, `valve2`  
Hedef değişken: `anomaly` (0 = normal, 1 = anomali)

### BATADAL
Bir şehir su dağıtım sistemine yapılmış simüle siber saldırı verileri.  
Kaynak: http://www.batadal.net/data.html  
Kullanılan dosya: `BATADAL_dataset04.csv` (Training Dataset 2)  
Hedef değişken: `ATT_FLAG` (0 = normal, 1 = saldırı)

---

## Kurulum

### Gereksinimler
- Python 3.11+
- Git

### Adımlar

**1. Repoyu klonla:**
```bash
git clone <repo-linki>
cd Yazlab2_proj2
```

**2. Sanal ortam oluştur ve aktif et:**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Kütüphaneleri yükle:**
```bash
pip install -r requirements.txt
```

**4. Veri setlerini indir ve doğru klasörlere koy:**

SKAB için `valve1` ve `valve2` klasörlerini şuraya koy:
```
data/SKAB/valve1/
data/SKAB/valve2/
```

BATADAL için `BATADAL_dataset04.csv` dosyasını şuraya koy:
```
data/BATADAL/
```

---

## Klasör Yapısı

```
Yazlab2_proj2/
├── config.py               # Tüm parametreler burada
├── requirements.txt        # Gerekli kütüphaneler
├── data/                   # Veri setleri (GitHub'a gitmez)
│   ├── SKAB/
│   │   ├── valve1/
│   │   └── valve2/
│   └── BATADAL/
├── preprocessing/          # Veri yükleme ve temizleme
│   ├── data_loader.py      # Veri okuma ve birleştirme
│   └── preprocessor.py     # Normalizasyon, PCA
├── models/                 # Model kodları
│   ├── deep_learning/      # LSTM, GRU, CNN (arkadaşın kısmı)
│   └── automata/           # Otomata modeli (benim kısmım)
├── explainability/         # Açıklanabilirlik modülü
├── experiments/            # Deney sonuçları
└── tests/                  # Birim testler
```

---

## Şu Ana Kadar Yapılanlar

### ✅ Tamamlanan
- [x] Proje iskelet yapısı oluşturuldu
- [x] Merkezi konfigürasyon dosyası (`config.py`) yazıldı
- [x] Veri yükleme modülü (`data_loader.py`) yazıldı
  - SKAB: valve1 + valve2 birleştirildi → 22.472 satır
  - BATADAL: tek dosya → 4.177 satır
- [x] Veri ön işleme modülü (`preprocessor.py`) yazıldı
  - BATADAL etiket düzeltmesi (-999 → 0)
  - Eksik veri kontrolü
  - Normalizasyon (StandardScaler)
  - PCA (tek bileşene indirgeme)

### 🔄 Devam Eden
- [ ] Veri bölme (SKAB için GroupKFold, BATADAL için %60-20-20)
- [ ] Otomata modeli (PAA + SAX + sliding window)
- [ ] Derin öğrenme modelleri (LSTM, GRU veya CNN)
- [ ] Açıklanabilirlik modülü
- [ ] Deneyler (normal, gürültülü, unseen veri)
- [ ] Grafikler ve raporlama

---

## İş Bölümü

| Kişi | Sorumluluk |
|---|---|
| **Furkan** | Veri ön işleme (ortak), Otomata modeli, Açıklanabilirlik modülü |
| **Arkadaş** | Derin öğrenme modelleri (LSTM, GRU/CNN), Deneyler, Grafikler |

---

## Önemli Notlar

- `venv/` klasörü GitHub'a gitmez, herkes kendi bilgisayarında oluşturur
- Veri setleri GitHub'a gitmez, yukarıdaki kurulum adımlarını takip et
- Tüm parametreler `config.py` üzerinden yönetilir, hard-code değer kullanma
- Normalizasyon ve PCA sadece train verisi üzerinde fit edilmeli
