# =============================================================
# config.py - Merkezi Ayar Dosyası
# Tüm parametreler buradan yönetilir.
# Bir şeyi değiştirmek istersen sadece buraya gel.
# =============================================================

import os

# -------------------------------------------------------------
# GENEL AYARLAR
# -------------------------------------------------------------

# Projenin ana klasörü (config.py neredeyse orası)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Veri klasörü
DATA_DIR = os.path.join(BASE_DIR, "data")

# Deney sonuçlarının kaydedileceği klasör
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

# -------------------------------------------------------------
# TEKRARLANABILIRLIK
# Aynı sonuçları elde etmek için sabit seed değerleri kullanıyoruz.
# Her deney bu 5 seed ile tekrar çalıştırılacak.
# -------------------------------------------------------------

RANDOM_SEEDS = [42, 123, 2026, 7, 999]

# -------------------------------------------------------------
# VERİ ÖN İŞLEME
# -------------------------------------------------------------

# Test ve doğrulama oranları (BATADAL için)
# %60 eğitim, %20 doğrulama, %20 test
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20

# -------------------------------------------------------------
# OTOMATA MODELİ PARAMETRELERİ
# PAA: veriyi parçalara böler
# SAX: her parçayı bir harfe çevirir
# -------------------------------------------------------------

# Sabit parametreler (karşılaştırma deneyleri için)
WINDOW_SIZE   = 4   # kaç veri noktası bir arada değerlendiriliyor
ALPHABET_SIZE = 3   # kaç farklı harf kullanılıyor (a, b, c)

# Parametre varyasyonu (duyarlılık analizi için)
WINDOW_SIZES   = [3, 4, 5, 6]
ALPHABET_SIZES = [3, 4, 5, 6]

# -------------------------------------------------------------
# DERİN ÖĞRENME MODELİ PARAMETRELERİ
# -------------------------------------------------------------

EPOCH_LIMIT  = 50   # maksimum eğitim turu
BATCH_SIZE   = 32   # her adımda kaç örnek işleniyor
PATIENCE     = 5    # doğrulama kaybı iyileşmezse kaç tur bekle (early stopping)

# -------------------------------------------------------------
# GÜRÜLTÜ (GAUSSIAN NOISE) PARAMETRELERİ
# Gürültü deneyi için veriye eklenen gürültü miktarı
# -------------------------------------------------------------

NOISE_MEAN  = 0.0   # gürültünün ortalaması
NOISE_STD   = 0.1   # gürültünün standart sapması

# -------------------------------------------------------------
# VERİ SETİ AYARLARI
# -------------------------------------------------------------

SKAB_DIR   = os.path.join(DATA_DIR, "SKAB")    # SKAB veri seti klasörü
BATADAL_DIR = os.path.join(DATA_DIR, "BATADAL") # BATADAL veri seti klasörü

# SKAB: hangi klasörler kullanılacak
SKAB_FOLDERS = ["valve1", "valve2"]

# SKAB: hedef değişken (tahmin etmeye çalıştığımız sütun)
SKAB_TARGET = "anomaly"

# BATADAL: eğitim için kullanılacak dosya
BATADAL_TRAIN_FILE = "BATADAL_dataset04.csv"

# BATADAL: hedef değişken
BATADAL_TARGET = "ATT_FLAG"
