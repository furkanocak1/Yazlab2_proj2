# =============================================================
# preprocessing/preprocessor.py - Veri Ön İşleme Modülü
# Temizleme, normalizasyon ve PCA işlemlerini yapar.
# =============================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.data_loader import (
    load_skab, load_batadal,
    get_skab_features, get_batadal_features
)


# =============================================================
# BATADAL ETİKET DÜZELTMESİ
# =============================================================

def fix_batadal_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    BATADAL'da normal satırlar -999 ile işaretlenmiş.
    Bunu 0'a çeviriyoruz:
      -999 → 0 (normal)
         1 → 1 (anomali)
    """
    df = df.copy()
    df[config.BATADAL_TARGET] = df[config.BATADAL_TARGET].replace(-999, 0)
    print(f"BATADAL etiket dağılımı:\n{df[config.BATADAL_TARGET].value_counts()}\n")
    return df


# =============================================================
# EKSİK VERİ TEMİZLEME
# =============================================================

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eksik verileri temizler.
    Sayısal sütunlarda ortalama ile doldurur.
    """
    eksik = df.isnull().sum().sum()
    if eksik > 0:
        print(f"Eksik değer bulundu: {eksik} adet, ortalama ile dolduruluyor...")
        df = df.fillna(df.mean(numeric_only=True))
    else:
        print("Eksik değer yok.")
    return df


# =============================================================
# NORMALİZASYON
# Önemli kural: Scaler sadece train verisi üzerinde fit edilir!
# Aynı scaler validation ve test verisine uygulanır.
# =============================================================

def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Train verisi üzerinde scaler'ı eğitir ve döndürür.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    """
    Daha önce eğitilmiş scaler'ı veriye uygular.
    """
    return scaler.transform(X)


# =============================================================
# PCA - BOYUT İNDİRGEME
# Otomata modeli tek boyutlu veri ile çalışır.
# Çok değişkenli veriyi tek boyuta indirgemek için PCA kullanırız.
# Önemli kural: PCA sadece train verisi üzerinde fit edilir!
# =============================================================

def fit_pca(X_train: np.ndarray) -> PCA:
    """
    Train verisi üzerinde PCA'yı eğitir ve döndürür.
    Tek bileşen (PC1) kullanılır.
    """
    pca = PCA(n_components=1)
    pca.fit(X_train)
    print(f"PCA açıklanan varyans oranı (PC1): {pca.explained_variance_ratio_[0]:.4f}")
    return pca


def apply_pca(pca: PCA, X: np.ndarray) -> np.ndarray:
    """
    Daha önce eğitilmiş PCA'yı veriye uygular.
    Tek boyutlu dizi döndürür.
    """
    return pca.transform(X).flatten()


# =============================================================
# SKAB İÇİN TAM ÖN İŞLEME PIPELINE'I
# =============================================================

def preprocess_skab(df: pd.DataFrame, scaler=None, pca=None, fit=True):
    """
    SKAB verisini baştan sona hazırlar.

    Parametreler:
    - df     : ham SKAB verisi
    - scaler : daha önce eğitilmiş scaler (None ise yeni oluşturulur)
    - pca    : daha önce eğitilmiş PCA (None ise yeni oluşturulur)
    - fit    : True ise scaler ve PCA bu veri üzerinde eğitilir (train için)
               False ise sadece dönüşüm uygulanır (val/test için)

    Döndürür:
    - X_scaled : normalizasyon uygulanmış sensör verisi
    - X_pca    : PCA uygulanmış tek boyutlu veri (otomata için)
    - y        : hedef değişken (anomaly)
    - scaler   : eğitilmiş scaler
    - pca      : eğitilmiş PCA
    """

    # eksik veri temizle
    df = handle_missing(df)

    # sensör sütunlarını al
    X = get_skab_features(df)
    y = df[config.SKAB_TARGET].values

    if fit:
        scaler = fit_scaler(X)
        X_scaled = apply_scaler(scaler, X)
        pca = fit_pca(X_scaled)
        X_pca = apply_pca(pca, X_scaled)
    else:
        X_scaled = apply_scaler(scaler, X)
        X_pca = apply_pca(pca, X_scaled)

    return X_scaled, X_pca, y, scaler, pca


# =============================================================
# BATADAL İÇİN TAM ÖN İŞLEME PIPELINE'I
# =============================================================

def preprocess_batadal(df: pd.DataFrame, scaler=None, pca=None, fit=True):
    """
    BATADAL verisini baştan sona hazırlar.
    preprocess_skab ile aynı mantıkta çalışır.
    """

    # -999 etiketlerini düzelt
    df = fix_batadal_labels(df)

    # eksik veri temizle
    df = handle_missing(df)

    # sensör sütunlarını al
    X = get_batadal_features(df)
    y = df[config.BATADAL_TARGET].values

    if fit:
        scaler = fit_scaler(X)
        X_scaled = apply_scaler(scaler, X)
        pca = fit_pca(X_scaled)
        X_pca = apply_pca(pca, X_scaled)
    else:
        X_scaled = apply_scaler(scaler, X)
        X_pca = apply_pca(pca, X_scaled)

    return X_scaled, X_pca, y, scaler, pca


# =============================================================
# Test
# =============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("SKAB ÖN İŞLEME TESTİ")
    print("=" * 50)
    skab = load_skab()
    X_scaled, X_pca, y, scaler, pca = preprocess_skab(skab, fit=True)
    print(f"X_scaled boyutu : {X_scaled.shape}")
    print(f"X_pca boyutu    : {X_pca.shape}")
    print(f"y boyutu        : {y.shape}")
    print(f"Anomali oranı   : {y.mean():.4f}")

    print("\n" + "=" * 50)
    print("BATADAL ÖN İŞLEME TESTİ")
    print("=" * 50)
    batadal = load_batadal()
    X_scaled, X_pca, y, scaler, pca = preprocess_batadal(batadal, fit=True)
    print(f"X_scaled boyutu : {X_scaled.shape}")
    print(f"X_pca boyutu    : {X_pca.shape}")
    print(f"y boyutu        : {y.shape}")
    print(f"Anomali oranı   : {y.mean():.4f}")
