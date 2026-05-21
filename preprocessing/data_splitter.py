# =============================================================
# preprocessing/data_splitter.py - Veri Bölme Modülü
# SKAB için GroupKFold, BATADAL için zaman sıralı bölme yapar.
# =============================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.data_loader import load_skab, load_batadal
from preprocessing.preprocessor import preprocess_skab, preprocess_batadal


# =============================================================
# BATADAL - ZAMAN SIRALI BÖLME
# %60 train, %20 validation, %20 test
# Satırlar zaman sırasına göre ayrılır, karıştırılmaz.
# =============================================================

def split_batadal(df: pd.DataFrame):
    """
    BATADAL verisini zaman sırasına göre böler.
    Önce ön işleme yapar, sonra böler.

    Döndürür:
    - X_train, X_val, X_test : normalize edilmiş sensör verisi
    - X_pca_train, X_pca_val, X_pca_test : PCA uygulanmış veri (otomata için)
    - y_train, y_val, y_test : etiketler
    - scaler, pca : eğitilmiş scaler ve PCA nesneleri
    """

    n = len(df)
    train_sonu = int(n * config.TRAIN_RATIO)        # %60
    val_sonu   = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))  # %80

    df_train = df.iloc[:train_sonu]
    df_val   = df.iloc[train_sonu:val_sonu]
    df_test  = df.iloc[val_sonu:]

    print(f"BATADAL bölme:")
    print(f"  Train : {len(df_train)} satır ({len(df_train)/n*100:.1f}%)")
    print(f"  Val   : {len(df_val)} satır ({len(df_val)/n*100:.1f}%)")
    print(f"  Test  : {len(df_test)} satır ({len(df_test)/n*100:.1f}%)")

    # Scaler ve PCA sadece train üzerinde fit edilir
    X_train, X_pca_train, y_train, scaler, pca = preprocess_batadal(df_train, fit=True)
    X_val,   X_pca_val,   y_val,   _,      _   = preprocess_batadal(df_val,   scaler=scaler, pca=pca, fit=False)
    X_test,  X_pca_test,  y_test,  _,      _   = preprocess_batadal(df_test,  scaler=scaler, pca=pca, fit=False)

    return (
        X_train, X_val, X_test,
        X_pca_train, X_pca_val, X_pca_test,
        y_train, y_val, y_test,
        scaler, pca
    )


# =============================================================
# SKAB - GROUPKFOLD BÖLME
# Aynı CSV dosyasına ait satırlar hem train hem test'te olmaz.
# Her fold için scaler ve PCA yeniden fit edilir.
# =============================================================

def split_skab_kfold(df: pd.DataFrame, n_splits: int = 5):
    """
    SKAB verisini dosya bazlı GroupKFold ile böler.

    Her iterasyonda bir fold test, kalanlar train olur.
    Validation olarak train'in son %20'si ayrılır.

    Döndürür:
    - Her fold için sözlük listesi:
      {
        'fold'       : fold numarası,
        'X_train'    : train sensör verisi,
        'X_val'      : validation sensör verisi,
        'X_test'     : test sensör verisi,
        'X_pca_train': train PCA verisi,
        'X_pca_val'  : validation PCA verisi,
        'X_pca_test' : test PCA verisi,
        'y_train'    : train etiketleri,
        'y_val'      : validation etiketleri,
        'y_test'     : test etiketleri,
        'scaler'     : eğitilmiş scaler,
        'pca'        : eğitilmiş PCA,
        'test_files' : test'teki dosya isimleri
      }
    """

    from preprocessing.preprocessor import (
        handle_missing, fix_batadal_labels,
        fit_scaler, apply_scaler, fit_pca, apply_pca
    )
    from preprocessing.data_loader import get_skab_features

    df = handle_missing(df.copy())

    # GroupKFold için grup değişkeni: source_file
    groups = df["source_file"].values
    X_full = get_skab_features(df)
    y_full = df[config.SKAB_TARGET].values

    gkf = GroupKFold(n_splits=n_splits)
    foldlar = []

    for fold_no, (train_idx, test_idx) in enumerate(gkf.split(X_full, y_full, groups)):

        # train ve test ayır
        X_train_full = X_full.iloc[train_idx]
        y_train_full = y_full[train_idx]
        X_test_fold  = X_full.iloc[test_idx]
        y_test_fold  = y_full[test_idx]

        # train'den validation ayır (son %20)
        val_baslangic = int(len(X_train_full) * 0.8)
        X_train_fold = X_train_full.iloc[:val_baslangic]
        y_train_fold = y_train_full[:val_baslangic]
        X_val_fold   = X_train_full.iloc[val_baslangic:]
        y_val_fold   = y_train_full[val_baslangic:]

        # Scaler ve PCA sadece train üzerinde fit et
        scaler = fit_scaler(X_train_fold)
        X_train_scaled = apply_scaler(scaler, X_train_fold)
        X_val_scaled   = apply_scaler(scaler, X_val_fold)
        X_test_scaled  = apply_scaler(scaler, X_test_fold)

        pca = fit_pca(X_train_scaled)
        X_pca_train = apply_pca(pca, X_train_scaled)
        X_pca_val   = apply_pca(pca, X_val_scaled)
        X_pca_test  = apply_pca(pca, X_test_scaled)

        test_dosyalari = list(df.iloc[test_idx]["source_file"].unique())

        print(f"Fold {fold_no+1}: train={len(X_train_fold)}, val={len(X_val_fold)}, test={len(X_test_fold)} | test dosyaları: {test_dosyalari}")

        foldlar.append({
            "fold"        : fold_no + 1,
            "X_train"     : X_train_scaled,
            "X_val"       : X_val_scaled,
            "X_test"      : X_test_scaled,
            "X_pca_train" : X_pca_train,
            "X_pca_val"   : X_pca_val,
            "X_pca_test"  : X_pca_test,
            "y_train"     : y_train_fold,
            "y_val"       : y_val_fold,
            "y_test"      : y_test_fold,
            "scaler"      : scaler,
            "pca"         : pca,
            "test_files"  : test_dosyalari
        })

    return foldlar


# =============================================================
# Test
# =============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("BATADAL BÖLME TESTİ")
    print("=" * 50)
    batadal = load_batadal()
    sonuc = split_batadal(batadal)
    X_train, X_val, X_test = sonuc[0], sonuc[1], sonuc[2]
    y_train, y_val, y_test = sonuc[6], sonuc[7], sonuc[8]
    print(f"\nTrain anomali oranı : {y_train.mean():.4f}")
    print(f"Val   anomali oranı : {y_val.mean():.4f}")
    print(f"Test  anomali oranı : {y_test.mean():.4f}")

    print("\n" + "=" * 50)
    print("SKAB GROUPKFOLD TESTİ")
    print("=" * 50)
    skab = load_skab()
    foldlar = split_skab_kfold(skab, n_splits=5)
    print(f"\nToplam fold sayısı: {len(foldlar)}")
