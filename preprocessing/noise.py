# =============================================================
# preprocessing/noise.py - Gürültü Ekleme Modülü
# Normal veriye Gaussian gürültü ekler.
# Modelin gürültüye dayanıklılığını test etmek için kullanılır.
# =============================================================

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def gaussian_gurultu_ekle(veri: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Veriye Gaussian gürültü ekler.

    Gaussian gürültü nedir?
    Her değere küçük rastgele bir sayı ekliyoruz.
    Bu sayılar normal dağılımdan geliyor:
      - Çoğu gürültü sıfıra yakın (küçük etki)
      - Nadiren büyük sapmalar oluyor

    Parametreler:
    - veri : gürültü eklenecek numpy dizisi
    - seed : tekrarlanabilirlik için random seed

    config.py'dan alınan parametreler:
    - NOISE_MEAN = 0.0  (gürültünün ortalaması)
    - NOISE_STD  = 0.1  (gürültünün standart sapması)
    """
    np.random.seed(seed)

    gurultu = np.random.normal(
        loc   = config.NOISE_MEAN,  # ortalama
        scale = config.NOISE_STD,   # standart sapma
        size  = veri.shape
    )

    gurultulu_veri = veri + gurultu

    print(f"Gürültü eklendi:")
    print(f"  Orjinal veri  - Ortalama: {veri.mean():.4f}, Std: {veri.std():.4f}")
    print(f"  Gürültülü veri - Ortalama: {gurultulu_veri.mean():.4f}, Std: {gurultulu_veri.std():.4f}")

    return gurultulu_veri


def unseen_veri_olustur(
    X_pca_train : np.ndarray,
    X_pca_test  : np.ndarray,
    model,
    seed        : int = 42
) -> np.ndarray:
    """
    Test verisinde unseen pattern'lar oluşturur.

    Nasıl çalışır?
    - Eğitim verisinden SAX sözlüğü çıkarılır (known patterns)
    - Test verisine gürültü eklenerek bazı pattern'ların
      sözlükte bulunmaması sağlanır
    - Bu sayede unseen senaryosu simüle edilir

    Parametreler:
    - X_pca_train : eğitim verisi (known pattern'lar bundan çıkarıldı)
    - X_pca_test  : test verisi
    - model       : eğitilmiş otomata modeli
    - seed        : tekrarlanabilirlik için random seed

    Döndürür:
    - unseen_veri    : unseen pattern içeren test verisi
    - unseen_orani   : kaç pattern'ın unseen olduğu oranı
    """
    np.random.seed(seed)

    # Farklı gürültü seviyeleri dene, yeterince unseen pattern çıkana kadar
    for gurultu_std in [0.5, 1.0, 2.0, 3.0]:
        gurultu = np.random.normal(0, gurultu_std, X_pca_test.shape)
        unseen_veri = X_pca_test + gurultu

        # Kaç pattern unseen?
        patterns = model.zaman_serisi_to_patterns(unseen_veri)
        unseen_sayisi = sum(
            1 for p in patterns if p not in model.known_patterns
        )
        unseen_orani = unseen_sayisi / len(patterns) if patterns else 0

        print(f"Gürültü std={gurultu_std:.1f}: {unseen_sayisi}/{len(patterns)} unseen pattern ({unseen_orani*100:.1f}%)")

        if unseen_orani >= 0.05:  # en az %5 unseen pattern olsun
            break

    return unseen_veri, unseen_orani


# =============================================================
# Test
# =============================================================

if __name__ == "__main__":

    from preprocessing.data_loader import load_skab, load_batadal
    from preprocessing.data_splitter import split_skab_kfold, split_batadal
    from models.automata.automata import ProbabilisticAutomata

    print("=" * 55)
    print("GÜRÜLTÜ MODÜLÜ TESTİ - SKAB")
    print("=" * 55)

    skab    = load_skab()
    foldlar = split_skab_kfold(skab, n_splits=5)
    fold    = foldlar[0]

    # Model eğit
    model = ProbabilisticAutomata(
        window_size=config.WINDOW_SIZE,
        alphabet_size=config.ALPHABET_SIZE
    )
    model.fit(fold["X_pca_train"], fold["y_train"])

    # Normal test verisi
    print("\n--- Normal Veri ---")
    tahmin_normal = model.predict(fold["X_pca_test"])
    print(f"Anomali oranı: {tahmin_normal.mean():.4f}")

    # Gürültülü test verisi
    print("\n--- Gürültülü Veri ---")
    gurultulu = gaussian_gurultu_ekle(fold["X_pca_test"])
    tahmin_gurultulu = model.predict(gurultulu)
    print(f"Anomali oranı: {tahmin_gurultulu.mean():.4f}")

    # Unseen veri
    print("\n--- Unseen Veri ---")
    unseen, unseen_orani = unseen_veri_olustur(
        fold["X_pca_train"],
        fold["X_pca_test"],
        model
    )
    tahmin_unseen = model.predict(unseen)
    print(f"Anomali oranı: {tahmin_unseen.mean():.4f}")
    print(f"Unseen pattern oranı: {unseen_orani*100:.1f}%")

    print("\n" + "=" * 55)
    print("GÜRÜLTÜ MODÜLÜ TESTİ - BATADAL")
    print("=" * 55)

    batadal = load_batadal()
    sonuc   = split_batadal(batadal)
    X_pca_train, X_pca_test = sonuc[3], sonuc[5]
    y_train                 = sonuc[6]

    model_b = ProbabilisticAutomata(
        window_size=config.WINDOW_SIZE,
        alphabet_size=config.ALPHABET_SIZE
    )
    model_b.fit(X_pca_train, y_train)

    print("\n--- Normal Veri ---")
    tahmin_normal_b = model_b.predict(X_pca_test)
    print(f"Anomali oranı: {tahmin_normal_b.mean():.4f}")

    print("\n--- Gürültülü Veri ---")
    gurultulu_b = gaussian_gurultu_ekle(X_pca_test)
    tahmin_gurultulu_b = model_b.predict(gurultulu_b)
    print(f"Anomali oranı: {tahmin_gurultulu_b.mean():.4f}")

    print("\n--- Unseen Veri ---")
    unseen_b, unseen_orani_b = unseen_veri_olustur(
        X_pca_train, X_pca_test, model_b
    )
    tahmin_unseen_b = model_b.predict(unseen_b)
    print(f"Anomali oranı: {tahmin_unseen_b.mean():.4f}")
    print(f"Unseen pattern oranı: {unseen_orani_b*100:.1f}%")
