# =============================================================
# preprocessing/data_loader.py - Veri Yükleme Modülü
# SKAB ve BATADAL veri setlerini yükler ve hazırlar.
# =============================================================

import os
import pandas as pd
import sys

# config.py'ı import edebilmek için ana klasörü path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_skab() -> pd.DataFrame:
    """
    SKAB veri setini yükler.
    valve1 ve valve2 klasörlerindeki tüm CSV dosyalarını
    okuyup tek bir tabloda birleştirir.

    Döndürdüğü tablo şu ek sütunları içerir:
    - source_group : 'valve1' veya 'valve2'
    - source_file  : dosyanın adı (örn: '1.csv')
    """

    tum_dosyalar = []  # tüm CSV'leri buraya toplayacağız

    for klasor_adi in config.SKAB_FOLDERS:  # valve1, valve2
        klasor_yolu = os.path.join(config.SKAB_DIR, klasor_adi)

        if not os.path.exists(klasor_yolu):
            print(f"UYARI: {klasor_yolu} klasörü bulunamadı, atlanıyor.")
            continue

        # klasördeki tüm CSV dosyalarını bul
        csv_dosyalari = [f for f in os.listdir(klasor_yolu) if f.endswith(".csv")]

        if len(csv_dosyalari) == 0:
            print(f"UYARI: {klasor_yolu} klasöründe CSV dosyası bulunamadı.")
            continue

        for dosya_adi in sorted(csv_dosyalari):
            dosya_yolu = os.path.join(klasor_yolu, dosya_adi)

            # CSV'yi oku (SKAB noktalı virgülle ayrılmış)
            df = pd.read_csv(dosya_yolu, sep=";")

            # hangi klasör ve dosyadan geldiğini işaretle
            df["source_group"] = klasor_adi   # valve1 veya valve2
            df["source_file"]  = dosya_adi    # örn: 1.csv

            tum_dosyalar.append(df)
            print(f"Yüklendi: {klasor_adi}/{dosya_adi} → {len(df)} satır")

    if len(tum_dosyalar) == 0:
        raise FileNotFoundError("Hiç SKAB verisi yüklenemedi. Klasör yollarını kontrol et.")

    # hepsini üst üste birleştir
    skab_df = pd.concat(tum_dosyalar, ignore_index=True)
    print(f"\nSKAB toplam: {len(skab_df)} satır, {len(skab_df.columns)} sütun")

    return skab_df


def load_batadal() -> pd.DataFrame:
    """
    BATADAL veri setini yükler.
    Sütun adlarındaki baştaki/sondaki boşlukları temizler.
    """

    dosya_yolu = os.path.join(config.BATADAL_DIR, config.BATADAL_TRAIN_FILE)

    if not os.path.exists(dosya_yolu):
        raise FileNotFoundError(f"BATADAL dosyası bulunamadı: {dosya_yolu}")

    df = pd.read_csv(dosya_yolu)

    # sütun adlarındaki baştaki/sondaki boşlukları temizle
    # (terminalden gördüğümüz ' ATT_FLAG' gibi boşlukları kaldırır)
    df.columns = df.columns.str.strip()

    print(f"BATADAL yüklendi: {len(df)} satır, {len(df.columns)} sütun")

    return df


def get_skab_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    SKAB verisinden sadece model girdisi olacak sensör sütunlarını döndürür.
    datetime, changepoint, source_group, source_file, anomaly çıkarılır.
    """

    girdiye_dahil_edilmeyecekler = [
        "datetime",
        "changepoint",
        "source_group",
        "source_file",
        config.SKAB_TARGET  # anomaly
    ]

    # bu sütunlar dışındaki her şey sensör verisi
    sensor_sutunlari = [s for s in df.columns if s not in girdiye_dahil_edilmeyecekler]

    return df[sensor_sutunlari]


def get_batadal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    BATADAL verisinden sadece model girdisi olacak sensör sütunlarını döndürür.
    Zaman sütunu ve hedef değişken çıkarılır.
    """

    girdiye_dahil_edilmeyecekler = [
        "DATETIME",
        config.BATADAL_TARGET  # ATT_FLAG
    ]

    sensor_sutunlari = [s for s in df.columns if s not in girdiye_dahil_edilmeyecekler]

    return df[sensor_sutunlari]


# -------------------------------------------------------------
# Test: bu dosyayı direkt çalıştırırsan veriyi yükleyip
# özet bilgi yazdırır. Import edildiğinde çalışmaz.
# -------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("SKAB VERİSİ YÜKLENİYOR...")
    print("=" * 50)
    skab = load_skab()
    print("\nİlk 3 satır:")
    print(skab.head(3))
    print("\nSütunlar:", skab.columns.tolist())

    print("\n" + "=" * 50)
    print("BATADAL VERİSİ YÜKLENİYOR...")
    print("=" * 50)
    batadal = load_batadal()
    print("\nİlk 3 satır:")
    print(batadal.head(3))
    print("\nSütunlar:", batadal.columns.tolist())
