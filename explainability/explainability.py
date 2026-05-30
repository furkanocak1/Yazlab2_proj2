# =============================================================
# explainability/explainability.py - Açıklanabilirlik Modülü
# Her karar için detaylı açıklama ve JSON çıktısı üretir.
# =============================================================

import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.automata.automata import ProbabilisticAutomata


class AutomataExplainer:
    """
    Otomata modelinin kararlarını açıklar.

    Her karar için şunları üretir:
    - Mevcut state ve gelen pattern
    - Pattern daha önce görüldü mü (known/unseen)
    - Unseen ise Levenshtein ile neye eşlendi
    - Gerçekleşen geçişler ve olasılıkları
    - Toplam log path probability
    - Nihai karar (anomali/normal) ve güven skoru
    """

    def __init__(self, model: ProbabilisticAutomata):
        self.model = model

    # ==========================================================
    # TEK BİR PENCEREYİ AÇIKLA
    # ==========================================================

    def explain_window(self, zaman_serisi: np.ndarray, time_step: int = 0) -> dict:
        """
        Tek bir zaman penceresi için açıklama üretir.

        Parametreler:
        - zaman_serisi : açıklanacak pencere (numpy dizisi)
        - time_step    : kaçıncı adım olduğu (raporlama için)

        Döndürür:
        - Açıklama sözlüğü (JSON'a dönüştürülebilir)
        """

        # Zaman serisini pattern'lara çevir
        pattern_listesi = self.model.zaman_serisi_to_patterns(zaman_serisi)

        if len(pattern_listesi) < 2:
            return {
                "time_step" : time_step,
                "hata"      : "Yeterli pattern çıkarılamadı",
                "karar"     : "belirsiz"
            }

        # Her pattern için known/unseen durumu ve geçişleri hesapla
        gecisler      = []
        log_toplam    = 0.0
        onceki_pattern = None

        for i, pattern in enumerate(pattern_listesi):
            cozulen, durum, mesafe = self.model.pattern_coz(pattern)

            if onceki_pattern is not None:
                # Geçiş olasılığını hesapla
                olasilik     = self.model.gecis_olasiligi(onceki_pattern, cozulen)
                log_olasilik = np.log(olasilik)
                log_toplam  += log_olasilik

                gecis_bilgisi = {
                    "adim"          : i,
                    "kaynak"        : onceki_pattern,
                    "hedef"         : cozulen,
                    "ham_pattern"   : pattern,
                    "durum"         : durum,
                    "olasilik"      : round(olasilik, 6),
                    "log_olasilik"  : round(log_olasilik, 4),
                }

                if durum == "unseen":
                    gecis_bilgisi["eslesme_mesafesi"] = mesafe
                    gecis_bilgisi["eslestirilen"]     = cozulen

                gecisler.append(gecis_bilgisi)

            onceki_pattern = cozulen

        # Karar ve güven skoru
        anomali_mi = log_toplam < self.model.threshold

        # Güven skoru: log olasılığı 0-1 arasına normalize et
        # Ne kadar düşük log → o kadar yüksek anomali güveni
        esik        = self.model.threshold
        guvenscore  = self._guvenscore_hesapla(log_toplam, esik)

        # Unseen pattern sayısı
        unseen_sayisi = sum(1 for g in gecisler if g["durum"] == "unseen")

        aciklama = {
            "time_step"          : time_step,
            "ilk_state"          : pattern_listesi[0],
            "son_state"          : pattern_listesi[-1],
            "toplam_pattern"     : len(pattern_listesi),
            "unseen_pattern_say" : unseen_sayisi,
            "gecisler"           : gecisler,
            "log_path_prob"      : round(log_toplam, 4),
            "esik_degeri"        : round(esik, 4),
            "karar"              : "anomali" if anomali_mi else "normal",
            "guven_skoru"        : round(guvenscore, 4),
        }

        return aciklama

    def _guvenscore_hesapla(self, log_prob: float, esik: float) -> float:
        """
        Log probability'den güven skoru üretir (0-1 arası).

        Anomali kararı için: eşikten ne kadar düşük → o kadar güvenli
        Normal karar için  : eşikten ne kadar yüksek → o kadar güvenli
        """
        fark = abs(log_prob - esik)
        # fark büyüdükçe güven artar, sigmoid benzeri normalize
        guven = 1 - np.exp(-fark / 5)
        return float(np.clip(guven, 0.0, 1.0))

    # ==========================================================
    # YAZILI AÇIKLAMA ÜRETİCİ
    # ==========================================================

    def yazili_acikla(self, aciklama: dict) -> str:
        """
        Sözlük formatındaki açıklamayı okunabilir metne çevirir.
        Dökümanın istediği [SYSTEM DECISION] formatında.
        """

        satirlar = []
        satirlar.append("=" * 55)
        satirlar.append("[SYSTEM DECISION]")
        satirlar.append("=" * 55)
        satirlar.append(f"Time Step       : {aciklama['time_step']}")
        satirlar.append(f"İlk State       : {aciklama['ilk_state']}")
        satirlar.append(f"Son State       : {aciklama['son_state']}")
        satirlar.append(f"Toplam Pattern  : {aciklama['toplam_pattern']}")
        satirlar.append(f"Unseen Pattern  : {aciklama['unseen_pattern_say']}")
        satirlar.append("")
        satirlar.append("Geçişler:")

        for g in aciklama["gecisler"][:5]:  # ilk 5 geçişi göster
            durum_str = ""
            if g["durum"] == "unseen":
                durum_str = f" [UNSEEN → {g['eslestirilen']} (mesafe={g['eslesme_mesafesi']})]"
            satirlar.append(
                f"  {g['kaynak']} → {g['hedef']} : {g['olasilik']:.6f}{durum_str}"
            )

        if len(aciklama["gecisler"]) > 5:
            satirlar.append(f"  ... ve {len(aciklama['gecisler'])-5} geçiş daha")

        satirlar.append("")
        satirlar.append(f"Log Path Prob   : {aciklama['log_path_prob']}")
        satirlar.append(f"Eşik Değeri     : {aciklama['esik_degeri']}")
        satirlar.append("")
        satirlar.append(f"Karar           : {aciklama['karar'].upper()}")
        satirlar.append(f"Güven Skoru     : {aciklama['guven_skoru']}")
        satirlar.append("=" * 55)

        return "\n".join(satirlar)

    # ==========================================================
    # JSON ÇIKTI
    # ==========================================================

    def json_cikti(self, aciklama: dict) -> str:
        """Açıklamayı JSON formatında döndürür."""
        return json.dumps(aciklama, ensure_ascii=False, indent=2)

    # ==========================================================
    # TOPLU AÇIKLAMA
    # ==========================================================

    def explain_dataset(self, X_pca: np.ndarray, maks_aciklama: int = 5) -> list:
        """
        Veri seti üzerinde birden fazla pencere için açıklama üretir.
        Sadece anomali olarak işaretlenen pencereler açıklanır.

        Parametreler:
        - X_pca        : tüm veri
        - maks_aciklama: kaç pencere açıklanacak (çok uzun olmasın)
        """

        aciklamalar = []
        adim        = self.model.window_size
        pencere     = self.model.score_window
        sayac       = 0

        for i in range(0, len(X_pca) - pencere + 1, adim):
            if sayac >= maks_aciklama:
                break

            dilim     = X_pca[i : i + pencere]
            aciklama  = self.explain_window(dilim, time_step=i)

            if aciklama.get("karar") == "anomali":
                aciklamalar.append(aciklama)
                sayac += 1

        return aciklamalar


# =============================================================
# Test
# =============================================================

if __name__ == "__main__":

    from preprocessing.data_loader import load_skab
    from preprocessing.data_splitter import split_skab_kfold

    print("=" * 55)
    print("AÇIKLANABİLİRLİK MODÜLÜ TESTİ - SKAB")
    print("=" * 55)

    skab    = load_skab()
    foldlar = split_skab_kfold(skab, n_splits=5)
    fold    = foldlar[0]

    # Otomata modelini eğit
    model = ProbabilisticAutomata(
        window_size=config.WINDOW_SIZE,
        alphabet_size=config.ALPHABET_SIZE
    )
    model.fit(fold["X_pca_train"], fold["y_train"])

    # Açıklayıcıyı oluştur
    explainer = AutomataExplainer(model)

    # İlk pencereyi açıkla
    print("\n--- İlk Pencere Açıklaması ---")
    ilk_pencere = fold["X_pca_test"][:model.score_window]
    aciklama    = explainer.explain_window(ilk_pencere, time_step=0)

    # Yazılı açıklama
    print(explainer.yazili_acikla(aciklama))

    # JSON çıktı
    print("\n--- JSON Çıktı ---")
    print(explainer.json_cikti(aciklama))

    # Anomali olan pencereleri bul ve açıkla
    print("\n--- Anomali Tespiti Açıklamaları (ilk 3) ---")
    anomali_aciklamalari = explainer.explain_dataset(
        fold["X_pca_test"],
        maks_aciklama=3
    )
    print(f"Tespit edilen anomali penceresi: {len(anomali_aciklamalari)}")
    for a in anomali_aciklamalari:
        print(explainer.yazili_acikla(a))
        print()
