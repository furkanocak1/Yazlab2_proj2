# =============================================================
# models/automata/automata.py - Olasılıksal Otomata Modeli
# PAA + SAX + Sliding Window ile anomali tespiti yapar.
# =============================================================

import os
import sys
import numpy as np
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    def levenshtein_distance(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]


class ProbabilisticAutomata:
    """
    Olasılıksal Otomata Modeli.

    Çalışma mantığı:
    1. PAA  : veriyi parçalara böl, her parçanın ortalamasını al
    2. SAX  : ortalamaları harflere çevir
    3. Sliding Window : art arda gelen harfleri grupla → pattern
    4. Geçiş olasılıkları : pattern'lar arası geçişleri say
    5. Anomali tespiti : validation setiyle kalibre edilmiş eşik

    Eşik kalibrasyonu:
    - Train verisiyle model eğitilir
    - Validation verisinde farklı eşikler denenir
    - En iyi F1 skorunu veren eşik seçilir
    - O eşikle test verisi değerlendirilir
    """

    def __init__(self, window_size=None, alphabet_size=None):
        self.window_size   = window_size   or config.WINDOW_SIZE
        self.alphabet_size = alphabet_size or config.ALPHABET_SIZE
        self.score_window  = 100

        self.breakpoints       = None
        self.known_patterns    = set()
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_totals = {}
        self.alpha             = 0.1  # Laplace smoothing factor
        self.vocab_size        = 0

        # Train istatistikleri (z-score için)
        self.train_mean = None
        self.train_std  = None

        # Eşik değeri (validation ile kalibre edilecek)
        self.threshold     = None
        self.z_threshold   = -2.0  # varsayılan, kalibrasyonla güncellenir

    # ==========================================================
    # ADIM 1: PAA
    # ==========================================================

    def paa(self, zaman_serisi: np.ndarray) -> np.ndarray:
        n = len(zaman_serisi)
        parcalar = []
        for i in range(0, n - self.window_size + 1, self.window_size):
            parca = zaman_serisi[i : i + self.window_size]
            parcalar.append(np.mean(parca))
        return np.array(parcalar)

    # ==========================================================
    # ADIM 2: SAX
    # ==========================================================

    def fit_breakpoints(self, paa_dizi: np.ndarray):
        yüzdeler = np.linspace(0, 100, self.alphabet_size + 1)[1:-1]
        self.breakpoints = np.percentile(paa_dizi, yüzdeler)
        return self.breakpoints

    def sayiyi_harfe_cevir(self, sayi: float) -> str:
        harf_indeksi = np.searchsorted(self.breakpoints, sayi)
        return chr(ord('a') + harf_indeksi)

    def sax(self, paa_dizi: np.ndarray) -> list:
        return [self.sayiyi_harfe_cevir(x) for x in paa_dizi]

    # ==========================================================
    # ADIM 3: SLIDING WINDOW
    # ==========================================================

    def sliding_window(self, sax_sembolleri: list) -> list:
        pattern_listesi = []
        for i in range(len(sax_sembolleri) - self.window_size + 1):
            pattern = ''.join(sax_sembolleri[i : i + self.window_size])
            pattern_listesi.append(pattern)
        return pattern_listesi

    def zaman_serisi_to_patterns(self, zaman_serisi: np.ndarray) -> list:
        paa_dizi        = self.paa(zaman_serisi)
        sax_sembolleri  = self.sax(paa_dizi)
        pattern_listesi = self.sliding_window(sax_sembolleri)
        return pattern_listesi

    # ==========================================================
    # ADIM 4: GEÇİŞ OLASILIKLARINI HESAPLA
    # ==========================================================

    def fit(self, X_pca_train: np.ndarray, y_train: np.ndarray):
        """Otomata modelini train verisi üzerinde eğitir."""

        print(f"Otomata eğitiliyor... (window={self.window_size}, alphabet={self.alphabet_size})")

        tum_paa = self.paa(X_pca_train)
        self.fit_breakpoints(tum_paa)
        print(f"SAX breakpoints: {self.breakpoints}")

        tum_patterns = self.zaman_serisi_to_patterns(X_pca_train)
        self.known_patterns = set(tum_patterns)
        print(f"Toplam benzersiz pattern sayısı: {len(self.known_patterns)}")

        for i in range(len(tum_patterns) - 1):
            simdi   = tum_patterns[i]
            sonraki = tum_patterns[i + 1]
            self.transition_counts[simdi][sonraki] += 1

        self.transition_totals = {}
        for kaynak, hedefler in self.transition_counts.items():
            self.transition_totals[kaynak] = sum(hedefler.values())
            
        self.vocab_size = len(self.known_patterns)

        # Train istatistiklerini kaydet
        train_skorlar   = self._hesapla_skorlar(X_pca_train)
        self.train_mean = train_skorlar.mean()
        self.train_std  = train_skorlar.std() + 1e-10

        # Varsayılan eşik: z = -2
        self.threshold = self.train_mean + self.z_threshold * self.train_std

        print(f"Train skor aralığı : [{train_skorlar.min():.4f}, {train_skorlar.max():.4f}]")
        print(f"Train ortalama     : {self.train_mean:.4f}")
        print(f"Train std          : {self.train_std:.4f}")
        print(f"Varsayılan eşik    : {self.threshold:.4f}")
        print("Eğitim tamamlandı. Eşik kalibrasyonu için calibrate_threshold() çağırın.\n")

    # ==========================================================
    # EŞİK KALİBRASYONU (VALİDATION SETİ İLE)
    # ==========================================================

    def calibrate_threshold(self, X_pca_val: np.ndarray, y_val: np.ndarray):
        """
        Validation setindeki gerçek etiketlere bakarak
        en iyi F1 skorunu veren eşiği bulur.

        Bu işlem normalizasyon değil, sadece karar sınırını
        nereye koyacağımızı belirleme işlemidir.

        Parametreler:
        - X_pca_val : validation verisi
        - y_val     : validation gerçek etiketleri
        """
        from sklearn.metrics import f1_score

        val_skorlar = self._hesapla_skorlar(X_pca_val)

        # Validation skorlarının aralığında farklı eşikler dene
        esik_adaylari = np.percentile(val_skorlar, np.linspace(1, 99, 100))

        en_iyi_f1   = -1
        en_iyi_esik = self.threshold

        for esik in esik_adaylari:
            # Bu eşikle tahmin yap
            tahminler_pencere = (val_skorlar < esik).astype(int)

            # Pencere tahminlerini orijinal uzunluğa genişlet
            tahminler_tam = np.zeros(len(X_pca_val), dtype=int)
            for idx, tahmin in enumerate(tahminler_pencere):
                baslangic = idx * self.window_size
                bitis     = min(baslangic + self.score_window, len(X_pca_val))
                tahminler_tam[baslangic:bitis] = tahmin

            # F1 skoru hesapla
            if len(np.unique(tahminler_tam)) > 1:
                f1 = f1_score(y_val[:len(tahminler_tam)], tahminler_tam, zero_division=0)
                if f1 > en_iyi_f1:
                    en_iyi_f1   = f1
                    en_iyi_esik = esik

        self.threshold = en_iyi_esik
        print(f"Eşik kalibrasyonu tamamlandı.")
        print(f"En iyi F1 skoru : {en_iyi_f1:.4f}")
        print(f"Kalibrasyon eşiği: {self.threshold:.4f}\n")

        return en_iyi_f1

    # ==========================================================
    # UNSEEN PATTERN YÖNETİMİ
    # ==========================================================

    def en_yakin_pattern(self, pattern: str) -> tuple:
        en_yakin  = None
        en_mesafe = float('inf')
        for bilinen in self.known_patterns:
            mesafe = levenshtein_distance(pattern, bilinen)
            if mesafe < en_mesafe:
                en_mesafe = mesafe
                en_yakin  = bilinen
        return en_yakin, en_mesafe

    def pattern_coz(self, pattern: str) -> tuple:
        if pattern in self.known_patterns:
            return pattern, 'known', 0
        else:
            en_yakin, mesafe = self.en_yakin_pattern(pattern)
            return en_yakin, 'unseen', mesafe

    def gecis_olasiligi(self, kaynak: str, hedef: str) -> float:
        # Laplace Smoothing: Modelin hiç görmediği veri geçişlerinde olasılığın
        # -sonsuza (çok küçük değerlere) çökmesini engeller.
        V = self.vocab_size if self.vocab_size > 0 else 1
        
        if kaynak not in self.transition_counts:
            return 1.0 / V
            
        toplam = self.transition_totals.get(kaynak, 0)
        sayi = self.transition_counts[kaynak].get(hedef, 0)
        
        return (sayi + self.alpha) / (toplam + self.alpha * V)

    # ==========================================================
    # LOG PATH PROBABILITY
    # ==========================================================

    def log_path_probability(self, pattern_listesi: list) -> float:
        if len(pattern_listesi) < 2:
            return 0.0

        log_toplam = 0.0
        for i in range(len(pattern_listesi) - 1):
            kaynak, _, _ = self.pattern_coz(pattern_listesi[i])
            hedef,  _, _ = self.pattern_coz(pattern_listesi[i + 1])
            olasilik = self.gecis_olasiligi(kaynak, hedef)
            log_toplam += np.log(olasilik)

        return log_toplam

    def _hesapla_skorlar(self, X_pca: np.ndarray) -> np.ndarray:
        skorlar = []
        adim    = self.window_size

        for i in range(0, len(X_pca) - self.score_window + 1, adim):
            dilim    = X_pca[i : i + self.score_window]
            patterns = self.zaman_serisi_to_patterns(dilim)
            skor     = self.log_path_probability(patterns)
            skorlar.append(skor)

        return np.array(skorlar) if skorlar else np.array([0.0])

    # ==========================================================
    # TAHMİN
    # ==========================================================

    def predict(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Veri için anomali tahminleri üretir.
        0 = normal, 1 = anomali
        Eşik değerinin altındaki skorlar anomali sayılır.
        """
        skorlar   = self._hesapla_skorlar(X_pca)
        tahminler = (skorlar < self.threshold).astype(int)

        tahminler_tam = np.zeros(len(X_pca), dtype=int)
        adim = self.window_size

        for idx, tahmin in enumerate(tahminler):
            baslangic = idx * adim
            bitis     = min(baslangic + self.score_window, len(X_pca))
            tahminler_tam[baslangic:bitis] = tahmin

        return tahminler_tam

    def predict_proba(self, X_pca: np.ndarray) -> np.ndarray:
        """Her pencere için anomali skoru döndürür (0-1 arası)."""
        skorlar = self._hesapla_skorlar(X_pca)
        min_s   = skorlar.min()
        max_s   = skorlar.max()
        if max_s == min_s:
            return np.zeros(len(skorlar))
        normalize = (skorlar - min_s) / (max_s - min_s)
        return 1 - normalize


# =============================================================
# Test
# =============================================================

if __name__ == "__main__":

    from preprocessing.data_loader import load_skab, load_batadal
    from preprocessing.data_splitter import split_skab_kfold, split_batadal
    from sklearn.metrics import classification_report

    print("=" * 55)
    print("OTOMATA MODELİ TESTİ - SKAB (ilk fold)")
    print("=" * 55)

    skab    = load_skab()
    foldlar = split_skab_kfold(skab, n_splits=5)
    fold    = foldlar[0]

    model_s = ProbabilisticAutomata(
        window_size=config.WINDOW_SIZE,
        alphabet_size=config.ALPHABET_SIZE
    )
    model_s.fit(fold["X_pca_train"], fold["y_train"])
    model_s.calibrate_threshold(fold["X_pca_val"], fold["y_val"])

    tahminler_s = model_s.predict(fold["X_pca_test"])
    n = min(len(tahminler_s), len(fold["y_test"]))
    print(classification_report(fold["y_test"][:n], tahminler_s[:n], zero_division=0))

    print("\n" + "=" * 55)
    print("OTOMATA MODELİ TESTİ - BATADAL")
    print("=" * 55)

    batadal = load_batadal()
    sonuc   = split_batadal(batadal)
    X_pca_train, X_pca_val, X_pca_test = sonuc[3], sonuc[4], sonuc[5]
    y_train, y_val, y_test              = sonuc[6], sonuc[7], sonuc[8]

    model_b = ProbabilisticAutomata(
        window_size=config.WINDOW_SIZE,
        alphabet_size=config.ALPHABET_SIZE
    )
    model_b.fit(X_pca_train, y_train)
    model_b.calibrate_threshold(X_pca_val, y_val)

    tahminler_b = model_b.predict(X_pca_test)
    n = min(len(tahminler_b), len(y_test))
    print(classification_report(y_test[:n], tahminler_b[:n], zero_division=0))
