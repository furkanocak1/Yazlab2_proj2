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
    4. Geçiş olasılıkları : pattern'lar arası geçişleri say, olasılık hesapla
    5. Anomali tespiti : düşük log olasılıklı yollar anomali

    NOT: Çarpım yerine log toplamı kullanıyoruz.
      log(p1 * p2 * p3) = log(p1) + log(p2) + log(p3)
    """

    def __init__(self, window_size=None, alphabet_size=None):
        self.window_size   = window_size   or config.WINDOW_SIZE
        self.alphabet_size = alphabet_size or config.ALPHABET_SIZE

        # Skorlama için kullanılan pencere boyutu
        # Yeterince pattern çıkabilmesi için büyük tutuyoruz
        self.score_window = 100

        self.breakpoints       = None
        self.known_patterns    = set()
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs  = {}
        self.threshold         = None

    # ==========================================================
    # ADIM 1: PAA
    # ==========================================================

    def paa(self, zaman_serisi: np.ndarray) -> np.ndarray:
        """
        Zaman serisini window_size'lık parçalara böler,
        her parçanın ortalamasını alır.
        """
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
        """Train verisi üzerinde SAX breakpoint'lerini hesaplar."""
        yüzdeler = np.linspace(0, 100, self.alphabet_size + 1)[1:-1]
        self.breakpoints = np.percentile(paa_dizi, yüzdeler)
        return self.breakpoints

    def sayiyi_harfe_cevir(self, sayi: float) -> str:
        """Tek bir sayıyı harfe çevirir."""
        harf_indeksi = np.searchsorted(self.breakpoints, sayi)
        return chr(ord('a') + harf_indeksi)

    def sax(self, paa_dizi: np.ndarray) -> list:
        """PAA dizisini SAX sembollerine çevirir."""
        return [self.sayiyi_harfe_cevir(x) for x in paa_dizi]

    # ==========================================================
    # ADIM 3: SLIDING WINDOW
    # ==========================================================

    def sliding_window(self, sax_sembolleri: list) -> list:
        """SAX sembollerinden sliding window ile pattern'lar çıkarır."""
        pattern_listesi = []
        for i in range(len(sax_sembolleri) - self.window_size + 1):
            pattern = ''.join(sax_sembolleri[i : i + self.window_size])
            pattern_listesi.append(pattern)
        return pattern_listesi

    def zaman_serisi_to_patterns(self, zaman_serisi: np.ndarray) -> list:
        """Zaman serisini pattern listesine dönüştürür."""
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

        self.transition_probs = {}
        for kaynak, hedefler in self.transition_counts.items():
            toplam = sum(hedefler.values())
            self.transition_probs[kaynak] = {
                hedef: sayi / toplam
                for hedef, sayi in hedefler.items()
            }

        # Eşik değeri: tüm train pattern geçişlerinin log olasılıkları
        train_skorlar = self._hesapla_skorlar(X_pca_train)
        self.threshold = np.percentile(train_skorlar, 10)
        print(f"Skor aralığı: [{train_skorlar.min():.4f}, {train_skorlar.max():.4f}]")
        print(f"Anomali eşik değeri (log): {self.threshold:.4f}")
        print("Otomata eğitimi tamamlandı.\n")

    # ==========================================================
    # UNSEEN PATTERN YÖNETİMİ
    # ==========================================================

    def en_yakin_pattern(self, pattern: str) -> tuple:
        """Bilinmeyen pattern için en yakın bilinen pattern'ı bulur."""
        en_yakin  = None
        en_mesafe = float('inf')
        for bilinen in self.known_patterns:
            mesafe = levenshtein_distance(pattern, bilinen)
            if mesafe < en_mesafe:
                en_mesafe = mesafe
                en_yakin  = bilinen
        return en_yakin, en_mesafe

    def pattern_coz(self, pattern: str) -> tuple:
        """Pattern bilinen mi bilinmeyen mi kontrol eder."""
        if pattern in self.known_patterns:
            return pattern, 'known', 0
        else:
            en_yakin, mesafe = self.en_yakin_pattern(pattern)
            return en_yakin, 'unseen', mesafe

    def gecis_olasiligi(self, kaynak: str, hedef: str) -> float:
        """İki pattern arasındaki geçiş olasılığını döndürür."""
        smoothing = 1e-6
        if kaynak not in self.transition_probs:
            return smoothing
        return self.transition_probs[kaynak].get(hedef, smoothing)

    # ==========================================================
    # LOG PATH PROBABILITY
    # ==========================================================

    def log_path_probability(self, pattern_listesi: list) -> float:
        """
        Bir pattern dizisinin LOG olasılığını hesaplar.
        Daha düşük log değeri → daha anormal.
        """
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
        """
        Veri için log path probability skorlarını hesaplar.

        score_window boyutunda kayan pencere kullanır.
        Pencere yeterince büyük olmalı ki içinde birden fazla
        pattern çıkabilsin ve geçiş hesaplanabilsin.

        Daha düşük skor → daha anormal.
        """
        skorlar = []
        adim    = self.window_size  # kaydırma miktarı

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
        """
        skorlar   = self._hesapla_skorlar(X_pca)
        tahminler = (skorlar < self.threshold).astype(int)

        # Tahminleri orijinal veri uzunluğuna genişlet
        tahminler_tam = np.zeros(len(X_pca), dtype=int)
        adim = self.window_size

        for idx, tahmin in enumerate(tahminler):
            baslangic = idx * adim
            bitis     = min(baslangic + self.score_window, len(X_pca))
            tahminler_tam[baslangic:bitis] = tahmin

        return tahminler_tam

    def predict_proba(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Her pencere için anomali skoru döndürür (0-1 arası).
        Düşük log probability → yüksek anomali skoru.
        """
        skorlar = self._hesapla_skorlar(X_pca)
        min_s = skorlar.min()
        max_s = skorlar.max()
        if max_s == min_s:
            return np.zeros(len(skorlar))
        normalize = (skorlar - min_s) / (max_s - min_s)
        return 1 - normalize


# =============================================================
# Test
# =============================================================

if __name__ == "__main__":

    from preprocessing.data_loader import load_skab
    from preprocessing.data_splitter import split_skab_kfold

    print("=" * 50)
    print("OTOMATA MODELİ TESTİ - SKAB")
    print("=" * 50)

    skab    = load_skab()
    foldlar = split_skab_kfold(skab, n_splits=5)
    fold    = foldlar[0]

    model = ProbabilisticAutomata(
        window_size=config.WINDOW_SIZE,
        alphabet_size=config.ALPHABET_SIZE
    )

    model.fit(fold["X_pca_train"], fold["y_train"])

    tahminler = model.predict(fold["X_pca_test"])
    print(f"Tahmin edilen anomali oranı : {tahminler.mean():.4f}")
    print(f"Gerçek anomali oranı        : {fold['y_test'].mean():.4f}")
    print(f"Bilinen pattern sayısı      : {len(model.known_patterns)}")
    print(f"Anomali eşik değeri (log)   : {model.threshold:.4f}")
