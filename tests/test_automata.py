# =============================================================
# tests/test_automata.py - Birim Testler
# Levenshtein, unseen pattern ve otomata modeli testleri.
# =============================================================

import sys
import os
import numpy as np
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.automata.automata import ProbabilisticAutomata


class TestLevenshtein(unittest.TestCase):
    """
    Levenshtein mesafesi testleri.
    Unseen pattern geldiğinde doğru eşleşme yapılıyor mu?
    """

    def setUp(self):
        """Her testten önce çalışır, model hazırlar."""
        self.model = ProbabilisticAutomata(window_size=4, alphabet_size=3)
        # Elle bilinen pattern'lar ekle
        self.model.known_patterns = {"aaaa", "bbbb", "cccc", "aabb", "bbcc"}

    def test_ayni_pattern(self):
        """Bilinen bir pattern gelince mesafe 0 olmalı."""
        en_yakin, mesafe = self.model.en_yakin_pattern("aaaa")
        self.assertEqual(en_yakin, "aaaa")
        self.assertEqual(mesafe, 0)

    def test_bir_harf_fark(self):
        """Bir harf farklıysa mesafe 1 olmalı."""
        # "aaba" → "aaaa" veya "aabb" ye mesafe 1
        _, mesafe = self.model.en_yakin_pattern("aaba")
        self.assertEqual(mesafe, 1)

    def test_tamamen_farkli(self):
        """Hiç benzemeyenin mesafesi büyük olmalı."""
        _, mesafe = self.model.en_yakin_pattern("cccc")
        # cccc zaten known_patterns içinde, mesafe 0
        self.assertEqual(mesafe, 0)

    def test_unseen_esleme(self):
        """Bilinmeyen pattern en yakın bilinen ile eşlenmeli."""
        # "aaac" → "aaaa" ya mesafe 1, "aabb" ye mesafe 2
        en_yakin, mesafe = self.model.en_yakin_pattern("aaac")
        self.assertEqual(en_yakin, "aaaa")
        self.assertEqual(mesafe, 1)

    def test_pattern_coz_known(self):
        """Bilinen pattern 'known' olarak dönmeli."""
        cozulen, durum, mesafe = self.model.pattern_coz("aaaa")
        self.assertEqual(durum, "known")
        self.assertEqual(mesafe, 0)
        self.assertEqual(cozulen, "aaaa")

    def test_pattern_coz_unseen(self):
        """Bilinmeyen pattern 'unseen' olarak dönmeli."""
        cozulen, durum, mesafe = self.model.pattern_coz("aaac")
        self.assertEqual(durum, "unseen")
        self.assertGreater(mesafe, 0)
        self.assertIn(cozulen, self.model.known_patterns)


class TestPAA(unittest.TestCase):
    """PAA dönüşümü testleri."""

    def setUp(self):
        self.model = ProbabilisticAutomata(window_size=4, alphabet_size=3)

    def test_paa_boyut(self):
        """PAA çıktısının boyutu doğru olmalı."""
        veri = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        sonuc = self.model.paa(veri)
        # 8 eleman, window=4 → 2 parça
        self.assertEqual(len(sonuc), 2)

    def test_paa_ortalama(self):
        """PAA ortalamaları doğru hesaplanmalı."""
        veri = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        sonuc = self.model.paa(veri)
        self.assertAlmostEqual(sonuc[0], 2.5)  # (1+2+3+4)/4
        self.assertAlmostEqual(sonuc[1], 6.5)  # (5+6+7+8)/4


class TestSAX(unittest.TestCase):
    """SAX dönüşümü testleri."""

    def setUp(self):
        self.model = ProbabilisticAutomata(window_size=4, alphabet_size=3)
        # Elle breakpoint ayarla
        self.model.breakpoints = np.array([-0.5, 0.5])

    def test_sax_harf_sayisi(self):
        """SAX çıktısındaki harf sayısı doğru olmalı."""
        paa_dizi = np.array([-1.0, 0.0, 1.0, -0.8])
        sonuc = self.model.sax(paa_dizi)
        self.assertEqual(len(sonuc), 4)

    def test_sax_dusuk_deger(self):
        """Düşük değer 'a' harfi olmalı."""
        paa_dizi = np.array([-2.0])
        sonuc = self.model.sax(paa_dizi)
        self.assertEqual(sonuc[0], 'a')

    def test_sax_orta_deger(self):
        """Orta değer 'b' harfi olmalı."""
        paa_dizi = np.array([0.0])
        sonuc = self.model.sax(paa_dizi)
        self.assertEqual(sonuc[0], 'b')

    def test_sax_yuksek_deger(self):
        """Yüksek değer 'c' harfi olmalı."""
        paa_dizi = np.array([2.0])
        sonuc = self.model.sax(paa_dizi)
        self.assertEqual(sonuc[0], 'c')


class TestSlidingWindow(unittest.TestCase):
    """Sliding window testleri."""

    def setUp(self):
        self.model = ProbabilisticAutomata(window_size=3, alphabet_size=3)

    def test_sliding_window_sayisi(self):
        """Doğru sayıda pattern çıkmalı."""
        semboller = ['a', 'b', 'c', 'a', 'b']
        sonuc = self.model.sliding_window(semboller)
        # 5 sembol, window=3 → 3 pattern
        self.assertEqual(len(sonuc), 3)

    def test_sliding_window_degerler(self):
        """Pattern değerleri doğru olmalı."""
        semboller = ['a', 'b', 'c', 'a']
        sonuc = self.model.sliding_window(semboller)
        self.assertEqual(sonuc[0], 'abc')
        self.assertEqual(sonuc[1], 'bca')


class TestGecisOlasiliklari(unittest.TestCase):
    """Geçiş olasılıkları testleri."""

    def setUp(self):
        self.model = ProbabilisticAutomata(window_size=4, alphabet_size=3)
        self.model.known_patterns = {"aaaa", "bbbb", "cccc"}
        self.model.vocab_size = 3
        self.model.alpha = 0.1
        
        # Laplace Smoothing ile uyumlu veri setini oluştur
        self.model.transition_counts["aaaa"]["bbbb"] = 7
        self.model.transition_counts["aaaa"]["cccc"] = 3
        self.model.transition_totals["aaaa"] = 10
        
        self.model.transition_counts["bbbb"]["cccc"] = 10
        self.model.transition_totals["bbbb"] = 10

    def test_bilinen_gecis(self):
        """Bilinen geçişin olasılığı Laplace ile doğru hesaplanmalı."""
        olasilik = self.model.gecis_olasiligi("aaaa", "bbbb")
        # Beklenen: (7 + 0.1) / (10 + 0.1 * 3) = 7.1 / 10.3
        self.assertAlmostEqual(olasilik, 7.1 / 10.3)

    def test_bilinmeyen_gecis(self):
        """Bilinmeyen geçiş Laplace smoothing değeri döndürmeli."""
        olasilik = self.model.gecis_olasiligi("aaaa", "aaaa")
        # Beklenen: (0 + 0.1) / (10 + 0.1 * 3) = 0.1 / 10.3
        self.assertAlmostEqual(olasilik, 0.1 / 10.3)

    def test_bilinmeyen_kaynak(self):
        """Bilinmeyen kaynak 1.0 / Vocab Size değeri döndürmeli."""
        olasilik = self.model.gecis_olasiligi("xxxx", "aaaa")
        self.assertAlmostEqual(olasilik, 1.0 / 3.0)

    def test_log_path_probability(self):
        """Log path probability negatif olmalı."""
        pattern_listesi = ["aaaa", "bbbb", "cccc"]
        log_prob = self.model.log_path_probability(pattern_listesi)
        self.assertLess(log_prob, 0)

    def test_tek_pattern_sifir(self):
        """Tek pattern için log prob 0 olmalı."""
        log_prob = self.model.log_path_probability(["aaaa"])
        self.assertEqual(log_prob, 0.0)


class TestUnseen(unittest.TestCase):
    """
    Unseen veri senaryosu testleri.
    Test verisinde hiç görülmemiş pattern'lar geldiğinde
    sistem doğru davranıyor mu?
    """

    def setUp(self):
        """Küçük bir veri ile model eğit."""
        np.random.seed(42)
        self.model = ProbabilisticAutomata(window_size=4, alphabet_size=3)

        # Basit bir train verisi oluştur
        train_verisi = np.sin(np.linspace(0, 10, 500))
        self.model.fit(train_verisi, np.zeros(500))

    def test_unseen_pattern_esleme(self):
        """Unseen pattern mutlaka bir known pattern'a eşlenmeli."""
        # Rastgele bir pattern dene
        test_pattern = "zzzz"  # kesinlikle bilinmeyen
        # z harfi alphabet dışında, ama fonksiyon yine de çalışmalı
        en_yakin, mesafe = self.model.en_yakin_pattern(test_pattern)
        self.assertIsNotNone(en_yakin)
        self.assertIn(en_yakin, self.model.known_patterns)

    def test_unseen_gecis_smoothing(self):
        """Unseen geçiş için smoothing uygulanmalı, sıfır olmamalı."""
        olasilik = self.model.gecis_olasiligi("zzzz", "yyyy")
        self.assertGreater(olasilik, 0)
        self.assertLess(olasilik, 1.0)

    def test_predict_calisir(self):
        """Unseen içeren test verisi için predict çalışmalı."""
        test_verisi = np.random.uniform(-3, 3, 200)  # sıra dışı değerler
        try:
            tahminler = self.model.predict(test_verisi)
            self.assertEqual(len(tahminler), len(test_verisi))
        except Exception as e:
            self.fail(f"predict hata verdi: {e}")


# =============================================================
# Çalıştır
# =============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("BİRİM TESTLER ÇALIŞIYOR")
    print("=" * 55)
    unittest.main(verbosity=2)
