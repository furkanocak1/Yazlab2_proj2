# =============================================================
# experiments/evaluator.py - Değerlendirme ve İstatistiksel Testler
# ROC, Confusion Matrix hesaplamaları ve McNemar/Wilcoxon testleri.
# =============================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, f1_score
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Evaluator:
    @staticmethod
    def hesapla_metrikler(y_true, y_pred, y_prob=None):
        """Temel sınıflandırma metriklerini hesaplar."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Sadece 0 ve 1 içeren temiz bir veri olduğundan emin ol
        if len(y_true) == 0:
            return {}
            
        metrikler = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrikler["roc_auc"] = roc_auc_score(y_true, y_prob)
            except:
                pass
                
        return metrikler

    @staticmethod
    def istatistiksel_test_mcnemar(y_true, y_pred_model1, y_pred_model2):
        """
        McNemar Testi: İki sınıflandırıcının (örn: LSTM ve Otomata) hata oranlarının
        birbirinden istatistiksel olarak anlamlı ölçüde farklı olup olmadığını test eder.
        
        H0: İki modelin performansı aynıdır.
        H1: Modellerden biri anlamlı derecede daha iyidir.
        
        Döndürür: p-value
        Eğer p < 0.05 ise modeller arası anlamlı fark var demektir.
        """
        
        # Contingency table (Çapraz Tablo) oluştur
        # a: İkisi de doğru bildi
        # b: Model 1 doğru, Model 2 yanlış
        # c: Model 1 yanlış, Model 2 doğru
        # d: İkisi de yanlış bildi
        
        m1_dogru = (y_true == y_pred_model1)
        m2_dogru = (y_true == y_pred_model2)
        
        a = sum(m1_dogru & m2_dogru)
        b = sum(m1_dogru & ~m2_dogru)
        c = sum(~m1_dogru & m2_dogru)
        d = sum(~m1_dogru & ~m2_dogru)
        
        tablo = [[a, b], [c, d]]
        
        # Exact=False (Ki-kare yaklaşımı büyük veriler için daha hızlıdır)
        sonuc = mcnemar(tablo, exact=False, correction=True)
        return sonuc.pvalue

    @staticmethod
    def istatistiksel_test_wilcoxon(skorlar_model1, skorlar_model2):
        """
        Wilcoxon İşaretli Sıralama Testi (F1 skorları gibi sürekli değerler için).
        Örneğin 5 farklı seed için elde edilen F1 skorlarını karşılaştırır.
        """
        if len(skorlar_model1) != len(skorlar_model2) or len(skorlar_model1) < 3:
            return 1.0 # Yeterli veri yok
            
        fark = np.array(skorlar_model1) - np.array(skorlar_model2)
        if np.all(fark == 0):
            return 1.0 # Birebir aynı sonuçlar
            
        sonuc = wilcoxon(skorlar_model1, skorlar_model2)
        return sonuc.pvalue
