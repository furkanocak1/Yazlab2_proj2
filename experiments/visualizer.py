# =============================================================
# experiments/visualizer.py - Görselleştirme Modülü
# Heatmap, Confusion Matrix, ve Parametre grafikleri üretir.
# =============================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Sonuçların kaydedileceği klasör
RESULTS_DIR = os.path.join(config.EXPERIMENTS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

class Visualizer:
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", filename="cm.png"):
        """Karmaşıklık matrisini (Confusion Matrix) çizer ve kaydeder."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal (0)', 'Anomali (1)'], 
                    yticklabels=['Normal (0)', 'Anomali (1)'])
        plt.title(title)
        plt.ylabel('Gerçek Etiket')
        plt.xlabel('Tahmin Edilen')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
        plt.close()

    @staticmethod
    def plot_transition_heatmap(transition_counts, title="Transition Probability Heatmap", filename="heatmap.png"):
        """Otomata geçiş olasılıklarının ısı haritasını çizer."""
        # Toplam eşsiz state sayısını bul
        states = set()
        for src, hedefler in transition_counts.items():
            states.add(src)
            for tgt in hedefler.keys():
                states.add(tgt)
                
        states = sorted(list(states))
        
        # Çok fazla state varsa sadece en sık görülenleri al (okunabilirlik için)
        if len(states) > 20:
            frekanslar = {s: 0 for s in states}
            for src, hedefler in transition_counts.items():
                for tgt, sayi in hedefler.items():
                    frekanslar[src] += sayi
                    frekanslar[tgt] += sayi
            # En sık görülen 20 state'i seç
            states = sorted(sorted(frekanslar.keys(), key=lambda k: frekanslar[k], reverse=True)[:20])
            
        n = len(states)
        matris = np.zeros((n, n))
        
        for i, src in enumerate(states):
            toplam_cikis = sum(transition_counts.get(src, {}).values())
            for j, tgt in enumerate(states):
                if toplam_cikis > 0:
                    sayi = transition_counts.get(src, {}).get(tgt, 0)
                    matris[i, j] = sayi / toplam_cikis
                    
        plt.figure(figsize=(10, 8))
        sns.heatmap(matris, cmap='YlOrRd', xticklabels=states, yticklabels=states)
        plt.title(title)
        plt.ylabel('Kaynak Durum (Source)')
        plt.xlabel('Hedef Durum (Target)')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
        plt.close()

    @staticmethod
    def plot_parameter_sensitivity(param_values, f1_scores, param_name="Window Size", filename="sensitivity.png"):
        """Parametre varyasyonlarının F1 skoruna etkisini çizgi grafiği olarak çizer."""
        plt.figure(figsize=(8, 5))
        plt.plot(param_values, f1_scores, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.title(f"{param_name} vs F1 Score")
        plt.ylabel('F1 Score')
        plt.xlabel(param_name)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
        plt.close()
