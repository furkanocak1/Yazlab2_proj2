# =============================================================
# experiments/parameter_analysis.py - Parametre Duyarlılık Analizi
# Farklı window ve alphabet boyutlarının performansını test eder.
# =============================================================

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.data_loader import load_skab
from preprocessing.data_splitter import split_skab_kfold
from models.automata.automata import ProbabilisticAutomata
from experiments.visualizer import Visualizer
from experiments.evaluator import Evaluator

def run_parameter_analysis():
    print("=" * 55)
    print("PARAMETRE DUYARLILIK ANALİZİ - SKAB")
    print("=" * 55)
    
    skab = load_skab()
    # Hızlı analiz için sadece ilk fold'u kullanıyoruz
    foldlar = split_skab_kfold(skab, n_splits=5)
    fold = foldlar[0]
    
    # 1. WINDOW SIZE ANALİZİ (Alphabet sabit)
    sabit_alphabet = config.ALPHABET_SIZE
    window_sizes = config.WINDOW_SIZES
    f1_skorlari_window = []
    
    print(f"\n--- 1. Window Size Testleri (Alphabet={sabit_alphabet}) ---")
    for w in window_sizes:
        print(f"Test ediliyor: Window Size = {w}")
        model = ProbabilisticAutomata(window_size=w, alphabet_size=sabit_alphabet)
        model.fit(fold["X_pca_train"], fold["y_train"])
        f1 = model.calibrate_threshold(fold["X_pca_val"], fold["y_val"])
        f1_skorlari_window.append(f1)
        
    Visualizer.plot_parameter_sensitivity(
        window_sizes, f1_skorlari_window, 
        param_name="Window Size", 
        filename="window_size_sensitivity.png"
    )
    
    # 2. ALPHABET SIZE ANALİZİ (Window sabit)
    sabit_window = config.WINDOW_SIZE
    alphabet_sizes = config.ALPHABET_SIZES
    f1_skorlari_alphabet = []
    
    print(f"\n--- 2. Alphabet Size Testleri (Window={sabit_window}) ---")
    for a in alphabet_sizes:
        print(f"Test ediliyor: Alphabet Size = {a}")
        model = ProbabilisticAutomata(window_size=sabit_window, alphabet_size=a)
        model.fit(fold["X_pca_train"], fold["y_train"])
        f1 = model.calibrate_threshold(fold["X_pca_val"], fold["y_val"])
        f1_skorlari_alphabet.append(f1)
        
    Visualizer.plot_parameter_sensitivity(
        alphabet_sizes, f1_skorlari_alphabet, 
        param_name="Alphabet Size", 
        filename="alphabet_size_sensitivity.png"
    )
    
    print("\nAnaliz tamamlandı. Grafikler experiments/results klasörüne kaydedildi.")

if __name__ == "__main__":
    run_parameter_analysis()
