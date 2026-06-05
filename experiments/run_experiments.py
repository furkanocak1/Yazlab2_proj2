# =============================================================
# experiments/run_experiments.py - Ana Deney Çalıştırıcı
# Modelleri (Automata ve LSTM) orijinal, gürültülü ve unseen 
# veri üzerinde 5 farklı random_seed ile test eder.
# =============================================================

import os
import sys
import numpy as np
import warnings

# Gürültülü uyarıları kapat (PyTorch/Sklearn)
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.data_loader import load_batadal, load_skab
from preprocessing.data_splitter import split_batadal, split_skab_kfold
from preprocessing.noise import gaussian_gurultu_ekle, unseen_veri_olustur
from models.automata.automata import ProbabilisticAutomata
from models.deep_learning.lstm_model import LSTMModel
from models.deep_learning.gru_model import GRUModel
from models.deep_learning.cnn_model import CNNModel
from models.deep_learning.trainer import DLTrainer
from experiments.evaluator import Evaluator
from experiments.visualizer import Visualizer

def run_experiment_suite(veri_adi, X_train, y_train, X_val, y_val, X_test, y_test, 
                         X_pca_train, X_pca_val, X_pca_test, is_skab=False):
    """
    Belirli bir veri seti (fold veya BATADAL) üzerinde deneyleri koşturur.
    """
    
    seeds = config.RANDOM_SEEDS
    
    sonuclar = {
        "Automata": {"Original": [], "Noise": [], "Unseen": []},
        "LSTM": {"Original": [], "Noise": [], "Unseen": []},
        "GRU": {"Original": [], "Noise": [], "Unseen": []},
        "CNN": {"Original": [], "Noise": [], "Unseen": []}
    }
    
    # Tüm tahminleri saklayacağız (McNemar için)
    # Sadece ilk seed'in tahminlerini alsak yeterli olur karşılaştırma için.
    tahmin_kayitlari = {"Automata": None, "LSTM": None, "GRU": None, "CNN": None, "y_true": y_test}
    
    # Giriş boyutu (Feature sayısı)
    input_size = X_train.shape[1]
    
    print(f"\n[{veri_adi}] Deneyleri Başlıyor ({len(seeds)} farklı seed ile)...")
    
    for idx, seed in enumerate(seeds):
        print(f"\n--- Seed {idx+1}/{len(seeds)}: {seed} ---")
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        
        # ---------------------------------------------------------
        # 1. OTOMATA MODELİ
        # ---------------------------------------------------------
        print(">> Automata eğitiliyor...")
        model_automata = ProbabilisticAutomata(window_size=config.WINDOW_SIZE, alphabet_size=config.ALPHABET_SIZE)
        model_automata.fit(X_pca_train, y_train)
        model_automata.calibrate_threshold(X_pca_val, y_val)
        
        # Orijinal Veri
        tahmin_auto_orig = model_automata.predict(X_pca_test)
        f1_auto_orig = Evaluator.hesapla_metrikler(y_test, tahmin_auto_orig).get('f1', 0)
        sonuclar["Automata"]["Original"].append(f1_auto_orig)
        
        # Gürültülü Veri
        pca_test_noise = gaussian_gurultu_ekle(X_pca_test, seed=seed)
        tahmin_auto_noise = model_automata.predict(pca_test_noise)
        sonuclar["Automata"]["Noise"].append(Evaluator.hesapla_metrikler(y_test, tahmin_auto_noise).get('f1', 0))
        
        # Unseen Veri (Sadece Automata için geçerli, çünkü PCA üzerinde çalışıyor ve sözlüğü var)
        pca_test_unseen, _ = unseen_veri_olustur(X_pca_train, X_pca_test, model_automata, seed=seed)
        tahmin_auto_unseen = model_automata.predict(pca_test_unseen)
        sonuclar["Automata"]["Unseen"].append(Evaluator.hesapla_metrikler(y_test, tahmin_auto_unseen).get('f1', 0))
        
        # Sadece ilk seed için Confusion Matrix kaydet ve Heatmap çiz
        if idx == 0:
            tahmin_kayitlari["Automata"] = tahmin_auto_orig
            Visualizer.plot_confusion_matrix(y_test, tahmin_auto_orig, 
                                            title=f"{veri_adi} - Automata", 
                                            filename=f"{veri_adi}_cm_automata.png")
            Visualizer.plot_transition_heatmap(model_automata.transition_counts,
                                              title=f"{veri_adi} - Automata Transitions",
                                              filename=f"{veri_adi}_heatmap_automata.png")
                                              
        # ---------------------------------------------------------
        # 2. LSTM MODELİ (Derin Öğrenme)
        # ---------------------------------------------------------
        # Uyarı: Hızlı test için num_layers ve hidden_size küçük tutuldu.
        print(">> LSTM eğitiliyor...")
        model_lstm = LSTMModel(input_size=input_size, hidden_size=32, num_layers=1)
        trainer = DLTrainer(model_lstm, lr=0.005)
        trainer.fit(X_train, y_train, X_val, y_val)
        
        # Orijinal Veri
        tahmin_lstm_orig = trainer.predict(X_test)
        f1_lstm_orig = Evaluator.hesapla_metrikler(y_test, tahmin_lstm_orig).get('f1', 0)
        sonuclar["LSTM"]["Original"].append(f1_lstm_orig)
        
        # Gürültülü Veri
        X_test_noise = gaussian_gurultu_ekle(X_test, seed=seed)
        tahmin_lstm_noise = trainer.predict(X_test_noise)
        sonuclar["LSTM"]["Noise"].append(Evaluator.hesapla_metrikler(y_test, tahmin_lstm_noise).get('f1', 0))
        
        # Unseen Veri
        # LSTM sözlük kullanmaz, onun yerine büyük varyanslı gürültü verilir.
        X_test_unseen = X_test + np.random.normal(0, 2.0, X_test.shape)
        tahmin_lstm_unseen = trainer.predict(X_test_unseen)
        sonuclar["LSTM"]["Unseen"].append(Evaluator.hesapla_metrikler(y_test, tahmin_lstm_unseen).get('f1', 0))
        
        if idx == 0:
            tahmin_kayitlari["LSTM"] = tahmin_lstm_orig
            Visualizer.plot_confusion_matrix(y_test, tahmin_lstm_orig, 
                                            title=f"{veri_adi} - LSTM", 
                                            filename=f"{veri_adi}_cm_lstm.png")
                                            
        # ---------------------------------------------------------
        # 3. GRU MODELİ (Derin Öğrenme)
        # ---------------------------------------------------------
        print(">> GRU eğitiliyor...")
        model_gru = GRUModel(input_size=input_size, hidden_size=32, num_layers=1)
        trainer_gru = DLTrainer(model_gru, lr=0.005)
        trainer_gru.fit(X_train, y_train, X_val, y_val)
        
        # Orijinal Veri
        tahmin_gru_orig = trainer_gru.predict(X_test)
        f1_gru_orig = Evaluator.hesapla_metrikler(y_test, tahmin_gru_orig).get('f1', 0)
        sonuclar["GRU"]["Original"].append(f1_gru_orig)
        
        # Gürültülü Veri
        tahmin_gru_noise = trainer_gru.predict(X_test_noise)
        sonuclar["GRU"]["Noise"].append(Evaluator.hesapla_metrikler(y_test, tahmin_gru_noise).get('f1', 0))
        
        # Unseen Veri
        tahmin_gru_unseen = trainer_gru.predict(X_test_unseen)
        sonuclar["GRU"]["Unseen"].append(Evaluator.hesapla_metrikler(y_test, tahmin_gru_unseen).get('f1', 0))
        
        if idx == 0:
            tahmin_kayitlari["GRU"] = tahmin_gru_orig
            Visualizer.plot_confusion_matrix(y_test, tahmin_gru_orig, 
                                            title=f"{veri_adi} - GRU", 
                                            filename=f"{veri_adi}_cm_gru.png")

        # ---------------------------------------------------------
        # 4. CNN MODELİ (Derin Öğrenme)
        # ---------------------------------------------------------
        print(">> CNN eğitiliyor...")
        model_cnn = CNNModel(input_size=input_size)
        trainer_cnn = DLTrainer(model_cnn, lr=0.005)
        trainer_cnn.fit(X_train, y_train, X_val, y_val)
        
        # Orijinal Veri
        tahmin_cnn_orig = trainer_cnn.predict(X_test)
        f1_cnn_orig = Evaluator.hesapla_metrikler(y_test, tahmin_cnn_orig).get('f1', 0)
        sonuclar["CNN"]["Original"].append(f1_cnn_orig)
        
        # Gürültülü Veri
        tahmin_cnn_noise = trainer_cnn.predict(X_test_noise)
        sonuclar["CNN"]["Noise"].append(Evaluator.hesapla_metrikler(y_test, tahmin_cnn_noise).get('f1', 0))
        
        # Unseen Veri
        tahmin_cnn_unseen = trainer_cnn.predict(X_test_unseen)
        sonuclar["CNN"]["Unseen"].append(Evaluator.hesapla_metrikler(y_test, tahmin_cnn_unseen).get('f1', 0))
        
        if idx == 0:
            tahmin_kayitlari["CNN"] = tahmin_cnn_orig
            Visualizer.plot_confusion_matrix(y_test, tahmin_cnn_orig, 
                                            title=f"{veri_adi} - CNN", 
                                            filename=f"{veri_adi}_cm_cnn.png")
                                            
    # =========================================================
    # İSTATİSTİKSEL KARŞILAŞTIRMA (İlk seed üzerinden)
    # =========================================================
    p_value = Evaluator.istatistiksel_test_mcnemar(
        tahmin_kayitlari["y_true"], 
        tahmin_kayitlari["LSTM"], 
        tahmin_kayitlari["Automata"]
    )
    
    anlamli_fark = "VAR" if p_value < 0.05 else "YOK"
    
    print("\n" + "="*50)
    print(f"[{veri_adi}] DENEY SONUÇLARI (5 Seed Ortalaması)")
    print("="*50)
    for model_name in ["Automata", "LSTM", "GRU", "CNN"]:
        print(f"\n{model_name} Modeli (F1 Skorları):")
        for senaryo in ["Original", "Noise", "Unseen"]:
            skorlar = sonuclar[model_name][senaryo]
            ort = np.mean(skorlar)
            std = np.std(skorlar)
            print(f"  - {senaryo:8s} : {ort:.4f} ± {std:.4f}")
            
    print(f"\n[McNemar Testi] LSTM vs Automata P-Value: {p_value:.4e} -> Anlamlı fark: {anlamli_fark}")
    print("="*50)
    

if __name__ == "__main__":
    # Sonuçların temiz görünmesi için uyarıları kapatıyoruz
    import warnings
    warnings.filterwarnings('ignore')
    
    print("=" * 60)
    print("FROM BLACK-BOX TO EXPLAINABILITY - ANA DENEY SÜRECİ")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # 1. BATADAL Deneyleri
    # ---------------------------------------------------------
    batadal = load_batadal()
    b_split = split_batadal(batadal)
    
    run_experiment_suite(
        veri_adi="BATADAL",
        X_train=b_split[0], y_train=b_split[6],
        X_val=b_split[1],   y_val=b_split[7],
        X_test=b_split[2],  y_test=b_split[8],
        X_pca_train=b_split[3], X_pca_val=b_split[4], X_pca_test=b_split[5]
    )
    
    # ---------------------------------------------------------
    # 2. SKAB Deneyleri
    # ---------------------------------------------------------
    # Not: SKAB için 5 fold var. Süreden tasarruf etmek adına
    # sadece ilk fold üzerinde deney yapıyoruz. Tam sonuçlar için
    # döngüye alınabilir.
    skab = load_skab()
    foldlar = split_skab_kfold(skab, n_splits=5)
    f1 = foldlar[0]
    
    run_experiment_suite(
        veri_adi="SKAB_Fold1",
        X_train=f1["X_train"], y_train=f1["y_train"],
        X_val=f1["X_val"],     y_val=f1["y_val"],
        X_test=f1["X_test"],   y_test=f1["y_test"],
        X_pca_train=f1["X_pca_train"], X_pca_val=f1["X_pca_val"], X_pca_test=f1["X_pca_test"]
    )
    
    print("\nTüm deneyler başarıyla tamamlandı!")
    print("Grafikler (Confusion Matrix, Heatmap) 'experiments/results' klasörüne kaydedildi.")
