# =============================================================
# models/deep_learning/trainer.py - Derin Öğrenme Eğitimi
# PyTorch modellerini eğitmek ve early stopping uygulamak için.
# =============================================================

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class DLTrainer:
    def __init__(self, model, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = config.BATCH_SIZE
        self.patience = config.PATIENCE
        self.epochs = config.EPOCH_LIMIT
        
    def _to_tensor(self, X, y=None):
        X_t = torch.FloatTensor(X).to(self.device)
        if y is not None:
            y_t = torch.FloatTensor(y).view(-1, 1).to(self.device)
            return X_t, y_t
        return X_t

    def fit(self, X_train, y_train, X_val, y_val):
        """Modeli eğitir ve early stopping uygular."""
        # Veriyi tensöre çevir
        X_t, y_t = self._to_tensor(X_train, y_train)
        X_v, y_v = self._to_tensor(X_val, y_val)
        
        # Slayd window ile sekans oluşturma (DL modelleri için 3D input: batch, seq_len, features)
        # Eğer veri zaten 3D ise (CNN/LSTM için), dönüştürmeye gerek yok.
        # Bu projede X boyutları (N, features) şeklinde. Basitçe seq_len=1 olarak treat edebiliriz
        # veya window size kadar geçmişi verebiliriz.
        # Biz burada basitçe seq_len=1 yapıyoruz: (N, 1, features)
        if len(X_t.shape) == 2:
            X_t = X_t.unsqueeze(1)
            X_v = X_v.unsqueeze(1)
            
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"Model {self.model.__class__.__name__} eğitiliyor... (Cihaz: {self.device})")
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(dataloader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_v)
                val_loss = self.criterion(val_outputs, y_v).item()
                
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"  Epoch {epoch+1:02d}: Early stopping tetiklendi (Val Loss: {val_loss:.4f})")
                break
                
        # En iyi modeli yükle
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
    def predict(self, X_test):
        """Tahmin sonuçlarını (0 veya 1) döndürür."""
        probs = self.predict_proba(X_test)
        return (probs > 0.5).astype(int)
        
    def predict_proba(self, X_test):
        """Olasılıkları döndürür."""
        self.model.eval()
        X_t = self._to_tensor(X_test)
        if len(X_t.shape) == 2:
            X_t = X_t.unsqueeze(1)
            
        with torch.no_grad():
            outputs = self.model(X_t)
            
        return outputs.cpu().numpy().flatten()
