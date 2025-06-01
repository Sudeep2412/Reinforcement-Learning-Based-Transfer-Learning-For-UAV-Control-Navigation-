import numpy as np
from scipy import signal
import control as ctrl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class HInfinityTransferLearning:
    def __init__(self, source_model_path=None, gamma=1.0, learning_rate=0.001, device=None):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.source_model = None
        self.target_model = None
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if self.device == 'cpu':
            print("Warning: CUDA is not available. Running on CPU may slow down computations.")
        
        if source_model_path:
            self.source_model = torch.load(source_model_path)
    
    def create_robust_controller(self, plant, uncertainty_bound):
        aug_plant = self._augment_plant_with_weights(plant, uncertainty_bound)
        controller, _, gamma_achieved = ctrl.hinfsyn(aug_plant, 1, 1)
        print(f"H∞ synthesis achieved gamma: {gamma_achieved}")
        return controller
    
    def _augment_plant_with_weights(self, plant, uncertainty_bound):
        w_uncertainty = ctrl.tf([uncertainty_bound], [1])
        w_performance = ctrl.tf([1, 0.1], [0.01, 1])
        aug_plant = ctrl.augw(plant, w_performance, w_uncertainty, None)
        return aug_plant

    def build_neural_trajectory_model(self, input_dim, hidden_layers=[64, 32]):
        class TrajectoryModel(nn.Module):
            def __init__(self, input_dim, hidden_layers, output_dim=6):
                super(TrajectoryModel, self).__init__()
                layers = []
                prev_dim = input_dim
                
                for units in hidden_layers:
                    layers.append(nn.Linear(prev_dim, units))
                    layers.append(nn.ReLU())
                    layers.append(nn.BatchNorm1d(units))
                    prev_dim = units
                
                layers.append(nn.Linear(prev_dim, output_dim))
                self.model = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)
        
        return TrajectoryModel(input_dim, hidden_layers).to(self.device)
    
    def hinf_loss(self, y_pred, y_true):
        mse_loss = torch.mean((y_true - y_pred) ** 2)
        max_error, _ = torch.max(torch.abs(y_true - y_pred), dim=1)
        hinf_term = torch.mean(max_error ** 2)
        return mse_loss + self.gamma * hinf_term
    
    def train_model(self, model, train_loader, val_loader=None, epochs=100):
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            batches = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = model(X_batch)
                loss = self.hinf_loss(y_pred, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batches += 1
            
            avg_train_loss = epoch_loss / batches
            history['train_loss'].append(avg_train_loss)
            
            if val_loader:
                model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                        val_batch_loss = self.hinf_loss(model(X_val), y_val).item()
                        val_loss += val_batch_loss
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                history['val_loss'].append(avg_val_loss)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}")
        
        return model, history
