import numpy as np
from scipy import signal
import control as ctrl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class HInfinityTransferLearning:
    """
    Implementation of H∞-Optimization Transfer Learning for UAV Control Systems.
    This model focuses on transferring trajectory knowledge between different UAV control systems
    by leveraging robust control theory and deep learning.
    """
    
    def __init__(self, source_model_path=None, gamma=1.0, learning_rate=0.001, device=None):
        """
        Initialize the H∞-Optimization Transfer Learning model.
        
        Args:
            source_model_path: Path to pre-trained source model (if any)
            gamma: H∞ performance bound
            learning_rate: Learning rate for neural network training
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.source_model = None
        self.target_model = None
        
        # Set device for PyTorch computations
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if source_model_path:
            self.source_model = torch.load(source_model_path)
    
    def create_robust_controller(self, plant, uncertainty_bound):
        """
        Create a robust H∞ controller for the given plant model.
        
        Args:
            plant: State-space representation of the plant
            uncertainty_bound: Bound on model uncertainty
            
        Returns:
            H∞ optimal controller
        """
        # Augment plant with performance weights
        aug_plant = self._augment_plant_with_weights(plant, uncertainty_bound)
        
        # Synthesize H∞ controller
        controller, _, gamma_achieved = ctrl.hinfsyn(aug_plant, 1, 1)
        
        print(f"H∞ synthesis achieved gamma: {gamma_achieved}")
        return controller
    
    def _augment_plant_with_weights(self, plant, uncertainty_bound):
        """
        Augment plant with performance weights for H∞ synthesis.
        
        Args:
            plant: Original plant model
            uncertainty_bound: Uncertainty bound
            
        Returns:
            Augmented plant
        """
        # Create uncertainty weight
        w_uncertainty = ctrl.tf([uncertainty_bound], [1])
        
        # Create performance weight (typically low-pass)
        w_performance = ctrl.tf([1, 0.1], [0.01, 1])
        
        # Augment plant for mixed-sensitivity synthesis
        aug_plant = ctrl.augw(plant, w_performance, w_uncertainty, None)
        
        return aug_plant
    
    def build_neural_trajectory_model(self, input_dim, hidden_layers=[64, 32]):
        """
        Build neural network for trajectory prediction/generation.
        
        Args:
            input_dim: Input dimension (state dimension)
            hidden_layers: List of hidden layer sizes
            
        Returns:
            PyTorch neural network model
        """
        class TrajectoryModel(nn.Module):
            def __init__(self, input_dim, hidden_layers, output_dim=6):
                super(TrajectoryModel, self).__init__()
                layers = []
                prev_dim = input_dim
                
                # Create hidden layers
                for units in hidden_layers:
                    layers.append(nn.Linear(prev_dim, units))
                    layers.append(nn.ReLU())
                    layers.append(nn.BatchNorm1d(units))
                    prev_dim = units
                
                # Output layer for trajectory prediction (x, y, z, yaw, pitch, roll)
                layers.append(nn.Linear(prev_dim, output_dim))
                
                self.model = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)
        
        model = TrajectoryModel(input_dim, hidden_layers).to(self.device)
        return model
    
    def hinf_loss(self, y_pred, y_true):
        """
        Custom H∞-inspired loss function for robust trajectory learning.
        
        Args:
            y_pred: Predicted trajectory points
            y_true: True trajectory points
            
        Returns:
            H∞ loss value
        """
        # Standard MSE component
        mse_loss = torch.mean((y_true - y_pred) ** 2)
        
        # Add robustness term inspired by H∞ norm minimization
        # We penalize the worst-case error more heavily
        max_error, _ = torch.max(torch.abs(y_true - y_pred), dim=1)
        hinf_term = torch.mean(max_error ** 2)
        
        # Combined loss with weighting
        combined_loss = mse_loss + self.gamma * hinf_term
        
        return combined_loss
    
    def train_model(self, model, train_loader, val_loader=None, epochs=100):
        """
        Train a PyTorch model.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of training epochs
            
        Returns:
            Trained model and training history
        """
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0
            batches = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                y_pred = model(X_batch)
                loss = self.hinf_loss(y_pred, y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batches += 1
            
            avg_train_loss = epoch_loss / batches
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        X_val = X_val.to(self.device)
                        y_val = y_val.to(self.device)
                        
                        y_val_pred = model(X_val)
                        val_batch_loss = self.hinf_loss(y_val_pred, y_val).item()
                        
                        val_loss += val_batch_loss
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                history['val_loss'].append(avg_val_loss)
                
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}")
        
        return model, history
    
    def transfer_knowledge(self, source_data, target_data, epochs=100, batch_size=32):
        """
        Transfer knowledge from source to target domain.
        
        Args:
            source_data: Training data from source domain (X, y)
            target_data: Training data from target domain (X, y)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Trained target model
        """
        if self.source_model is None:
            raise ValueError("Source model must be loaded or trained first.")
        
        # Extract source and target data
        X_source, y_source = source_data
        X_target, y_target = target_data
        
        # Convert NumPy arrays to PyTorch tensors
        X_target_tensor = torch.tensor(X_target, dtype=torch.float32)
        y_target_tensor = torch.tensor(y_target, dtype=torch.float32)
        
        # Create DataLoader for target data
        target_dataset = TensorDataset(X_target_tensor, y_target_tensor)
        
        # Split into training and validation sets
        val_size = int(0.2 * len(target_dataset))
        train_size = len(target_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(target_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create target model with same structure as source
        input_dim = X_target.shape[1]
        self.target_model = self.build_neural_trajectory_model(input_dim)
        
        # Copy weights from source model to target model
        if isinstance(self.source_model, nn.Module):
            self.target_model.load_state_dict(self.source_model.state_dict())
        
        # Fine-tune on target domain
        self.target_model, history = self.train_model(
            self.target_model,
            train_loader,
            val_loader,
            epochs=epochs
        )
        
        return self.target_model
    
    def domain_adaptation_loss(self, source_features, target_features):
        """
        Implement domain adaptation to align source and target feature distributions.
        
        Args:
            source_features: Features from source domain
            target_features: Features from target domain
            
        Returns:
            Domain adaptation loss (Maximum Mean Discrepancy)
        """
        # Maximum Mean Discrepancy (MMD) loss for domain adaptation
        # Compute the Gram matrices
        xx = torch.matmul(source_features, source_features.t())
        yy = torch.matmul(target_features, target_features.t())
        xy = torch.matmul(source_features, target_features.t())
        
        n_x = source_features.size(0)
        n_y = target_features.size(0)
        
        # Compute MMD loss
        mmd = torch.sum(xx) / (n_x * n_x) + \
              torch.sum(yy) / (n_y * n_y) - \
              2 * torch.sum(xy) / (n_x * n_y)
        
        return mmd
    
    def evaluate_robustness(self, model, test_data, perturbation_scale=0.1):
        """
        Evaluate robustness of the model against perturbations.
        
        Args:
            model: Trained model to evaluate
            test_data: Test data (X, y)
            perturbation_scale: Scale of perturbations to apply
            
        Returns:
            Dictionary of robustness metrics
        """
        X_test, y_test = test_data
        
        # Convert to PyTorch tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        
        # Evaluate on baseline test data
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor)
            baseline_loss = self.hinf_loss(y_pred, y_test_tensor).item()
        
        # Apply perturbations
        X_perturbed = X_test + perturbation_scale * np.random.randn(*X_test.shape)
        X_perturbed_tensor = torch.tensor(X_perturbed, dtype=torch.float32).to(self.device)
        
        # Evaluate on perturbed test data
        with torch.no_grad():
            y_perturbed_pred = model(X_perturbed_tensor)
            perturbed_loss = self.hinf_loss(y_perturbed_pred, y_test_tensor).item()
        
        # Calculate robustness metrics
        robustness_ratio = perturbed_loss / baseline_loss if baseline_loss > 0 else float('inf')
        
        return {
            "baseline_loss": baseline_loss,
            "perturbed_loss": perturbed_loss,
            "robustness_ratio": robustness_ratio
        }
    
    def save_models(self, source_path="source_model.pt", target_path="target_model.pt"):
        """
        Save source and target models.
        
        Args:
            source_path: Path to save source model
            target_path: Path to save target model
        """
        if self.source_model:
            torch.save(self.source_model, source_path)
            print(f"Source model saved to {source_path}")
            
        if self.target_model:
            torch.save(self.target_model, target_path)
            print(f"Target model saved to {target_path}")


# Example usage for UAV trajectory transfer
def example_usage():
    # Create sample plant models for two different UAVs
    # First UAV (quadcopter)
    A1 = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ])
    B1 = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ])
    C1 = np.eye(6)
    D1 = np.zeros((6, 4))
    
    # Second UAV (fixed-wing)
    A2 = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, -0.1, 0, 0, 0, 0],  # Different dynamics
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, -0.2, 0, 0],  # Different dynamics
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, -0.3]   # Different dynamics
    ])
    B2 = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])
    C2 = np.eye(6)
    D2 = np.zeros((6, 3))
    
    # Create state-space representations
    uav1_plant = ctrl.ss(A1, B1, C1, D1)
    uav2_plant = ctrl.ss(A2, B2, C2, D2)
    
    # Initialize the transfer learning model
    h_inf_tl = HInfinityTransferLearning(gamma=1.5)
    
    # Design robust controllers
    uav1_controller = h_inf_tl.create_robust_controller(uav1_plant, 0.2)
    uav2_controller = h_inf_tl.create_robust_controller(uav2_plant, 0.3)
    
    # Create synthetic trajectory data
    # In practice, this would come from real UAV flights
    np.random.seed(42)
    n_samples = 1000
    
    # Source UAV data (easier trajectory)
    X_source = np.random.randn(n_samples, 6) * 0.5
    y_source = np.sin(X_source[:, 0].reshape(-1, 1)) * np.array([1, 0.5, 0.3, 0.1, 0.1, 0.1])
    
    # Target UAV data (limited samples)
    X_target = np.random.randn(200, 6) * 0.5
    y_target = np.sin(X_target[:, 0].reshape(-1, 1)) * np.array([0.9, 0.4, 0.25, 0.15, 0.05, 0.2])
    
    # Convert NumPy arrays to PyTorch tensors
    X_source_tensor = torch.tensor(X_source, dtype=torch.float32)
    y_source_tensor = torch.tensor(y_source, dtype=torch.float32)
    
    # Create DataLoader for source data
    source_dataset = TensorDataset(X_source_tensor, y_source_tensor)
    source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
    
    # Split into training and validation sets
    val_size = int(0.2 * len(source_dataset))
    train_size = len(source_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(source_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Build and train source model
    h_inf_tl.source_model = h_inf_tl.build_neural_trajectory_model(6)
    h_inf_tl.source_model, _ = h_inf_tl.train_model(
        h_inf_tl.source_model,
        train_loader,
        val_loader,
        epochs=20
    )
    
    # Transfer knowledge to target domain
    target_model = h_inf_tl.transfer_knowledge(
        (X_source, y_source),
        (X_target, y_target),
        epochs=10
    )
    
    # Evaluate robustness
    X_test = np.random.randn(100, 6) * 0.5
    y_test = np.sin(X_test[:, 0].reshape(-1, 1)) * np.array([0.9, 0.4, 0.25, 0.15, 0.05, 0.2])
    
    robustness_metrics = h_inf_tl.evaluate_robustness(target_model, (X_test, y_test))
    print("Robustness metrics:", robustness_metrics)
    
    # Save models
    h_inf_tl.save_models()


if __name__ == "__main__":
    example_usage()