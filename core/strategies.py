# 📁 core/strategies.py
"""
Стратегии обучения для разных типов моделей
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime
import time

from .models import Model, TrainedModel, ModelSettings, ModelType


class TrainingStrategy(ABC):  
    def __init__(self, model_settings: ModelSettings):
        self.settings = model_settings
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and model_settings.use_gpu else "cpu"
        )
    
    @abstractmethod
    def create_model(self, input_shape: Tuple[int, ...], num_classes: int) -> nn.Module:
        pass
    
    @abstractmethod
    def train(
        self,
        model: Model,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None
    ) -> TrainedModel:
        pass
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_data: np.ndarray,
        test_labels: np.ndarray
    ) -> Dict[str, Any]:
        model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(test_data).to(self.device)
            predictions = model(test_tensor)
            pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
            
            accuracy = accuracy_score(test_labels, pred_classes)
            cm = confusion_matrix(test_labels, pred_classes)
            
            return {
                "accuracy": accuracy,
                "confusion_matrix": cm.tolist(),
                "predictions": pred_classes.tolist()
            }


class CNNTrainingStrategy(TrainingStrategy):
    
    def create_model(self, input_shape: Tuple[int, ...], num_classes: int) -> nn.Module:
        class SimpleCNN(nn.Module):
            def __init__(self, input_channels: int, num_classes: int):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(64 * (input_shape[1]//4) * (input_shape[2]//4), 128)
                self.fc2 = nn.Linear(128, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.5)
                
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        input_channels = input_shape[0] if len(input_shape) == 3 else 1
        return SimpleCNN(input_channels, num_classes)
    
    def train(self, model: Model, train_data: np.ndarray, train_labels: np.ndarray, 
              val_data: Optional[np.ndarray] = None, val_labels: Optional[np.ndarray] = None) -> TrainedModel:
        start_time = time.time()
    
        if len(train_data.shape) == 3:  
            train_data = np.expand_dims(train_data, axis=1)
        
        pytorch_model = self.create_model(train_data.shape[1:], len(np.unique(train_labels)))
        pytorch_model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(pytorch_model.parameters(), lr=self.settings.learning_rate)
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(train_data),
            torch.LongTensor(train_labels)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.settings.batch_size, 
            shuffle=True
        )
        
        history = {"loss": [], "accuracy": []}
        for epoch in range(self.settings.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = pytorch_model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            epoch_loss /= len(train_loader)
            epoch_acc = correct / total
            
            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_acc)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{self.settings.epochs}, "
                      f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        trained_model = TrainedModel(
            **model.dict(),
            status="trained",
            trained_at=datetime.now(),
            training_time=time.time() - start_time,
            dataset_size=len(train_data),
            training_history=history
        )
        
        if self.settings.auto_save:
            self._save_model(pytorch_model, trained_model)
        
        return trained_model
    
    def _save_model(self, pytorch_model: nn.Module, trained_model: TrainedModel):
        """Сохранение модели PyTorch"""
        save_path = f"{self.settings.save_path}/{trained_model.id}.pth"
        torch.save({
            'model_state_dict': pytorch_model.state_dict(),
            'model_info': trained_model.dict()
        }, save_path)
        trained_model.saved_path = save_path


class MLPTrainingStrategy(TrainingStrategy):
    
    def create_model(self, input_shape: Tuple[int, ...], num_classes: int) -> nn.Module:
        input_size = np.prod(input_shape)
        
        class SimpleMLP(nn.Module):
            def __init__(self, input_size: int, num_classes: int):
                super(SimpleMLP, self).__init__()
                self.fc1 = nn.Linear(input_size, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                x = x.view(x.size(0), -1)  
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        return SimpleMLP(input_size, num_classes)
    
    def train(self, model: Model, train_data: np.ndarray, train_labels: np.ndarray, 
              val_data: Optional[np.ndarray] = None, val_labels: Optional[np.ndarray] = None) -> TrainedModel:

        pass


class LSTMTrainingStrategy(TrainingStrategy):
    def create_model(self, input_shape: Tuple[int, ...], num_classes: int) -> nn.Module:
        class SimpleLSTM(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
                super(SimpleLSTM, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, num_classes)
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out
        
        return SimpleLSTM(input_shape[1], 128, 2, num_classes)
    
    def train(self, model: Model, train_data: np.ndarray, train_labels: np.ndarray, 
              val_data: Optional[np.ndarray] = None, val_labels: Optional[np.ndarray] = None) -> TrainedModel:
        pass


class TrainingStrategyFactory:    
    @staticmethod
    def create_strategy(model_type: ModelType, settings: ModelSettings) -> TrainingStrategy:
        strategies = {
            ModelType.CNN: CNNTrainingStrategy,
            ModelType.MLP: MLPTrainingStrategy,
            ModelType.LSTM: LSTMTrainingStrategy,
        }
        
        strategy_class = strategies.get(model_type)
        if not strategy_class:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
        
        return strategy_class(settings)