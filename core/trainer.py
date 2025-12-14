import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
from pathlib import Path

from .models import Model, TrainedModel, ModelStatus
from .repository import ModelRepository


class NeuralNetworkTrainer:

    
    def __init__(self, model_repository: ModelRepository):
        self.model_repository = model_repository
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def create_cnn_model(self, input_shape: Tuple[int, ...], num_classes: int) -> nn.Module:

        class SimpleCNN(nn.Module):
            def __init__(self, input_channels: int = 1, num_classes: int = 10):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(32 * 7 * 7, 128)
                self.fc2 = nn.Linear(128, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.25)
                
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
    
    def create_mlp_model(self, input_shape: Tuple[int, ...], num_classes: int) -> nn.Module:
        input_size = np.prod(input_shape)
        
        class SimpleMLP(nn.Module):
            def __init__(self, input_size: int, num_classes: int):
                super(SimpleMLP, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, num_classes)
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
    
    def load_dataset(self, dataset_name: str) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
        try:
            if dataset_name == "mnist":
                from torchvision import datasets, transforms
                
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                

                train_dataset = datasets.MNIST(
                    root='./data', train=True, download=True, transform=transform
                )
                test_dataset = datasets.MNIST(
                    root='./data', train=False, download=True, transform=transform
                )
                
                train_size = min(1000, len(train_dataset))
                test_size = min(200, len(test_dataset))
                
                train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
                test_dataset = torch.utils.data.Subset(test_dataset, range(test_size))
                
 
                train_size = int(0.8 * len(train_dataset))
                val_size = len(train_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(
                    train_dataset, [train_size, val_size]
                )
                
                print(f"MNIST dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
                
            elif dataset_name == "iris":
                from sklearn.datasets import load_iris
                from sklearn.preprocessing import StandardScaler
                
                iris = load_iris()
                X = iris.data.astype(np.float32)
                y = iris.target.astype(np.int64)
                
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
   
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
   
                X_train_tensor = torch.FloatTensor(X_train)
                y_train_tensor = torch.LongTensor(y_train)
                X_val_tensor = torch.FloatTensor(X_val)
                y_val_tensor = torch.LongTensor(y_val)
                X_test_tensor = torch.FloatTensor(X_test)
                y_test_tensor = torch.LongTensor(y_test)
                
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                
                print(f"Iris dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
                
            else:

                return self.create_synthetic_data(dataset_name)
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            print(f"⚠️ Error loading dataset {dataset_name}: {e}")
            return self.create_synthetic_data(dataset_name)
    
    def create_synthetic_data(self, dataset_name: str) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
        print(f"Creating synthetic data for {dataset_name}")
        
        if dataset_name == "mnist":
            X_train = np.random.randn(800, 1, 28, 28).astype(np.float32) * 0.1 + 0.5
            X_val = np.random.randn(200, 1, 28, 28).astype(np.float32) * 0.1 + 0.5
            X_test = np.random.randn(200, 1, 28, 28).astype(np.float32) * 0.1 + 0.5
            y_train = np.random.randint(0, 10, 800).astype(np.int64)
            y_val = np.random.randint(0, 10, 200).astype(np.int64)
            y_test = np.random.randint(0, 10, 200).astype(np.int64)
            
        elif dataset_name == "tabular":
            X_train = np.random.randn(800, 10).astype(np.float32)
            X_val = np.random.randn(200, 10).astype(np.float32)
            X_test = np.random.randn(200, 10).astype(np.float32)
            y_train = np.random.randint(0, 3, 800).astype(np.int64)
            y_val = np.random.randint(0, 3, 200).astype(np.int64)
            y_test = np.random.randint(0, 3, 200).astype(np.int64)
            
        else:
            X_train = np.random.randn(800, 10).astype(np.float32)
            X_val = np.random.randn(200, 10).astype(np.float32)
            X_test = np.random.randn(200, 10).astype(np.float32)
            y_train = np.random.randint(0, 2, 800).astype(np.int64)
            y_val = np.random.randint(0, 2, 200).astype(np.int64)
            y_test = np.random.randint(0, 2, 200).astype(np.int64)
        

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        return train_dataset, val_dataset, test_dataset
    
    def train_model(
        self,
        model_id: str,
        model_type: str,
        dataset_name: str,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """Настоящее обучение модели"""
        print(f"\nStarting REAL training:")
        print(f"   Model: {model_type}")
        print(f"   Dataset: {dataset_name}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        start_time = time.time()
        
        try:
            model_data = self.model_repository.get_model(model_id)
            if not model_data:
                raise ValueError(f"Model {model_id} not found")

            self.model_repository.update_model_metadata(model_id, {"status": "training"})

            train_dataset, val_dataset, test_dataset = self.load_dataset(dataset_name)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            sample_data, _ = next(iter(train_loader))
            input_shape = sample_data.shape[1:]
            

            all_labels = []
            for _, labels in train_loader:
                all_labels.append(labels)
            num_classes = len(torch.unique(torch.cat(all_labels)))
            

            if model_type.upper() == "CNN":
                pytorch_model = self.create_cnn_model(input_shape, num_classes)
            elif model_type.upper() == "MLP":
                pytorch_model = self.create_mlp_model(input_shape, num_classes)
            elif model_type.upper() == "LSTM":
                pytorch_model = self.create_mlp_model(input_shape, num_classes)  
            else:
                pytorch_model = self.create_mlp_model(input_shape, num_classes)
            
            pytorch_model.to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(pytorch_model.parameters(), lr=learning_rate)
            
            history = {
                "train_loss": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_accuracy": []
            }
            
            for epoch in range(epochs):

                pytorch_model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = pytorch_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    train_total += target.size(0)
                    train_correct += predicted.eq(target).sum().item()
                    
                    if batch_idx % 10 == 0:
                        print(f"   Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
                avg_train_loss = train_loss / len(train_loader)
                train_accuracy = 100. * train_correct / train_total
                
                pytorch_model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = pytorch_model(data)
                        loss = criterion(output, target)
                        
                        val_loss += loss.item()
                        _, predicted = output.max(1)
                        val_total += target.size(0)
                        val_correct += predicted.eq(target).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100. * val_correct / val_total
                

                history["train_loss"].append(avg_train_loss)
                history["train_accuracy"].append(train_accuracy)
                history["val_loss"].append(avg_val_loss)
                history["val_accuracy"].append(val_accuracy)
                
                print(f"Epoch {epoch+1}/{epochs} completed")
                print(f"   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
                print(f"   Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
                print("-" * 50)
            
            test_loss, test_accuracy = self.evaluate_model(pytorch_model, test_loader, criterion)
            
            model_path = self.save_pytorch_model(pytorch_model, model_id, model_data.name)
            
            training_time = time.time() - start_time
            
            results = {
                "model_id": model_id,
                "model_name": model_data.name,
                "model_type": model_type,
                "dataset": dataset_name,
                "training_time": training_time,
                "epochs_completed": epochs,
                "final_train_loss": history["train_loss"][-1],
                "final_train_accuracy": history["train_accuracy"][-1],
                "final_val_loss": history["val_loss"][-1],
                "final_val_accuracy": history["val_accuracy"][-1],
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "history": history,
                "model_path": model_path,
                "status": "completed"
            }

            self.model_repository.update_model_metadata(model_id, {
                "status": "trained",
                "accuracy": test_accuracy / 100.0,
                "loss": test_loss,
                "training_time": training_time,
                "dataset_size": len(train_dataset) + len(val_dataset),
                "dataset_name": dataset_name
            })

            if model_data:
                model_data.status = "trained"
                model_data.accuracy = test_accuracy / 100.0
                model_data.loss = test_loss
                model_data.training_time = training_time
                model_data.dataset_size = len(train_dataset) + len(val_dataset)
                model_data.dataset_name = dataset_name
                model_data.training_history = history
                model_data.saved_path = model_path
                self.model_repository.save_model(model_data)
            
            print(f"\nTraining completed successfully!")
            print(f"Total time: {training_time:.2f} seconds")
            print(f"Test Accuracy: {test_accuracy:.2f}%")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Model saved to: {model_path}")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()

            self.model_repository.update_model_metadata(model_id, {"status": "failed"})
            
            return {
                "model_id": model_id,
                "status": "failed",
                "error": str(e)
            }
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, criterion) -> Tuple[float, float]:

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_accuracy = 100. * correct / total
        
        return test_loss, test_accuracy
    
    def save_pytorch_model(self, model: nn.Module, model_id: str, model_name: str) -> str:

        models_dir = Path("data/trained_models")
        models_dir.mkdir(parents=True, exist_ok=True)

        safe_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        model_filename = f"{model_id}_{safe_name}.pth"
        model_path = models_dir / model_filename
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_id': model_id,
            'model_name': model_name,
            'save_time': datetime.now().isoformat()
        }, model_path)
        
        return str(model_path)