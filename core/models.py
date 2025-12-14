# 📁 core/models.py (исправленная версия)
"""
Основные классы модели с исправлениями
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
import torch
import torch.nn as nn


class ModelType(str, Enum):
    CNN = "CNN"
    MLP = "MLP"
    LSTM = "LSTM"
    TRANSFORMER = "Transformer"
    CUSTOM = "Custom"


class ModelStatus(str, Enum):
    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


class ModelTemplate(BaseModel):
    id: str
    name: str
    model_type: ModelType
    description: str
    default_architecture: Dict[str, Any]
    recommended_datasets: List[str]
    default_hyperparams: Dict[str, Any]
    
    model_config = ConfigDict(from_attributes=True)


class Model(BaseModel):
    id: str
    name: str
    model_type: Union[ModelType, str]
    created_at: datetime = Field(default_factory=datetime.now)
    architecture: Dict[str, Any] = {}
    hyperparameters: Dict[str, Any] = {}
    status: Union[ModelStatus, str] = ModelStatus.NOT_TRAINED
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    metrics: Dict[str, float] = {}
    confusion_matrix: Optional[List[List[int]]] = None
    training_history: Dict[str, List[float]] = {}
    saved_path: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class TrainedModel(Model):
    trained_at: datetime = Field(default_factory=datetime.now)
    training_time: Optional[float] = None
    dataset_size: Optional[int] = None
    dataset_name: Optional[str] = None
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        # Здесь будет реализация предсказания
        pass
    
    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        # Здесь будет реализация оценки
        pass


class ModelSettings(BaseModel):
    epochs: int = Field(default=10, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=1, le=1024)
    learning_rate: float = Field(default=0.001, ge=1e-5, le=1.0)
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5)
    auto_save: bool = True
    save_path: str = "models/"
    use_gpu: bool = True
    seed: Optional[int] = None
    
    model_config = ConfigDict(from_attributes=True)