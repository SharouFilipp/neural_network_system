# 📁 utils/data_loader.py
"""
Утилиты для загрузки и обработки данных
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import pickle
import json


class DataLoader:

    
    @staticmethod
    def load_csv(filepath: str, label_column: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        df = pd.read_csv(filepath)
        
        if label_column:
            if label_column not in df.columns:
                raise ValueError(f"Column '{label_column}' not found in CSV")
            
            labels = df[label_column].values
            features = df.drop(columns=[label_column]).values
            return features.astype(np.float32), labels
        else:
            return df.values.astype(np.float32), None
    
    @staticmethod
    def load_npy(filepath: str) -> np.ndarray:

        return np.load(filepath)
    
    @staticmethod
    def load_npz(filepath: str) -> Dict[str, np.ndarray]:
 
        data = np.load(filepath)
        return {key: data[key] for key in data.files}
    
    @staticmethod
    def load_sample_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:

        from sklearn.datasets import load_digits, load_iris, load_breast_cancer
        from sklearn.model_selection import train_test_split
        
        datasets = {
            "digits": load_digits,
            "iris": load_iris,
            "breast_cancer": load_breast_cancer
        }
        
        if dataset_name not in datasets:
            raise ValueError(f"Dataset '{dataset_name}' not available. Choose from: {list(datasets.keys())}")
        
        data = datasets[dataset_name]()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        
        return X_train.astype(np.float32), y_train, X_test.astype(np.float32), y_test
    
    @staticmethod
    def preprocess_image_data(images: np.ndarray, normalize: bool = True) -> np.ndarray:

        if normalize:
            images = images.astype(np.float32) / 255.0

        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=1)
        
        return images
    
    @staticmethod
    def preprocess_tabular_data(features: np.ndarray, normalize: bool = True) -> np.ndarray:

        if normalize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        
        return features.astype(np.float32)
    
    @staticmethod
    def split_data(
        features: np.ndarray, 
        labels: np.ndarray, 
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple:
        
        from sklearn.model_selection import train_test_split
        

        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, 
            test_size=test_size, 
            random_state=random_state
        )
        
    
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class DataAugmentor:

    
    @staticmethod
    def augment_images(images: np.ndarray, augmentations: Dict[str, Any] = None) -> np.ndarray:

        if augmentations is None:
            augmentations = {
                "rotation_range": 10,
                "width_shift_range": 0.1,
                "height_shift_range": 0.1,
                "horizontal_flip": True
            }

        return images
    
    @staticmethod
    def augment_tabular_data(features: np.ndarray, method: str = "noise") -> np.ndarray:

        if method == "noise":
            noise = np.random.normal(0, 0.01, features.shape)
            return features + noise
        elif method == "smote":

            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)

            return features
        else:
            return features