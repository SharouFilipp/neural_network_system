import json
import pickle
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
from pathlib import Path

from .models import Model, TrainedModel, ModelTemplate, ModelStatus


class ModelRepository:

    def __init__(self, base_path: str = "data/models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_path / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"models": {}, "next_id": 1}
    
    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def save_model(self, model: TrainedModel) -> str:
        model_id = str(self.metadata["next_id"])
        self.metadata["next_id"] += 1
        
        model_path = self.base_path / f"{model_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model.model_dump(), f)
        
        self.metadata["models"][model_id] = {
            "id": model_id,
            "name": model.name,
            "model_type": model.model_type.value if hasattr(model.model_type, 'value') else str(model.model_type),
            "created_at": model.created_at.isoformat(),
            "trained_at": model.trained_at.isoformat() if model.trained_at else None,
            "accuracy": model.accuracy,
            "status": model.status.value if hasattr(model.status, 'value') else str(model.status),
            "file_path": str(model_path)
        }
        
        self._save_metadata()
        return model_id
    
    def get_model(self, model_id: str) -> Optional[TrainedModel]:
        if model_id not in self.metadata["models"]:
            return None
        
        model_info = self.metadata["models"][model_id]
        model_path = Path(model_info["file_path"])
        
        if not model_path.exists():
            return None
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        if 'model_type' in model_data:
            model_data['model_type'] = model_data['model_type']
        if 'status' in model_data:
            model_data['status'] = model_data['status']
        
        return TrainedModel(**model_data)
    
    def get_user_models(self, user_id: str) -> List[Model]:
        user_models = []
        for model_id, model_info in self.metadata["models"].items():
            model = Model(
                id=model_id,
                name=model_info["name"],
                model_type=model_info["model_type"],
                created_at=datetime.fromisoformat(model_info["created_at"]),
                architecture={},
                hyperparameters={},
                status=model_info["status"],
                accuracy=model_info.get("accuracy")
            )
            user_models.append(model)
        
        return sorted(user_models, key=lambda x: x.created_at, reverse=True)
    
    def delete_model(self, model_id: str) -> bool:
        if model_id not in self.metadata["models"]:
            return False
        
        model_info = self.metadata["models"][model_id]
        model_path = Path(model_info["file_path"])
        
        if model_path.exists():
            model_path.unlink()
        
        del self.metadata["models"][model_id]
        self._save_metadata()
        
        return True
    
    def update_model_metadata(self, model_id: str, updates: Dict[str, Any]) -> bool:
        if model_id not in self.metadata["models"]:
            return False
        
        self.metadata["models"][model_id].update(updates)
        self._save_metadata()
        return True