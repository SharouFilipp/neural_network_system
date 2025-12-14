import os
from pathlib import Path
from typing import Dict, Any


class Config:

    

    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = DATA_DIR / "models"
    REPORTS_DIR = DATA_DIR / "reports"
    DATASETS_DIR = DATA_DIR / "datasets"
    LOGS_DIR = BASE_DIR / "logs"
 
    for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, DATASETS_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    

    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/nn_studio.db")

    DEFAULT_TRAINING_SETTINGS = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validation_split": 0.2,
        "early_stopping_patience": 5,
        "auto_save": True,
        "use_gpu": True
    }
 
    DEFAULT_MODEL_SETTINGS = {
        "name_prefix": "Model_",
        "default_type": "MLP",
        "auto_generate_id": True
    }
    

    REPORT_SETTINGS = {
        "default_format": "pdf",
        "include_graphs": True,
        "include_confusion_matrix": True,
        "include_hyperparameters": True
    }

    SECURITY_SETTINGS = {
        "require_authentication": False,
        "session_timeout_minutes": 60,
        "max_upload_size_mb": 100
    }

    PERFORMANCE_SETTINGS = {
        "max_concurrent_trainings": 3,
        "model_cache_size": 10,
        "enable_caching": True
    }
    
    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        return {
            "paths": {
                "base_dir": str(cls.BASE_DIR),
                "models_dir": str(cls.MODELS_DIR),
                "reports_dir": str(cls.REPORTS_DIR),
                "datasets_dir": str(cls.DATASETS_DIR),
                "logs_dir": str(cls.LOGS_DIR)
            },
            "database": {
                "url": cls.DATABASE_URL
            },
            "defaults": {
                "training": cls.DEFAULT_TRAINING_SETTINGS,
                "model": cls.DEFAULT_MODEL_SETTINGS
            },
            "reports": cls.REPORT_SETTINGS,
            "security": cls.SECURITY_SETTINGS,
            "performance": cls.PERFORMANCE_SETTINGS
        }
    
    @classmethod
    def update_setting(cls, section: str, key: str, value: Any):
        if hasattr(cls, f"{section.upper()}_SETTINGS"):
            settings = getattr(cls, f"{section.upper()}_SETTINGS")
            if key in settings:
                settings[key] = value
            else:
                raise KeyError(f"Key '{key}' not found in {section} settings")
        else:
            raise AttributeError(f"Settings section '{section}' not found")