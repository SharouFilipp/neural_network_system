from typing import Optional, Dict, Any, List
import numpy as np
from datetime import datetime

from .models import Model, TrainedModel, ModelSettings, ModelType, ModelStatus
from .strategies import TrainingStrategyFactory, TrainingStrategy
from .repository import ModelRepository
from .report_generator import ReportGenerator


class TrainingController:
    def __init__(self, model_repository: ModelRepository):
        self.model_repository = model_repository
        self.current_training = None
    
    def create_model_from_template(
        self, 
        template_id: str, 
        model_name: str,
        custom_settings: Optional[Dict[str, Any]] = None
    ) -> Model:
        from .repository import ModelTemplateRepository
        
        template_repo = ModelTemplateRepository()
        template = template_repo.get_template(template_id)
        
        if not template:
            raise ValueError(f"Шаблон {template_id} не найден")
        
        settings = ModelSettings(**template.default_hyperparams)
        if custom_settings:
            for key, value in custom_settings.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
        
        model = Model(
            id=str(datetime.now().timestamp()),
            name=model_name,
            model_type=template.model_type,
            architecture=template.default_architecture,
            hyperparameters=settings.dict(),
            status=ModelStatus.NOT_TRAINED
        )
        
        return model
    
    def train_model(
        self,
        model: Model,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None
    ) -> TrainedModel:
        try:
            model.status = ModelStatus.TRAINING
            
            settings = ModelSettings(**model.hyperparameters)
            strategy = TrainingStrategyFactory.create_strategy(model.model_type, settings)
            
            trained_model = strategy.train(model, train_data, train_labels, val_data, val_labels)
            
            if val_data is not None and val_labels is not None:
                evaluation_results = strategy.evaluate_model(
                    strategy.create_model(train_data.shape[1:], len(np.unique(train_labels))),
                    val_data,
                    val_labels
                )
                trained_model.accuracy = evaluation_results["accuracy"]
                trained_model.confusion_matrix = evaluation_results["confusion_matrix"]
            
            if settings.auto_save:
                self.model_repository.save_model(trained_model)
            
            return trained_model
            
        except Exception as e:
            model.status = ModelStatus.FAILED
            raise Exception(f"Ошибка обучения модели: {str(e)}")
    
    def stop_training(self) -> bool:
        if self.current_training:
            self.current_training = None
            return True
        return False


class AnalysisController:
    
    def __init__(self, model_repository: ModelRepository):
        self.model_repository = model_repository
    
    def analyze_model(
        self, 
        model_id: str, 
        test_data: np.ndarray, 
        test_labels: np.ndarray
    ) -> Dict[str, Any]:
        model = self.model_repository.get_model(model_id)
        
        if not model:
            raise ValueError(f"Модель {model_id} не найдена")
        
        if model.status != ModelStatus.TRAINED:
            raise ValueError("Модель не обучена")
        
        analysis_results = {
            "model_id": model_id,
            "model_name": model.name,
            "accuracy": model.accuracy,
            "confusion_matrix": model.confusion_matrix,
            "training_history": model.training_history,
            "metrics": model.metrics,
            "analysis_time": datetime.now().isoformat()
        }
        
        return analysis_results
    
    def compare_models(
        self, 
        model_ids: List[str], 
        test_data: np.ndarray, 
        test_labels: np.ndarray
    ) -> Dict[str, Any]:
        comparison_results = {}
        
        for model_id in model_ids:
            analysis = self.analyze_model(model_id, test_data, test_labels)
            comparison_results[model_id] = {
                "name": analysis["model_name"],
                "accuracy": analysis["accuracy"],
                "metrics": analysis["metrics"]
            }
        
        sorted_results = dict(sorted(
            comparison_results.items(),
            key=lambda x: x[1]["accuracy"] if x[1]["accuracy"] else 0,
            reverse=True
        ))
        
        return sorted_results


class ReportController:
    
    def __init__(self):
        self.report_generator = ReportGenerator()
    
    def generate_training_report(self, trained_model: TrainedModel) -> str:
        report_data = {
            "model_info": {
                "name": trained_model.name,
                "type": trained_model.model_type.value,
                "created": trained_model.created_at.isoformat(),
                "trained": trained_model.trained_at.isoformat() if trained_model.trained_at else None
            },
            "training_info": {
                "training_time": trained_model.training_time,
                "dataset_size": trained_model.dataset_size,
                "hyperparameters": trained_model.hyperparameters
            },
            "results": {
                "accuracy": trained_model.accuracy,
                "loss": trained_model.loss,
                "confusion_matrix": trained_model.confusion_matrix
            },
            "history": trained_model.training_history
        }
        
        pdf_path = self.report_generator.generate_pdf_report(report_data)
        

        csv_path = self.report_generator.generate_csv_report(report_data)
        
        return {
            "pdf_report": pdf_path,
            "csv_report": csv_path,
            "report_data": report_data
        }
    
    def generate_comparison_report(
        self, 
        comparison_results: Dict[str, Any]
    ) -> str:
        return self.report_generator.generate_comparison_report(comparison_results)