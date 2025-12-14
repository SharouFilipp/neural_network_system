from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import numpy as np

from .models import Model, TrainedModel
from .controllers import TrainingController, AnalysisController


class Command(ABC):
    @abstractmethod
    def execute(self) -> Any:
        pass
    
    @abstractmethod
    def undo(self) -> bool:
        pass


class TrainModelCommand(Command):
    
    def __init__(
        self,
        controller: TrainingController,
        model: Model,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None
    ):
        self.controller = controller
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.trained_model: Optional[TrainedModel] = None
        self.original_status = model.status
    
    def execute(self) -> TrainedModel:
        self.trained_model = self.controller.train_model(
            self.model,
            self.train_data,
            self.train_labels,
            self.val_data,
            self.val_labels
        )
        return self.trained_model
    
    def undo(self) -> bool:
        if self.trained_model:
            self.model.status = self.original_status
            return True
        return False


class SaveModelCommand(Command):
    
    def __init__(self, model_repository, model: TrainedModel):
        self.model_repository = model_repository
        self.model = model
        self.saved_model_id: Optional[str] = None
    
    def execute(self) -> str:
        self.saved_model_id = self.model_repository.save_model(self.model)
        return self.saved_model_id
    
    def undo(self) -> bool:
        if self.saved_model_id:
            return self.model_repository.delete_model(self.saved_model_id)
        return False


class AnalyzeModelCommand(Command):

    def __init__(
        self,
        controller: AnalysisController,
        model_id: str,
        test_data: np.ndarray,
        test_labels: np.ndarray
    ):
        self.controller = controller
        self.model_id = model_id
        self.test_data = test_data
        self.test_labels = test_labels
        self.analysis_results: Optional[Dict] = None
    
    def execute(self) -> Dict:
        self.analysis_results = self.controller.analyze_model(
            self.model_id,
            self.test_data,
            self.test_labels
        )
        return self.analysis_results
    
    def undo(self) -> bool:
        self.analysis_results = None
        return True


class CommandInvoker:

    def __init__(self):
        self.history = []
        self.redo_stack = []
    
    def execute_command(self, command: Command) -> Any:
        result = command.execute()
        self.history.append(command)
        self.redo_stack.clear()
        return result
    
    def undo(self) -> bool:
        if not self.history:
            return False
        
        command = self.history.pop()
        success = command.undo()
        
        if success:
            self.redo_stack.append(command)
        
        return success
    
    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        
        command = self.redo_stack.pop()
        result = command.execute()
        self.history.append(command)
        return True
    
    def clear_history(self):
        self.history.clear()
        self.redo_stack.clear()