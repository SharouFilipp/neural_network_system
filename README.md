# neural_network_studio

Профессиональная система для обучения и управления нейронными сетями, разработанная на Python с использованием PyTorch. Система предоставляет полноценную среду для создания, обучения, анализа и экспорта моделей машинного обучения с веб-интерфейсом.

---

## 🚀 Быстрый старт

### Требования
- Python 3.13+
- PyTorch 2.0+

### Запуск
```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Запуск сервера
python run.py

# 3. Открыть в браузере
http://localhost:8000
```

---

## 🏗 Архитектура и Реализация

Проект построен на основе сервисно-ориентированной архитектуры с четким разделением на слои:
1.  **Domain Layer (Data Models):** Определяет сущности бизнес-логики.
2.  **Service Layer:** Реализует бизнес-правила и сценарии использования.
3.  **Presentation Layer:** Веб-интерфейс на Flask.

Ниже представлены ключевые фрагменты исходного кода, демонстрирующие точное соответствие спроектированной объектной модели (UML).

### 1. Иерархия Пользователей (Models & Training)

Система поддерживает различные типы нейросетевых архитектур через паттерн `Strategy`, позволяя гибко настраивать процесс обучения:

```python
# models.py
class Model(BaseModel):
    id: str
    name: str
    model_type: Union[ModelType, str]
    created_at: datetime = Field(default_factory=datetime.now)
    status: Union[ModelStatus, str] = ModelStatus.NOT_TRAINED
    accuracy: Optional[float] = None
    training_history: Dict[str, List[float]] = {}

class TrainedModel(Model):
    trained_at: datetime = Field(default_factory=datetime.now)
    training_time: Optional[float] = None
    dataset_size: Optional[int] = None
```

### 2. Паттерн Strategy для обучения

Реализация различных алгоритмов обучения через абстрактный класс `TrainingStrategy` и конкретные реализации для каждого типа модели:

```python
# strategies.py
class TrainingStrategy(ABC):
    @abstractmethod
    def create_model(self, input_shape: Tuple[int, ...], num_classes: int) -> nn.Module:
        pass
    
    @abstractmethod
    def train(self, model: Model, train_data: np.ndarray, train_labels: np.ndarray) -> TrainedModel:
        pass

class CNNTrainingStrategy(TrainingStrategy):
    """Стратегия обучения сверточных сетей"""
    def create_model(self, input_shape: Tuple[int, ...], num_classes: int) -> nn.Module:
        return SimpleCNN(input_shape[0], num_classes)
    
    def train(self, model: Model, train_data: np.ndarray, train_labels: np.ndarray) -> TrainedModel:
        # Реальная реализация обучения CNN
        pass

class TrainingStrategyFactory:
    """Фабрика для создания стратегий обучения"""
    @staticmethod
    def create_strategy(model_type: ModelType, settings: ModelSettings) -> TrainingStrategy:
        strategies = {
            ModelType.CNN: CNNTrainingStrategy,
            ModelType.MLP: MLPTrainingStrategy,
            ModelType.LSTM: LSTMTrainingStrategy,
        }
        return strategies[model_type](settings)
```

### 3. Полноценное обучение с PyTorch

Система выполняет настоящее обучение нейросетей с отслеживанием прогресса и метрик:

```python
# trainer.py
class NeuralNetworkTrainer:
    def train_model(self, model_id: str, model_type: str, dataset_name: str, 
                    epochs: int = 5, batch_size: int = 32) -> Dict[str, Any]:
        
        # Загрузка датасета
        train_dataset, val_dataset, test_dataset = self.load_dataset(dataset_name)
        
        # Создание модели PyTorch
        if model_type.upper() == "CNN":
            pytorch_model = self.create_cnn_model(input_shape, num_classes)
        elif model_type.upper() == "MLP":
            pytorch_model = self.create_mlp_model(input_shape, num_classes)
        
        # Обучение с валидацией
        history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
        
        for epoch in range(epochs):
            # Прямое и обратное распространение
            optimizer.zero_grad()
            output = pytorch_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Валидация
            val_loss, val_accuracy = self.validate_model(pytorch_model, val_loader, criterion)
            
            history["train_loss"].append(train_loss)
            history["val_accuracy"].append(val_accuracy)
```

### 4. Команды и отмена действий (Command Pattern)

Реализация паттерна Command для управления операциями обучения и анализа:

```python
# commands.py
class Command(ABC):
    @abstractmethod
    def execute(self) -> Any:
        pass
    
    @abstractmethod
    def undo(self) -> bool:
        pass

class TrainModelCommand(Command):
    """Команда обучения модели с возможностью отмены"""
    def execute(self) -> TrainedModel:
        self.trained_model = self.controller.train_model(...)
        return self.trained_model
    
    def undo(self) -> bool:
        self.model.status = self.original_status
        return True

class CommandInvoker:
    """Исполнитель команд с историей операций"""
    def __init__(self):
        self.history = []
        self.redo_stack = []
    
    def execute_command(self, command: Command) -> Any:
        result = command.execute()
        self.history.append(command)
        return result
```

### 5.  Генерация отчетов в различных форматах

Система автоматически создает подробные отчеты о результатах обучения:

```python
# report_generator.py
class ReportGenerator:
    def generate_pdf_report(self, report_data: Dict[str, Any]) -> str:
        """Создание PDF отчета с графиками и метриками"""
        doc = SimpleDocTemplate(...)
        # Добавление информации о модели
        # Добавление графиков обучения
        # Добавление результатов
        doc.build(elements)
        return filepath
    
    def generate_csv_report(self, report_data: Dict[str, Any]) -> str:
        """Экспорт данных в CSV для дальнейшего анализа"""
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)
        return filepath
```

### 6. Аутентификация и управление пользователями

Простая но эффективная система аутентификации с JWT токенами:

```python
# auth_simple.py
class AuthRepository:
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        user = self.get_user(username)
        if not user:
            return None
        
        hashed_password = self._hash_password(password)
        if hashed_password != user.hashed_password:
            return None
        
        return user
    
    def create_user(self, user_create: UserCreate) -> Optional[UserInDB]:
        """Создание нового пользователя"""
        user_id = f"user_{int(datetime.now().timestamp())}"
        hashed_password = self._hash_password(user_create.password)
        
        new_user = UserInDB(
            id=user_id,
            username=user_create.username,
            email=user_create.email,
            hashed_password=hashed_password
        )
        return new_user
```
---

## 🖥️ Интерфейс системы

### 1. Выбор роли
Экран входа, позволяющий выбрать режим работы: Трейдер, Аналитик или Управляющий.

![role_select.png](screenshots/role_select.png)


### 2. 📊 Панель управления (Dashboard)
Главный экран с котировками в реальном времени, графиком с ML-предсказаниями, торговыми сигналами и оценкой рисков.

![img_1.png](screenshots/img_1.png)
![img_2.png](screenshots/img_2.png)

### 3. 📈 Аналитика (Analytics)
Детальный анализ с возможностью выбора конкретной математической модели (ARIMA, Holt-Winters, MA) и просмотра доверительных интервалов.

![img_3.png](screenshots/img_3.png)

### 4. 📋 Отчёты (Reports)
Генерация PDF и CSV отчётов по портфелю и стратегиям.

![img_4.png](screenshots/img_4.png)

### 5. 💼 Портфель (Portfolio)
Инструменты управления активами: добавление/удаление позиций, ребалансировка и отслеживание P&L.

![img_5.png](screenshots/img_5.png)

---
