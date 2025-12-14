from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import uvicorn
import asyncio
from datetime import datetime
from pathlib import Path
import json
import zipfile
import io

from core.models import Model, TrainedModel, ModelType
from core.repository import ModelRepository
from core.trainer import NeuralNetworkTrainer
from core.auth_simple import (
    AuthRepository, User, UserCreate, UserLogin,
    authenticate_user, create_simple_jwt, get_current_user_from_token,
    ACCESS_TOKEN_EXPIRE_DAYS
)

app = FastAPI(
    title="Neural Network Studio",
    description="Neural network training system with authentication",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "web" / "templates"
STATIC_DIR = BASE_DIR / "web" / "static"
REPORTS_DIR = BASE_DIR / "data" / "reports"


for directory in [TEMPLATES_DIR, STATIC_DIR, REPORTS_DIR, 
                  BASE_DIR / "data" / "trained_models",
                  BASE_DIR / "data" / "exports",
                  BASE_DIR / "data" / "auth"]:
    directory.mkdir(parents=True, exist_ok=True)


templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


auth_repository = AuthRepository()
model_repository = ModelRepository()
trainer = NeuralNetworkTrainer(model_repository)


training_tasks: Dict[str, Dict[str, Any]] = {}


class CreateModelRequest(BaseModel):
    name: str
    model_type: str
    hyperparameters: Optional[Dict[str, Any]] = None

class TrainModelRequest(BaseModel):
    model_id: str
    dataset: str = "mnist"
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001



async def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[User]:
    if not authorization:
        return None
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return None
        
        user = get_current_user_from_token(token, auth_repository)
        return user
    except:
        return None


async def get_current_active_user(current_user: Optional[User] = Depends(get_current_user)) -> User:
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Studio</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            padding: 12px 30px;
            border-radius: 50px;
            font-weight: 600;
        }
        .card {
            border-radius: 15px;
            border: none;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .progress {
            height: 25px;
            border-radius: 12px;
        }
        .login-container {
            max-width: 400px;
            margin: 100px auto;
        }
    </style>
</head>
<body>
    <div id="app">
        <!-- Контент будет загружен через JavaScript -->
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <script>
        // Глобальные переменные
        let currentUser = null;
        let authToken = localStorage.getItem('authToken');
        let currentPage = 'login';
        
        // Инициализация
        document.addEventListener('DOMContentLoaded', function() {
            if (authToken) {
                checkAuth();
            } else {
                showLoginPage();
            }
        });
        
        // Проверка аутентификации
        async function checkAuth() {
            try {
                const response = await axios.get('/api/auth/me', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                currentUser = response.data;
                showDashboard();
            } catch (error) {
                localStorage.removeItem('authToken');
                authToken = null;
                showLoginPage();
            }
        }
        
        // Показать страницу логина
        function showLoginPage() {
            document.getElementById('app').innerHTML = `
                <div class="login-container">
                    <div class="glass-card p-5 text-center">
                        <h1 class="mb-4">
                            <i class="fas fa-brain text-primary"></i><br>
                            Neural Network Studio
                        </h1>
                        
                        <div class="mb-3">
                            <input type="text" class="form-control form-control-lg" 
                                   id="username" placeholder="Username" value="admin">
                        </div>
                        <div class="mb-3">
                            <input type="password" class="form-control form-control-lg" 
                                   id="password" placeholder="Password" value="admin123">
                        </div>
                        
                        <button class="btn btn-primary btn-lg w-100 mb-3" onclick="login()">
                            <i class="fas fa-sign-in-alt me-2"></i>Sign In
                        </button>
                        
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle me-2"></i>
                            Demo: admin / admin123
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Вход в систему
        async function login() {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            try {
                const response = await axios.post('/api/auth/login', {
                    username: username,
                    password: password
                });
                
                authToken = response.data.access_token;
                localStorage.setItem('authToken', authToken);
                
                currentUser = await getUserInfo();
                showDashboard();
                showAlert('Welcome back!', 'success');
                
            } catch (error) {
                showAlert('Invalid credentials', 'danger');
            }
        }
        
        // Получение информации о пользователе
        async function getUserInfo() {
            try {
                const response = await axios.get('/api/auth/me', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                return response.data;
            } catch (error) {
                return null;
            }
        }
        
        // Показать дашборд
        async function showDashboard() {
            try {
                const [modelsResponse, jobsResponse] = await Promise.all([
                    axios.get('/api/models', { headers: getAuthHeaders() }),
                    axios.get('/api/training/jobs', { headers: getAuthHeaders() })
                ]);
                
                const models = modelsResponse.data;
                const jobs = jobsResponse.data;
                
                const totalModels = models.length;
                const trainedModels = models.filter(m => m.status === 'trained').length;
                
                document.getElementById('app').innerHTML = `
                    <!-- Навигация -->
                    <nav class="navbar navbar-expand-lg navbar-dark" style="background: rgba(0,0,0,0.2);">
                        <div class="container">
                            <a class="navbar-brand fw-bold" href="#" onclick="showDashboard()">
                                <i class="fas fa-brain me-2"></i>Neural Studio
                            </a>
                            <div class="navbar-nav ms-auto">
                                <div class="nav-item dropdown">
                                    <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown">
                                        <i class="fas fa-user me-1"></i>${currentUser.username}
                                    </a>
                                    <ul class="dropdown-menu">
                                        <li><a class="dropdown-item" href="#" onclick="showDashboard()"><i class="fas fa-home me-2"></i>Dashboard</a></li>
                                        <li><a class="dropdown-item" href="#" onclick="showModelsPage()"><i class="fas fa-project-diagram me-2"></i>My Models</a></li>
                                        <li><a class="dropdown-item" href="#" onclick="showTrainingPage()"><i class="fas fa-cogs me-2"></i>Training</a></li>
                                        <li><hr class="dropdown-divider"></li>
                                        <li><a class="dropdown-item" href="#" onclick="logout()"><i class="fas fa-sign-out-alt me-2"></i>Logout</a></li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </nav>
                    
                    <!-- Основной контент -->
                    <div class="container py-5">
                        <div class="row mb-5">
                            <div class="col-md-3">
                                <div class="card text-center p-4 bg-primary text-white">
                                    <h3>${totalModels}</h3>
                                    <p>Total Models</p>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card text-center p-4 bg-success text-white">
                                    <h3>${trainedModels}</h3>
                                    <p>Trained</p>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card text-center p-4 bg-warning text-white">
                                    <h3>${Object.keys(jobs).length}</h3>
                                    <p>Active Jobs</p>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card text-center p-4 bg-info text-white">
                                    <h3>${currentUser.models_count || 0}</h3>
                                    <p>My Models</p>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6">
                                <div class="glass-card p-4 mb-4">
                                    <h4 class="mb-3">Quick Actions</h4>
                                    <button class="btn btn-primary w-100 mb-2" onclick="showCreateModelModal()">
                                        <i class="fas fa-plus me-2"></i>Create New Model
                                    </button>
                                    <button class="btn btn-outline-primary w-100" onclick="showTrainingPage()">
                                        <i class="fas fa-cogs me-2"></i>Start Training
                                    </button>
                                </div>
                                
                                <div class="glass-card p-4">
                                    <h4 class="mb-3">Recent Models</h4>
                                    <div id="recentModels">
                                        ${await loadRecentModelsHtml()}
                                    </div>
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="glass-card p-4">
                                    <h4 class="mb-3">Training Progress</h4>
                                    <div id="trainingProgress">
                                        ${await loadTrainingProgressHtml()}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                // Загрузить модальное окно создания модели
                loadCreateModelModal();
                
            } catch (error) {
                console.error('Error loading dashboard:', error);
                showAlert('Error loading dashboard', 'danger');
            }
        }
        
        // Загрузка недавних моделей
        async function loadRecentModelsHtml() {
            try {
                const response = await axios.get('/api/models', { 
                    headers: getAuthHeaders(),
                    params: { limit: 3 }
                });
                
                const models = response.data;
                
                if (models.length === 0) {
                    return '<p class="text-muted">No models yet</p>';
                }
                
                let html = '';
                models.forEach(model => {
                    const accuracy = model.accuracy ? (model.accuracy * 100).toFixed(1) + '%' : 'N/A';
                    
                    html += `
                        <div class="mb-2 p-2 border rounded">
                            <div class="d-flex justify-content-between">
                                <div>
                                    <strong>${model.name}</strong>
                                    <div class="small text-muted">${model.model_type} • ${accuracy}</div>
                                </div>
                                <button class="btn btn-sm btn-outline-primary" onclick="viewModel('${model.id}')">
                                    <i class="fas fa-eye"></i>
                                </button>
                            </div>
                        </div>
                    `;
                });
                
                return html;
                
            } catch (error) {
                return '<p class="text-muted">Error loading models</p>';
            }
        }
        
        // Загрузка прогресса обучения
        async function loadTrainingProgressHtml() {
            try {
                const response = await axios.get('/api/training/jobs', { 
                    headers: getAuthHeaders() 
                });
                
                const jobs = response.data;
                
                if (Object.keys(jobs).length === 0) {
                    return '<p class="text-muted">No active training</p>';
                }
                
                let html = '';
                Object.values(jobs).forEach(job => {
                    html += `
                        <div class="mb-3">
                            <div class="d-flex justify-content-between mb-1">
                                <strong>${job.model_name}</strong>
                                <span class="badge bg-${job.status === 'running' ? 'warning' : 'success'}">
                                    ${job.status}
                                </span>
                            </div>
                            <div class="progress">
                                <div class="progress-bar" style="width: ${job.progress || 0}%"></div>
                            </div>
                            <small class="text-muted">${job.dataset} • Epoch ${job.current_epoch || 0}/${job.epochs}</small>
                        </div>
                    `;
                });
                
                return html;
                
            } catch (error) {
                return '<p class="text-muted">Error loading progress</p>';
            }
        }
        
        // Показать страницу моделей
        async function showModelsPage() {
            try {
                const response = await axios.get('/api/models', { headers: getAuthHeaders() });
                const models = response.data;
                
                let modelsHtml = '';
                if (models.length === 0) {
                    modelsHtml = '<p class="text-center text-muted py-5">No models created yet</p>';
                } else {
                    models.forEach(model => {
                        const accuracy = model.accuracy ? (model.accuracy * 100).toFixed(1) + '%' : 'Not trained';
                        const date = new Date(model.created_at).toLocaleDateString();
                        
                        modelsHtml += `
                            <div class="col-md-4 mb-3">
                                <div class="card h-100">
                                    <div class="card-body">
                                        <h5 class="card-title">${model.name}</h5>
                                        <p class="card-text">
                                            <span class="badge bg-info">${model.model_type}</span>
                                            <span class="badge bg-${model.status === 'trained' ? 'success' : 'secondary'} ms-2">
                                                ${model.status}
                                            </span>
                                        </p>
                                        <p class="card-text">
                                            <small class="text-muted">Accuracy: ${accuracy}</small><br>
                                            <small class="text-muted">Created: ${date}</small>
                                        </p>
                                        <div class="btn-group w-100">
                                            <button class="btn btn-sm btn-outline-primary" onclick="viewModelDetails('${model.id}')">
                                                <i class="fas fa-eye"></i>
                                            </button>
                                            ${model.status === 'trained' ? `
                                                <button class="btn btn-sm btn-outline-success" onclick="downloadModel('${model.id}')">
                                                    <i class="fas fa-download"></i>
                                                </button>
                                            ` : ''}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                }
                
                document.getElementById('app').innerHTML = `
                    <nav class="navbar navbar-expand-lg navbar-dark" style="background: rgba(0,0,0,0.2);">
                        <div class="container">
                            <a class="navbar-brand fw-bold" href="#" onclick="showDashboard()">
                                <i class="fas fa-brain me-2"></i>Neural Studio
                            </a>
                            <div class="navbar-nav ms-auto">
                                <a class="nav-link" href="#" onclick="showDashboard()">Dashboard</a>
                                <a class="nav-link active" href="#" onclick="showModelsPage()">Models</a>
                                <a class="nav-link" href="#" onclick="showTrainingPage()">Training</a>
                                <a class="nav-link" href="#" onclick="logout()">Logout</a>
                            </div>
                        </div>
                    </nav>
                    
                    <div class="container py-5">
                        <div class="d-flex justify-content-between align-items-center mb-4">
                            <h1>My Models</h1>
                            <button class="btn btn-primary" onclick="showCreateModelModal()">
                                <i class="fas fa-plus me-2"></i>New Model
                            </button>
                        </div>
                        
                        <div class="row" id="modelsList">
                            ${modelsHtml}
                        </div>
                    </div>
                `;
                
                loadCreateModelModal();
                
            } catch (error) {
                console.error('Error loading models:', error);
                showAlert('Error loading models', 'danger');
            }
        }
        
        // Показать страницу обучения
        async function showTrainingPage() {
            document.getElementById('app').innerHTML = `
                <nav class="navbar navbar-expand-lg navbar-dark" style="background: rgba(0,0,0,0.2);">
                    <div class="container">
                        <a class="navbar-brand fw-bold" href="#" onclick="showDashboard()">
                            <i class="fas fa-brain me-2"></i>Neural Studio
                        </a>
                        <div class="navbar-nav ms-auto">
                            <a class="nav-link" href="#" onclick="showDashboard()">Dashboard</a>
                            <a class="nav-link" href="#" onclick="showModelsPage()">Models</a>
                            <a class="nav-link active" href="#" onclick="showTrainingPage()">Training</a>
                            <a class="nav-link" href="#" onclick="logout()">Logout</a>
                        </div>
                    </div>
                </nav>
                
                <div class="container py-5">
                    <h1 class="mb-4">Train New Model</h1>
                    
                    <div class="glass-card p-4">
                        <form id="trainingForm">
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label class="form-label">Model Name</label>
                                    <input type="text" class="form-control" id="modelName" value="My Neural Network" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Model Type</label>
                                    <select class="form-select" id="modelType" required>
                                        <option value="CNN">CNN (Image Classification)</option>
                                        <option value="MLP">MLP (Tabular Data)</option>
                                    </select>
                                </div>
                            </div>
                            
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label class="form-label">Dataset</label>
                                    <select class="form-select" id="dataset" required>
                                        <option value="mnist">MNIST (Handwritten Digits)</option>
                                        <option value="iris">Iris (Flower Classification)</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Training Parameters</label>
                                    <div class="input-group">
                                        <input type="number" class="form-control" id="epochs" value="5" placeholder="Epochs">
                                        <input type="number" class="form-control" id="batchSize" value="32" placeholder="Batch">
                                        <input type="number" class="form-control" id="learningRate" value="0.001" step="0.0001" placeholder="LR">
                                    </div>
                                </div>
                            </div>
                            
                            <button type="button" class="btn btn-primary btn-lg w-100" onclick="startTraining()">
                                <i class="fas fa-play me-2"></i>Start Training
                            </button>
                        </form>
                    </div>
                    
                    <div class="mt-4" id="activeTraining"></div>
                </div>
            `;
            
            // Загрузить активные тренировки
            loadActiveTraining();
        }
        
        // Загрузить активные тренировки
        async function loadActiveTraining() {
            try {
                const response = await axios.get('/api/training/jobs', { headers: getAuthHeaders() });
                const jobs = response.data;
                
                const container = document.getElementById('activeTraining');
                if (Object.keys(jobs).length === 0) {
                    container.innerHTML = '<p class="text-muted">No active training jobs</p>';
                    return;
                }
                
                let html = '<h3 class="mb-3">Active Training Jobs</h3>';
                Object.values(jobs).forEach(job => {
                    html += `
                        <div class="card mb-3">
                            <div class="card-body">
                                <h5>${job.model_name}</h5>
                                <div class="progress mb-2">
                                    <div class="progress-bar" style="width: ${job.progress || 0}%">
                                        ${job.progress || 0}%
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-6">
                                        <small>Status: <span class="badge bg-${job.status === 'running' ? 'warning' : 'success'}">${job.status}</span></small>
                                    </div>
                                    <div class="col-6">
                                        <small>Epoch: ${job.current_epoch || 0}/${job.epochs}</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                container.innerHTML = html;
                
            } catch (error) {
                console.error('Error loading training jobs:', error);
            }
        }
        
        // Начать обучение
        async function startTraining() {
            const modelName = document.getElementById('modelName').value;
            const modelType = document.getElementById('modelType').value;
            const dataset = document.getElementById('dataset').value;
            const epochs = parseInt(document.getElementById('epochs').value) || 5;
            const batchSize = parseInt(document.getElementById('batchSize').value) || 32;
            const learningRate = parseFloat(document.getElementById('learningRate').value) || 0.001;
            
            try {
                // Создать модель
                const createResponse = await axios.post('/api/models', {
                    name: modelName,
                    model_type: modelType,
                    hyperparameters: {
                        epochs: epochs,
                        batch_size: batchSize,
                        learning_rate: learningRate
                    }
                }, { headers: getAuthHeaders() });
                
                const modelId = createResponse.data.model_id;
                
                // Начать обучение
                const trainResponse = await axios.post('/api/train', {
                    model_id: modelId,
                    dataset: dataset,
                    epochs: epochs,
                    batch_size: batchSize,
                    learning_rate: learningRate
                }, { headers: getAuthHeaders() });
                
                showAlert('Training started successfully!', 'success');
                
                // Обновить страницу
                setTimeout(() => {
                    loadActiveTraining();
                }, 1000);
                
            } catch (error) {
                console.error('Error starting training:', error);
                showAlert('Error starting training', 'danger');
            }
        }
        
        // Посмотреть детали модели
        async function viewModelDetails(modelId) {
            try {
                const response = await axios.get(`/api/models/${modelId}`, { headers: getAuthHeaders() });
                const model = response.data;
                
                let analysisHtml = '';
                if (model.status === 'trained') {
                    analysisHtml = `
                        <div class="row mt-3">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Model Performance</h5>
                                        <p><strong>Accuracy:</strong> ${(model.accuracy * 100).toFixed(2)}%</p>
                                        <p><strong>Loss:</strong> ${model.loss ? model.loss.toFixed(4) : 'N/A'}</p>
                                        <p><strong>Training Time:</strong> ${model.training_time ? model.training_time.toFixed(2) + 's' : 'N/A'}</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Actions</h5>
                                        <button class="btn btn-primary w-100 mb-2" onclick="downloadModel('${modelId}')">
                                            <i class="fas fa-download me-2"></i>Download Model
                                        </button>
                                        <button class="btn btn-success w-100" onclick="exportModel('${modelId}')">
                                            <i class="fas fa-file-export me-2"></i>Export with Report
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }
                
                // Показать модальное окно
                const modal = new bootstrap.Modal(document.getElementById('modelDetailsModal'));
                document.getElementById('modelDetailsContent').innerHTML = `
                    <h4>${model.name}</h4>
                    <p><strong>Type:</strong> ${model.model_type}</p>
                    <p><strong>Status:</strong> <span class="badge bg-${model.status === 'trained' ? 'success' : 'secondary'}">${model.status}</span></p>
                    <p><strong>Created:</strong> ${new Date(model.created_at).toLocaleString()}</p>
                    ${analysisHtml}
                `;
                modal.show();
                
            } catch (error) {
                console.error('Error loading model details:', error);
                showAlert('Error loading model details', 'danger');
            }
        }
        
        // Скачать модель
        async function downloadModel(modelId) {
            try {
                const response = await axios.get(`/api/models/${modelId}/download`, {
                    headers: getAuthHeaders(),
                    responseType: 'blob'
                });
                
                // Создать ссылку для скачивания
                const url = window.URL.createObjectURL(new Blob([response.data]));
                const link = document.createElement('a');
                link.href = url;
                link.setAttribute('download', `model_${modelId}.pth`);
                document.body.appendChild(link);
                link.click();
                link.remove();
                
                showAlert('Model downloaded successfully!', 'success');
                
            } catch (error) {
                console.error('Error downloading model:', error);
                showAlert('Error downloading model', 'danger');
            }
        }
        
        // Экспортировать модель
        async function exportModel(modelId) {
            try {
                const response = await axios.get(`/api/models/${modelId}/export`, {
                    headers: getAuthHeaders(),
                    responseType: 'blob'
                });
                
                const url = window.URL.createObjectURL(new Blob([response.data]));
                const link = document.createElement('a');
                link.href = url;
                link.setAttribute('download', `model_export_${modelId}.zip`);
                document.body.appendChild(link);
                link.click();
                link.remove();
                
                showAlert('Model exported successfully!', 'success');
                
            } catch (error) {
                console.error('Error exporting model:', error);
                showAlert('Error exporting model', 'danger');
            }
        }
        
        // Загрузить модальное окно создания модели
        async function loadCreateModelModal() {
            const modalHtml = `
                <div class="modal fade" id="createModelModal" tabindex="-1">
                    <div class="modal-dialog">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title">Create New Model</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                            </div>
                            <div class="modal-body">
                                <form id="createModelForm">
                                    <div class="mb-3">
                                        <label class="form-label">Model Name</label>
                                        <input type="text" class="form-control" id="newModelName" required>
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">Model Type</label>
                                        <select class="form-select" id="newModelType" required>
                                            <option value="CNN">CNN - Image Classification</option>
                                            <option value="MLP">MLP - Tabular Data</option>
                                        </select>
                                    </div>
                                </form>
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                                <button type="button" class="btn btn-primary" onclick="createModel()">Create Model</button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="modal fade" id="modelDetailsModal" tabindex="-1">
                    <div class="modal-dialog modal-lg">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title">Model Details</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                            </div>
                            <div class="modal-body" id="modelDetailsContent">
                                <!-- Контент будет загружен здесь -->
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            if (!document.getElementById('createModelModal')) {
                document.body.insertAdjacentHTML('beforeend', modalHtml);
            }
        }
        
        // Показать модальное окно создания модели
        function showCreateModelModal() {
            const modal = new bootstrap.Modal(document.getElementById('createModelModal'));
            modal.show();
        }
        
        // Создать модель
        async function createModel() {
            const name = document.getElementById('newModelName').value;
            const modelType = document.getElementById('newModelType').value;
            
            try {
                await axios.post('/api/models', {
                    name: name,
                    model_type: modelType
                }, { headers: getAuthHeaders() });
                
                const modal = bootstrap.Modal.getInstance(document.getElementById('createModelModal'));
                modal.hide();
                
                showAlert('Model created successfully!', 'success');
                
                // Обновить страницу
                if (currentPage === 'dashboard') {
                    showDashboard();
                } else if (currentPage === 'models') {
                    showModelsPage();
                }
                
            } catch (error) {
                console.error('Error creating model:', error);
                showAlert('Error creating model', 'danger');
            }
        }
        
        // Выход из системы
        function logout() {
            localStorage.removeItem('authToken');
            authToken = null;
            currentUser = null;
            showLoginPage();
            showAlert('Logged out successfully', 'info');
        }
        
        // Вспомогательные функции
        function getAuthHeaders() {
            return {
                'Authorization': `Bearer ${authToken}`
            };
        }
        
        function showAlert(message, type = 'info') {
            const alert = document.createElement('div');
            alert.className = `alert alert-${type} alert-dismissible fade show`;
            alert.innerHTML = `
                ${message}
                <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
            `;
            
            Object.assign(alert.style, {
                position: 'fixed',
                top: '20px',
                right: '20px',
                zIndex: '9999',
                minWidth: '300px'
            });
            
            document.body.appendChild(alert);
            
            setTimeout(() => {
                if (alert.parentElement) {
                    alert.remove();
                }
            }, 5000);
        }
    </script>
</body>
</html>"""

(TEMPLATES_DIR / "index.html").write_text(HTML_TEMPLATE, encoding='utf-8')



@app.post("/api/auth/register", response_model=Dict[str, Any])
async def register(user_create: UserCreate):
    """Регистрация нового пользователя"""
    user = auth_repository.create_user(user_create)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    access_token = create_simple_jwt(user.id, user.username)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username
    }


@app.post("/api/auth/login", response_model=Dict[str, Any])
async def login(user_login: UserLogin):
    """Вход в систему"""
    user = authenticate_user(auth_repository, user_login.username, user_login.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = create_simple_jwt(user.id, user.username)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username
    }


@app.get("/api/auth/me", response_model=Dict[str, Any])
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    return current_user.model_dump()


@app.get("/api/models", response_model=List[Dict[str, Any]])
async def get_user_models(
    current_user: User = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100
):
    models = model_repository.get_user_models(current_user.id)
    models = models[skip:skip + limit]
    return [model.model_dump() for model in models]


@app.get("/api/models/{model_id}", response_model=Dict[str, Any])
async def get_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    model = model_repository.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model.model_dump()


@app.post("/api/models", response_model=Dict[str, Any])
async def create_model(
    request: CreateModelRequest,
    current_user: User = Depends(get_current_active_user)
):
    try:
        model = Model(
            id=f"model_{current_user.id}_{int(datetime.now().timestamp())}",
            name=request.name,
            model_type=request.model_type,
            hyperparameters=request.hyperparameters or {
                "epochs": 5,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            status="not_trained",
            created_at=datetime.now()
        )
        
        model_id = model_repository.save_model(TrainedModel(**model.model_dump()))
        
        auth_repository.update_user_models_count(current_user.id, 1)
        
        return {
            "model_id": model_id,
            "message": "Model created successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/train")
async def train_model(
    request: TrainModelRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):

    try:
        model = model_repository.get_model(request.model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        task_id = f"task_{current_user.id}_{int(datetime.now().timestamp())}"
        training_tasks[task_id] = {
            "task_id": task_id,
            "user_id": current_user.id,
            "model_id": request.model_id,
            "model_name": model.name,
            "model_type": model.model_type,
            "dataset": request.dataset,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "status": "pending",
            "start_time": datetime.now().isoformat(),
            "progress": 0,
            "current_epoch": 0
        }
        

        background_tasks.add_task(
            run_training,
            task_id,
            request.model_id,
            str(model.model_type),
            request.dataset,
            request.epochs,
            request.batch_size,
            request.learning_rate
        )
        
        return {
            "task_id": task_id,
            "message": "Training started successfully",
            "model_id": request.model_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def run_training(task_id: str, model_id: str, model_type: str, 
                      dataset: str, epochs: int, batch_size: int, learning_rate: float):
    try:
        if task_id in training_tasks:
            training_tasks[task_id].update({
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "progress": 0,
                "current_epoch": 0,
                "total_epochs": epochs
            })
        
    
        results = trainer.train_model(
            model_id=model_id,
            model_type=model_type,
            dataset_name=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
    
        if task_id in training_tasks:
            training_tasks[task_id].update({
                "status": "completed",
                "results": results,
                "end_time": datetime.now().isoformat(),
                "progress": 100,
                "current_epoch": epochs
            })
        
        print(f"✅ Training task {task_id} completed successfully")
        
    except Exception as e:
        print(f"❌ Training task {task_id} failed: {e}")
        if task_id in training_tasks:
            training_tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "end_time": datetime.now().isoformat()
            })


@app.get("/api/training/jobs")
async def get_training_jobs(
    current_user: User = Depends(get_current_active_user)
):
    user_jobs = {
        task_id: task for task_id, task in training_tasks.items()
        if task.get("user_id") == current_user.id
    }
    

    for task_id, task in user_jobs.items():
        if task["status"] == "running":
            elapsed = (datetime.now() - datetime.fromisoformat(task["start_time"])).total_seconds()
            total_estimated_time = task["epochs"] * 3
            
            progress = min(99, (elapsed / total_estimated_time) * 100)
            current_epoch = min(task["epochs"], int((elapsed / total_estimated_time) * task["epochs"]))
            
            user_jobs[task_id].update({
                "progress": round(progress, 1),
                "current_epoch": current_epoch,
                "elapsed_time": elapsed
            })
    
    return user_jobs


@app.get("/api/models/{model_id}/download")
async def download_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Скачать модель"""
    model = model_repository.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model.saved_path:
        raise HTTPException(status_code=404, detail="Model weights not found")
    
    from pathlib import Path
    model_path = Path(model.saved_path)
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        model_path,
        media_type='application/octet-stream',
        filename=f"{model.name.replace(' ', '_')}.pth"
    )


@app.get("/api/models/{model_id}/export")
async def export_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Экспортировать модель со всеми данными"""
    model = model_repository.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:

        model_info = model.model_dump()
        model_info_str = json.dumps(model_info, indent=2, default=str)
        zip_file.writestr("model_info.json", model_info_str)
        

        if model.saved_path and Path(model.saved_path).exists():
            zip_file.write(model.saved_path, "model_weights.pth")
        

        readme_content = f"""# Neural Network Model Export

Model: {model.name}
Type: {model.model_type}
ID: {model_id}
Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Contents:
- model_info.json: Model metadata and training history
- model_weights.pth: PyTorch model weights

## How to use:
1. Load weights in PyTorch: torch.load('model_weights.pth')
2. View training history in model_info.json

Generated by Neural Network Studio
"""
        zip_file.writestr("README.md", readme_content)
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type='application/zip',
        headers={
            'Content-Disposition': f'attachment; filename="{model.name}_export.zip"'
        }
    )



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("index.html", {"request": request})



@app.on_event("startup")
async def startup_event():
    """Действия при запуске приложения"""
    print("=" * 70)
    print("🧠 Neural Network Studio - Simple & Working Version")
    print("=" * 70)
    

    import torch
    print(f"\n📊 System Information:")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Training Device: {trainer.device}")
    
    print(f"\n✅ System ready!")
    print(f"🌐 Web Interface: http://localhost:8000")
    print(f"📚 API Documentation: http://localhost:8000/api/docs")
    print("\n🔐 Login with: admin / admin123")
    print("🚀 Features: Authentication, Real Training, Download, Export")
    print("=" * 70)


if __name__ == "__main__":
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )