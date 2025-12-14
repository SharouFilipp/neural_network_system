import sys
import os
from pathlib import Path

def main():
    try:
        import torch
        print(f"PyTorch {torch.__version__} installed")
    except ImportError:
        print("PyTorch not installed!")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    BASE_DIR = Path(__file__).parent
    directories = [
        BASE_DIR / "web" / "templates",
        BASE_DIR / "web" / "static",
        BASE_DIR / "data",
        BASE_DIR / "data" / "trained_models",
        BASE_DIR / "data" / "models",
        BASE_DIR / "data" / "auth"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")
    
    print("\n Starting Neural Network Studio...")
    print("\n Features:")
    print(" User Authentication")
    print(" REAL Neural Network Training")
    print("  Model Download & Export")
    print("Training Progress Tracking")
    
    print("\n Default Login:")
    print("   Username: admin")
    print("   Password: admin123")
    
    print("\n Access:")
    print("   Web Interface: http://localhost:8000")
    print("   API Docs:      http://localhost:8000/api/docs")
    

    
 
    os.system("uvicorn api.main_simple:app --reload --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()