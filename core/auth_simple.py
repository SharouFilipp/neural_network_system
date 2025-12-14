import hashlib
import hmac
import json
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from pathlib import Path

SECRET_KEY = "neural-network-studio-secret-key-2024"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7


class User(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    created_at: str
    models_count: int = 0


class UserInDB(User):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    username: str


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class AuthRepository:
    def __init__(self, file_path: str = "data/auth/users.json"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.users = self._load_users()
    
    def _load_users(self) -> Dict[str, UserInDB]:
        """Загрузка пользователей из файла"""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    users = {}
                    for user_id, user_data in data.items():
                        users[user_id] = UserInDB(**user_data)
                    return users
            except:
                pass

        admin_user = self._create_default_admin()
        return {admin_user.id: admin_user}
    
    def _save_users(self):
        data = {}
        for user_id, user in self.users.items():
            data[user_id] = user.model_dump()
        
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _create_default_admin(self) -> UserInDB:
        admin_id = "admin_001"
        hashed_password = self._hash_password("admin123")
        
        admin_user = UserInDB(
            id=admin_id,
            username="admin",
            email="admin@neuralstudio.com",
            full_name="Administrator",
            is_active=True,
            created_at=datetime.now().isoformat(),
            hashed_password=hashed_password,
            models_count=0
        )
        
        return admin_user
    
    def _hash_password(self, password: str) -> str:
        salt = "neural-studio-salt-2024"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        return self.users.get(user_id)
    
    def create_user(self, user_create: UserCreate) -> Optional[UserInDB]:
        if self.get_user(user_create.username):
            return None
        
        user_id = f"user_{int(datetime.now().timestamp())}"
        hashed_password = self._hash_password(user_create.password)
        
        new_user = UserInDB(
            id=user_id,
            username=user_create.username,
            email=user_create.email,
            full_name=user_create.full_name,
            hashed_password=hashed_password,
            is_active=True,
            created_at=datetime.now().isoformat(),
            models_count=0
        )
        
        self.users[user_id] = new_user
        self._save_users()
        
        return new_user
    
    def update_user_models_count(self, user_id: str, delta: int = 1):
        if user_id in self.users:
            self.users[user_id].models_count += delta
            self._save_users()
    
    def get_all_users(self) -> List[User]:
        return [User(**user.model_dump(exclude={'hashed_password'})) 
                for user in self.users.values()]


def create_simple_jwt(user_id: str, username: str) -> str:

    header = {
        "alg": "HS256",
        "typ": "JWT"
    }
    
    payload = {
        "sub": user_id,
        "username": username,
        "iat": datetime.utcnow().timestamp(),
        "exp": (datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)).timestamp()
    }
    

    header_encoded = base64.urlsafe_b64encode(
        json.dumps(header).encode()
    ).decode().rstrip("=")
    
    payload_encoded = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).decode().rstrip("=")
    
    message = f"{header_encoded}.{payload_encoded}"
    signature = hmac.new(
        SECRET_KEY.encode(),
        message.encode(),
        hashlib.sha256
    ).digest()
    
    signature_encoded = base64.urlsafe_b64encode(
        signature
    ).decode().rstrip("=")
    
    return f"{header_encoded}.{payload_encoded}.{signature_encoded}"


def verify_simple_jwt(token: str) -> Optional[Dict[str, Any]]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        
        header_encoded, payload_encoded, signature_encoded = parts
        
        message = f"{header_encoded}.{payload_encoded}"
        expected_signature = hmac.new(
            SECRET_KEY.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        
        expected_signature_encoded = base64.urlsafe_b64encode(
            expected_signature
        ).decode().rstrip("=")
        
        if signature_encoded != expected_signature_encoded:
            return None
        
        payload_json = base64.urlsafe_b64decode(payload_encoded + "=" * (-len(payload_encoded) % 4))
        payload = json.loads(payload_json)
        

        if payload["exp"] < datetime.utcnow().timestamp():
            return None
        
        return payload
        
    except:
        return None


def authenticate_user(auth_repo: AuthRepository, username: str, password: str) -> Optional[UserInDB]:
    user = auth_repo.get_user(username)
    if not user:
        return None

    hashed_password = auth_repo._hash_password(password)
    if hashed_password != user.hashed_password:
        return None
    
    return user


def get_current_user_from_token(token: str, auth_repo: AuthRepository) -> Optional[User]:
    payload = verify_simple_jwt(token)
    if not payload:
        return None
    
    user_id = payload.get("sub")
    if not user_id:
        return None
    
    user = auth_repo.get_user_by_id(user_id)
    if user is None:
        return None
    
    return User(**user.model_dump(exclude={'hashed_password'}))