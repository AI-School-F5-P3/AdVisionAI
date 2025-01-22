from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.config import settings
from app.schemas.schemas import User
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException, status
from app.db.session import SessionLocal  # Añadir esta importación
from app.db.models import User as UserModel 
from app.db.session import get_db
from sqlalchemy.orm import Session

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/token",scheme_name="JWT")

credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)

def get_user(user_id: str):
    db = SessionLocal()
    try:
        return db.query(UserModel).filter(UserModel.id == user_id).first()
    finally:
        db.close()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    print("\n=== DEBUG GET_CURRENT_USER ===")
    print(f"Token recibido: {token}")
    print(f"SECRET_KEY: {settings.SECRET_KEY}")
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        print(f"Payload decodificado: {payload}")
        user_id = payload.get("sub")
        print(f"User ID extraído: {user_id}")
        
        if user_id is None:
            raise credentials_exception
            
        user = db.query(UserModel).filter(UserModel.id == user_id).first()
        print(f"Usuario encontrado: {user}")
        
        if user is None:
            raise credentials_exception
            
        return user
    except JWTError as e:
        print(f"Error JWT en get_current_user: {str(e)}")
        raise credentials_exception
    except Exception as e:
        print(f"Error inesperado en get_current_user: {str(e)}")
        raise credentials_exception