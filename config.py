import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configuration class for Visionary AI Face Recognition System"""
    
    # Database settings
    DATABASE_PATH: str = os.getenv('DATABASE_PATH', 'face_database.db')
    
    # Face recognition settings
    FACES_FOLDER: str = os.getenv('FACES_FOLDER', r'E:\face_detection\faces')
    TOLERANCE: float = float(os.getenv('TOLERANCE', '0.6'))
    TARGET_FPS: int = int(os.getenv('TARGET_FPS', '30'))
    
    # Flask settings
    FLASK_HOST: str = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT: int = int(os.getenv('FLASK_PORT', '5000'))
    FLASK_DEBUG: bool = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Security settings
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
    MAX_UPLOAD_SIZE: int = int(os.getenv('MAX_UPLOAD_SIZE', '16777216'))  # 16MB
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
    
    # Camera settings
    CAMERA_INDEX: int = int(os.getenv('CAMERA_INDEX', '0'))
    CAMERA_WIDTH: int = int(os.getenv('CAMERA_WIDTH', '640'))
    CAMERA_HEIGHT: int = int(os.getenv('CAMERA_HEIGHT', '480'))
    
    # Analytics settings
    ANALYTICS_RETENTION_DAYS: int = int(os.getenv('ANALYTICS_RETENTION_DAYS', '30'))
    LOG_DETECTIONS: bool = os.getenv('LOG_DETECTIONS', 'True').lower() == 'true'
    
    # Performance settings
    FRAME_SKIP_RATIO: int = int(os.getenv('FRAME_SKIP_RATIO', '2'))  # Process every Nth frame
    FACE_DETECTION_MODEL: str = os.getenv('FACE_DETECTION_MODEL', 'hog')  # 'hog' or 'cnn'
    
    @classmethod
    def from_env_file(cls, env_file: str = '.env') -> 'Config':
        """Load configuration from .env file"""
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        return cls()
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'database_path': self.DATABASE_PATH,
            'faces_folder': self.FACES_FOLDER,
            'tolerance': self.TOLERANCE,
            'target_fps': self.TARGET_FPS,
            'flask_host': self.FLASK_HOST,
            'flask_port': self.FLASK_PORT,
            'flask_debug': self.FLASK_DEBUG,
            'camera_index': self.CAMERA_INDEX,
            'camera_width': self.CAMERA_WIDTH,
            'camera_height': self.CAMERA_HEIGHT,
            'analytics_retention_days': self.ANALYTICS_RETENTION_DAYS,
            'log_detections': self.LOG_DETECTIONS,
            'frame_skip_ratio': self.FRAME_SKIP_RATIO,
            'face_detection_model': self.FACE_DETECTION_MODEL
        }
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        if not os.path.exists(self.FACES_FOLDER):
            errors.append(f"Faces folder does not exist: {self.FACES_FOLDER}")
        
        if not (0.1 <= self.TOLERANCE <= 1.0):
            errors.append(f"Tolerance must be between 0.1 and 1.0, got: {self.TOLERANCE}")
        
        if not (1 <= self.TARGET_FPS <= 120):
            errors.append(f"Target FPS must be between 1 and 120, got: {self.TARGET_FPS}")
        
        if not (1 <= self.FRAME_SKIP_RATIO <= 10):
            errors.append(f"Frame skip ratio must be between 1 and 10, got: {self.FRAME_SKIP_RATIO}")
        
        if self.FACE_DETECTION_MODEL not in ['hog', 'cnn']:
            errors.append(f"Face detection model must be 'hog' or 'cnn', got: {self.FACE_DETECTION_MODEL}")
        
        if errors:
            print("❌ Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("✅ Configuration validation passed")
        return True
