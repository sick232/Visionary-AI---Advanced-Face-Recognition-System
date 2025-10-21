#!/usr/bin/env python3
"""
Visionary AI - Face Recognition System
Startup Script

This script provides an easy way to start the face recognition system
with proper configuration and error handling.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'opencv-python', 'face-recognition', 
        'numpy', 'Pillow', 'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n💡 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera is available and working")
                return True
            else:
                print("⚠️ Camera is detected but not responding properly")
                return False
        else:
            print("❌ Camera is not available")
            return False
    except Exception as e:
        print(f"❌ Error checking camera: {e}")
        return False

def setup_environment():
    """Setup environment and configuration"""
    print("🔧 Setting up environment...")
    
    # Create faces folder if it doesn't exist
    faces_folder = Path("faces")
    if not faces_folder.exists():
        faces_folder.mkdir()
        print("📁 Created faces folder")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path("config.env.example")
        if env_example.exists():
            env_file.write_text(env_example.read_text())
            print("📝 Created .env file from template")
        else:
            # Create basic .env file
            env_content = """# Visionary AI Configuration
FACES_FOLDER=faces
TOLERANCE=0.6
TARGET_FPS=30
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-change-this
"""
            env_file.write_text(env_content)
            print("📝 Created basic .env file")
    
    print("✅ Environment setup complete")

def start_system():
    """Start the face recognition system"""
    print("🚀 Starting Visionary AI Face Recognition System...")
    print("=" * 60)
    
    try:
        # Import and run the main application
        from app import app, config
        
        print(f"🌐 Web interface: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
        print("📡 API endpoints: http://localhost:5000/api/")
        print("🛑 Press Ctrl+C to stop the system")
        print("=" * 60)
        
        app.run(
            debug=config.FLASK_DEBUG,
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🎯 Visionary AI - Face Recognition System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        return
    
    # Setup environment
    setup_environment()
    
    # Check camera (optional)
    camera_available = check_camera()
    if not camera_available:
        print("⚠️ Camera issues detected. System will still start but camera features may not work.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\n" + "=" * 50)
    
    # Start the system
    start_system()

if __name__ == "__main__":
    main()



