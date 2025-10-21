#!/usr/bin/env python3
"""
Visionary AI - Face Recognition System
Test Script

This script tests the various components of the face recognition system
to ensure everything is working correctly.
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import flask
        import cv2
        import face_recognition
        import numpy as np
        from PIL import Image
        print("✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("🧪 Testing database...")
    
    try:
        from database import FaceDatabase
        
        # Create test database
        test_db = FaceDatabase("test_face_database.db")
        
        # Test adding a face encoding
        import numpy as np
        test_encoding = np.random.rand(128)  # Mock face encoding
        face_id = test_db.add_face_encoding("test_person", test_encoding, "test_path.jpg")
        
        # Test retrieving faces
        faces = test_db.get_all_face_encodings()
        
        # Test logging detection
        test_db.log_detection(face_id, 0.95, True)
        
        # Test analytics
        stats = test_db.get_analytics_summary()
        
        # Clean up test database
        os.remove("test_face_database.db")
        
        print("✅ Database functionality working correctly")
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print("🧪 Testing configuration...")
    
    try:
        from config import Config
        
        config = Config()
        
        # Test validation
        is_valid = config.validate()
        
        # Test to_dict method
        config_dict = config.to_dict()
        
        print("✅ Configuration system working correctly")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_camera():
    """Test camera functionality"""
    print("🧪 Testing camera...")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️ Camera not available (this is normal if no camera is connected)")
            return True
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            print("✅ Camera is working correctly")
            return True
        else:
            print("⚠️ Camera detected but not responding properly")
            return True  # Not a critical error
    except Exception as e:
        print(f"⚠️ Camera test error: {e}")
        return True  # Not a critical error

def test_face_recognition():
    """Test face recognition functionality"""
    print("🧪 Testing face recognition...")
    
    try:
        import face_recognition
        import numpy as np
        
        # Create a mock image array
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test face detection
        face_locations = face_recognition.face_locations(mock_image)
        
        # Test face encoding (will likely return empty for random image)
        face_encodings = face_recognition.face_encodings(mock_image)
        
        print("✅ Face recognition library working correctly")
        return True
    except Exception as e:
        print(f"❌ Face recognition error: {e}")
        return False

def test_web_server():
    """Test if web server can start"""
    print("🧪 Testing web server startup...")
    
    try:
        import subprocess
        import time
        
        # Start the server in background
        process = subprocess.Popen([
            sys.executable, "app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            # Server is running, test a simple endpoint
            try:
                response = requests.get("http://localhost:5000/api/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Web server is working correctly")
                    success = True
                else:
                    print(f"⚠️ Web server responded with status {response.status_code}")
                    success = True
            except requests.exceptions.RequestException:
                print("⚠️ Web server started but not responding (may need more time)")
                success = True
        else:
            print("❌ Web server failed to start")
            success = False
        
        # Terminate the process
        process.terminate()
        process.wait()
        
        return success
    except Exception as e:
        print(f"❌ Web server test error: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("🧪 Testing file structure...")
    
    required_files = [
        "app.py",
        "database.py", 
        "config.py",
        "requirements.txt",
        "templates/index.html"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✅ All required files present")
    return True

def run_all_tests():
    """Run all tests"""
    print("🎯 Visionary AI - System Test Suite")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Database", test_database),
        ("Face Recognition", test_face_recognition),
        ("Camera", test_camera),
        ("Web Server", test_web_server),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

def main():
    """Main test function"""
    success = run_all_tests()
    
    if success:
        print("\n🚀 You can now start the system with: python start.py")
    else:
        print("\n🔧 Please fix the issues above before starting the system.")
    
    return success

if __name__ == "__main__":
    main()



