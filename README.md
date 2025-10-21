# 🚀 Visionary AI - Advanced Face Recognition System

A modern, feature-rich face recognition system built with Flask, OpenCV, and face_recognition library. This system provides real-time face detection, recognition, analytics, and a beautiful web interface.

## ✨ Features

### 🎯 Core Features
- **Real-time Face Recognition**: Live video feed with instant face detection and recognition
- **Database Integration**: SQLite database for storing face encodings and detection logs
- **REST API**: Complete API for face management and analytics
- **Modern Web UI**: Beautiful, responsive interface with real-time analytics
- **Configuration Management**: Environment-based configuration system

### 📊 Analytics & Monitoring
- Real-time detection statistics
- Confidence score tracking
- Session analytics
- Detection history
- Performance metrics (FPS tracking)
- Data export functionality

### 🔧 Advanced Controls
- Adjustable recognition tolerance
- FPS control
- Face database management
- Settings persistence
- Health monitoring

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Webcam/Camera
- Windows/Linux/macOS

### Quick Setup

1. **Clone or download the project**
```bash
git clone <repository-url>
cd face_detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure the system**
```bash
# Copy the example configuration
cp config.env.example .env

# Edit .env file with your settings
# Most importantly, set your FACES_FOLDER path
```

4. **Add face images**
- Place face images in the `faces/` folder
- Supported formats: JPG, PNG, JPEG
- Each image should contain one clear face
- Name files with the person's name (e.g., `john.jpg`)

5. **Run the application**
```bash
python app.py
```

6. **Access the web interface**
- Open your browser to `http://localhost:5000`
- Click "Start Detection" to begin face recognition

## 📁 Project Structure

```
face_detection/
├── app.py                 # Main Flask application
├── main.py               # Standalone OpenCV script
├── database.py           # Database management
├── config.py             # Configuration system
├── requirements.txt      # Python dependencies
├── config.env.example    # Configuration template
├── faces/                # Face images folder
│   ├── person1.jpg
│   ├── person2.png
│   └── ...
├── templates/
│   └── index.html        # Web interface
└── face_database.db      # SQLite database (created automatically)
```

## 🌐 API Endpoints

### Core Endpoints
- `GET /` - Web interface
- `GET /video` - Video stream
- `POST /start` - Start detection
- `POST /stop` - Stop detection

### API Endpoints
- `GET /api/analytics` - Get analytics data
- `POST /api/settings` - Update settings
- `GET /api/faces` - List all faces
- `POST /api/faces` - Add new face
- `DELETE /api/faces/<id>` - Delete face
- `GET /api/detections` - Get detection logs
- `GET /api/export` - Export all data
- `GET /api/health` - Health check

### Example API Usage

**Get analytics:**
```bash
curl http://localhost:5000/api/analytics
```

**Update tolerance setting:**
```bash
curl -X POST http://localhost:5000/api/settings \
  -H "Content-Type: application/json" \
  -d '{"tolerance": 0.5}'
```

**Add a new face:**
```bash
curl -X POST http://localhost:5000/api/faces \
  -F "image=@person.jpg" \
  -F "name=John Doe"
```

## ⚙️ Configuration

The system uses environment variables for configuration. Copy `config.env.example` to `.env` and modify:

### Key Settings
- `FACES_FOLDER`: Path to folder containing face images
- `TOLERANCE`: Recognition sensitivity (0.1-1.0, lower = more strict)
- `TARGET_FPS`: Target frames per second
- `CAMERA_INDEX`: Camera device index (usually 0)
- `FLASK_PORT`: Web server port

### Performance Tuning
- `FRAME_SKIP_RATIO`: Process every Nth frame (higher = faster)
- `FACE_DETECTION_MODEL`: 'hog' (faster) or 'cnn' (more accurate)

## 🎮 Usage

### Web Interface
1. **Start Detection**: Click "Start Detection" to begin
2. **Adjust Settings**: Use sliders to modify tolerance and FPS
3. **View Analytics**: Monitor real-time statistics
4. **Export Data**: Download analytics as JSON

### Standalone Script
Run `main.py` for a simple OpenCV-based face recognition:
```bash
python main.py
```

Controls:
- `q`: Quit
- `f`: Toggle flip modes

## 📊 Analytics Dashboard

The web interface provides:
- **Total Detections**: Count of all face detections
- **Verified Faces**: Count of successful recognitions
- **Current FPS**: Real-time performance
- **Accuracy Rate**: Percentage of verified detections
- **Recent Detections**: Last 5 detection events
- **Export/Clear**: Data management tools

## 🔒 Security Features

- Rate limiting on API endpoints
- Input validation
- Secure file upload handling
- Database integrity checks
- Error handling and logging

## 🚀 Deployment

### Production Deployment
1. Set `FLASK_DEBUG=False` in `.env`
2. Use a production WSGI server (e.g., Gunicorn)
3. Configure reverse proxy (Nginx)
4. Set up SSL certificates
5. Configure firewall rules

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## 🐛 Troubleshooting

### Common Issues

**Camera not working:**
- Check camera permissions
- Try different `CAMERA_INDEX` values
- Ensure camera is not used by other applications

**Poor recognition accuracy:**
- Lower the `TOLERANCE` value
- Use higher quality face images
- Ensure good lighting conditions
- Use 'cnn' model for better accuracy

**Performance issues:**
- Increase `FRAME_SKIP_RATIO`
- Use 'hog' detection model
- Reduce camera resolution
- Close other applications

**Database errors:**
- Check file permissions
- Ensure disk space is available
- Delete `face_database.db` to reset

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) library
- [OpenCV](https://opencv.org/) for computer vision
- [Flask](https://flask.palletsprojects.com/) web framework
- [SQLite](https://www.sqlite.org/) database



---

**Built by Piyush**

