from flask import Flask, render_template, Response, jsonify
import cv2
import face_recognition
import numpy as np
import time
from datetime import datetime

from database import FaceDatabase
from config import Config

# Clean replacement for app.py (written to app_fixed.py so original file is preserved).
# Fixes:
# - correct DB id mapping when logging verified faces
# - throttle recognition and cap streamed FPS
# - draw last detection on intermediate frames for smooth UI

config = Config.from_env_file()
config.validate()

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

db = FaceDatabase(config.DATABASE_PATH)

# In-memory caches
known_faces = []
known_names = []
known_ids = []

camera = None
detection_enabled = False

TOLERANCE = config.TOLERANCE
TARGET_FPS = config.TARGET_FPS
FOLDER_PATH = config.FACES_FOLDER


def load_faces_from_database():
    global known_faces, known_names, known_ids
    faces = db.get_all_face_encodings()
    known_faces = [np.array(f['encoding']) for f in faces]
    known_names = [f['name'] for f in faces]
    known_ids = [f['id'] for f in faces]


if len(db.get_all_face_encodings()) == 0:
    try:
        db.load_faces_from_folder(FOLDER_PATH)
    except Exception:
        pass
load_faces_from_database()


def generate_frames():
    global camera, detection_enabled
    last_detection = None
    last_recog = 0.0
    frames = 0
    last_fps_time = time.time()

    recog_interval = 1.0 / max(1, min(TARGET_FPS, 10))

    while True:
        if not detection_enabled or camera is None:
            time.sleep(0.1)
            continue

        ok, frame = camera.read()
        if not ok:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        now = time.time()

        if now - last_recog >= recog_interval:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)

            results = []
            for enc in encodings:
                name = 'Not Verified'
                color = (0, 0, 255)
                conf = 0.0
                verified = False

                if len(known_faces) > 0:
                    dists = face_recognition.face_distance(known_faces, enc)
                    best_idx = int(np.argmin(dists))
                    best_d = float(dists[best_idx])
                    if best_d <= TOLERANCE:
                        verified = True
                        conf = max(0.0, 1.0 - best_d)
                        name = f"Verified ({known_names[best_idx]})"
                        color = (0, 255, 0)
                        db_id = known_ids[best_idx] if best_idx < len(known_ids) else None
                        try:
                            if db_id is not None:
                                db.log_detection(db_id, conf, True)
                        except Exception:
                            pass
                    else:
                        try:
                            db.log_detection(None, 0.0, False)
                        except Exception:
                            pass

                results.append((name, color, conf, verified))

            last_detection = (locations, results)
            last_recog = now

        if last_detection is not None:
            locations, results = last_detection
            for (top, right, bottom, left), (name, color, conf, verified) in zip(locations, results):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if conf > 0:
                    cv2.putText(frame, f"{conf:.2f}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frames += 1
        if time.time() - last_fps_time >= 1.0:
            fps = int(frames / (time.time() - last_fps_time))
            frames = 0
            last_fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if TARGET_FPS and TARGET_FPS > 0:
            interval = 1.0 / TARGET_FPS
            elapsed = time.time() - now
            to_sleep = interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start_detection():
    global camera, detection_enabled
    if camera is None:
        camera = cv2.VideoCapture(config.CAMERA_INDEX)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    detection_enabled = True
    return ('', 204)


@app.route('/stop', methods=['POST'])
def stop_detection():
    global camera, detection_enabled
    detection_enabled = False
    if camera:
        try:
            camera.release()
        except Exception:
            pass
        camera = None
    return ('', 204)


@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'faces_loaded': len(known_faces),
        'detection_active': detection_enabled,
        'camera_available': camera is not None
    })


if __name__ == '__main__':
    print('Starting face detection server...')
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)
