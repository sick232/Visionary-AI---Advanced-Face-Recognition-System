from flask import Flask, render_template, Response, jsonify
import cv2
import face_recognition
import numpy as np
import time
from datetime import datetime

from database import FaceDatabase
from config import Config

# Minimal, clean app.py implementing fixes requested by user.
# - Correct DB id mapping when logging verified faces
# - Throttle recognition (<=10 Hz) and cap output FPS to TARGET_FPS
# - Keep last detection drawn on intermediate frames for smooth UI

config = Config.from_env_file()
config.validate()

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

db = FaceDatabase(config.DATABASE_PATH)

# In-memory caches
known_faces = []  # list of numpy arrays (encodings)
known_names = []  # parallel list of names
known_ids = []    # parallel list of DB ids

# Camera and control flags
camera = None
detection_enabled = False

# Runtime settings
TOLERANCE = config.TOLERANCE
TARGET_FPS = config.TARGET_FPS
FOLDER_PATH = config.FACES_FOLDER


def load_faces_from_database():
    """Load face encodings and keep DB ids for correct logging."""
    global known_faces, known_names, known_ids
    faces = db.get_all_face_encodings()
    known_faces = [np.array(f['encoding']) for f in faces]
    known_names = [f['name'] for f in faces]
    known_ids = [f['id'] for f in faces]


# If DB empty, attempt to populate from folder once
if len(db.get_all_face_encodings()) == 0:
    try:
        db.load_faces_from_folder(FOLDER_PATH)
    except Exception:
        pass
load_faces_from_database()


def generate_frames():
    """MJPEG generator. Throttles heavy recognition and caps output FPS.

    Recognition is limited to max 10 Hz (or TARGET_FPS if lower).
    The last detection result is drawn on intermediate frames for visual continuity.
    """
    global camera, detection_enabled

    last_detection = None  # (locations, results)
    last_recog = 0.0
    frames = 0
    last_fps_time = time.time()

    # Limit recognition frequency to at most 10 Hz for CPU savings
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

        # Run recognition on a throttled interval
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

        # Draw the last detection for smoother UI
        if last_detection is not None:
            locations, results = last_detection
            for (top, right, bottom, left), (name, color, conf, verified) in zip(locations, results):
                # scale back up (we processed at 1/4 size)
                top *= 4; right *= 4; bottom *= 4; left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if conf > 0:
                    cv2.putText(frame, f"{conf:.2f}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS overlay (measured)
        frames += 1
        if time.time() - last_fps_time >= 1.0:
            fps = int(frames / (time.time() - last_fps_time))
            frames = 0
            last_fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Throttle to target FPS to avoid CPU burn and jitter
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


if __name__ == '__main__':
    print('Starting face detection server...')
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)


# Load config
config = Config.from_env_file()
config.validate()

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# Initialize DB
db = FaceDatabase(config.DATABASE_PATH)

# In-memory caches (populated from DB)
known_faces = []  # list of numpy arrays
known_names = []  # list of strings
known_ids = []    # list of DB ids (integers)

# Camera and control flags
camera = None
detection_enabled = False

# Runtime settings (can be updated via API)
TOLERANCE = config.TOLERANCE
TARGET_FPS = config.TARGET_FPS
FOLDER_PATH = config.FACES_FOLDER


def load_faces_from_database():
    """Load face encodings and keep DB ids for correct logging."""
    global known_faces, known_names, known_ids
    faces = db.get_all_face_encodings()
    known_faces = [np.array(f["encoding"]) for f in faces]
    known_names = [f["name"] for f in faces]
    known_ids = [f["id"] for f in faces]


# Ensure some faces are available in DB (attempt to load from folder once)
if len(db.get_all_face_encodings()) == 0:
    try:
        db.load_faces_from_folder(FOLDER_PATH)
    except Exception:
        pass
load_faces_from_database()


def generate_frames():
    """Yield MJPEG frames. Throttle heavy recognition and cap output FPS.

    - Recognition is limited to max 10 Hz (or TARGET_FPS if lower).
    - The last detection result is drawn on intermediate frames for visual
      continuity.
    """
    global camera, detection_enabled

    last_detection = None  # (locations, results) where results is list of tuples
    last_recog = 0.0
    frames = 0
    last_fps_time = time.time()

    # Limit recognition frequency to at most 10 Hz for CPU savings
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

        # Run recognition on a throttled interval
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
                    # face_distance returns lower = better match
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
                            # don't let DB errors kill the stream
                            pass
                    else:
                        # unverified detection
                        try:
                            db.log_detection(None, 0.0, False)
                        except Exception:
                            pass

                results.append((name, color, conf, verified))

            last_detection = (locations, results)
            last_recog = now

        # Draw the last detection for smoother UI
        if last_detection is not None:
            locations, results = last_detection
            for (top, right, bottom, left), (name, color, conf, verified) in zip(locations, results):
                # scale back up (we processed at 1/4 size)
                top *= 4; right *= 4; bottom *= 4; left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if conf > 0:
                    cv2.putText(frame, f"{conf:.2f}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS overlay (measured)
        frames += 1
        if time.time() - last_fps_time >= 1.0:
            fps = int(frames / (time.time() - last_fps_time))
            frames = 0
            last_fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Throttle to target FPS to avoid CPU burn and jitter
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
    # return 204 so simple form submissions don't reload the UI
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
from flask import Flask, render_template, Response, jsonify, request
import cv2
import face_recognition
import numpy as np
import time
import os
from database import FaceDatabase
from config import Config
from flask import Flask, render_template, Response, jsonify, request
import cv2
import face_recognition
import numpy as np
import time
import os
from database import FaceDatabase
from config import Config
from datetime import datetime

# Load config
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
    """Load face encodings and map DB ids into memory."""
    global known_faces, known_names, known_ids
    faces = db.get_all_face_encodings()
    known_faces = [np.array(f['encoding']) for f in faces]
    known_names = [f['name'] for f in faces]
    known_ids = [f['id'] for f in faces]


# Ensure some faces are available
if len(db.get_all_face_encodings()) == 0:
    db.load_faces_from_folder(FOLDER_PATH)
load_faces_from_database()


def generate_frames():
    """MJPEG generator. Throttles heavy recognition and caps stream FPS."""
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
            break

        frame = cv2.flip(frame, 1)
        now = time.time()

        # run recognition at limited rate
        if now - last_recog >= recog_interval:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)

            results = []
            for enc in encs:
                name = 'Not Verified'
                color = (0, 0, 255)
                conf = 0.0
                verified = False
                if len(known_faces) > 0:
                    dists = face_recognition.face_distance(known_faces, enc)
                    best = int(np.argmin(dists))
                    best_d = float(dists[best])
                    if best_d <= TOLERANCE:
                        verified = True
                        conf = max(0.0, 1.0 - best_d)
                        name = f"Verified ({known_names[best]})"
                        color = (0, 255, 0)
                        db_id = known_ids[best] if best < len(known_ids) else None
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

            last_detection = (locs, results)
            last_recog = now

        # draw last detection
        if last_detection is not None:
            locs, results = last_detection
            for (top, right, bottom, left), (name, color, conf, verified) in zip(locs, results):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # fps overlay
        frames += 1
        if time.time() - last_fps_time >= 1.0:
            fps = int(frames / (time.time() - last_fps_time))
            frames = 0
            last_fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # throttle to target fps
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
        camera.release()
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

db = FaceDatabase(config.DATABASE_PATH)

# Globals
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
    db.load_faces_from_folder(FOLDER_PATH)
load_faces_from_database()


def generate_frames():
    global camera, detection_enabled
    last_detection = None
    last_process = 0.0
    frame_count = 0
    last_fps_time = time.time()

    recognition_interval = 1.0 / max(1, min(TARGET_FPS, 10))

    while True:
        if not detection_enabled or camera is None:
            time.sleep(0.1)
            continue

        ok, frame = camera.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        now = time.time()

        if now - last_process >= recognition_interval:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locations)

            names = []
            for enc in encs:
                name = 'Not Verified'
                color = (0, 0, 255)
                conf = 0.0
                verified = False
                if len(known_faces) > 0:
                    dists = face_recognition.face_distance(known_faces, enc)
                    best = int(np.argmin(dists))
                    best_d = float(dists[best])
                    if best_d <= TOLERANCE:
                        conf = max(0.0, 1.0 - best_d)
                        verified = True
                        db_id = known_ids[best] if best < len(known_ids) else None
                        name = f"Verified ({known_names[best]})"
                        color = (0, 255, 0)
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
                names.append((name, color, conf, verified))

            last_detection = (locations, names)
            last_process = now

        if last_detection:
            locations, names = last_detection
            for (top, right, bottom, left), (name, color, conf, verified) in zip(locations, names):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        frame_count += 1
        if time.time() - last_fps_time >= 1.0:
            fps = int(frame_count / (time.time() - last_fps_time))
            frame_count = 0
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
        camera.release()
        camera = None
    return ('', 204)


@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'faces_loaded': len(known_faces),
        'detection_active': detection_enabled,
    })


if __name__ == '__main__':
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'faces_loaded': len(known_faces),
        'detection_active': detection_enabled,
        'camera_available': camera is not None
    })

if __name__ == "__main__":
    print("🚀 Starting Visionary AI Face Recognition System...")
    print(f"📊 Database initialized with {len(known_faces)} faces")
    print(f"⚙️ Configuration loaded: {config.to_dict()}")
    print(f"🌐 Web interface available at: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print("📡 API endpoints available at: http://localhost:5000/api/")
    print("🔒 Security features: Rate limiting, input validation, file size limits")
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)
from flask import Flask, render_template, Response, jsonify, request
import cv2
load_faces_from_database()

def generate_frames():
    global detection_enabled, camera, analytics_data, TOLERANCE, TARGET_FPS

    frame_count = 0
    last_fps_time = time.time()
    last_process_time = 0.0
    last_detection = None  # store tuples of (face_locations, face_names)

    # Determine minimum interval between heavy recognition passes
    recognition_interval = 1.0 / max(1, min(TARGET_FPS, 10))  # limit recognition to <=10Hz

    while True:
        if not detection_enabled or camera is None:
            time.sleep(0.1)
            continue

        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        now = time.time()

        # Only run expensive face detection occasionally (throttle)
        if now - last_process_time >= recognition_interval:
            # Resize for faster face recognition
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                name = "Not Verified"
                color = (0, 0, 255)
                confidence = 0.0
                verified = False

                if len(known_faces) > 0:
                    # compute distances and find best match
                    face_distances = face_recognition.face_distance(known_faces, face_encoding)
                    best_match_index = int(np.argmin(face_distances))
                    best_distance = float(face_distances[best_match_index])

                    if best_distance <= TOLERANCE:
                        confidence = max(0.0, 1.0 - best_distance)
                        verified = True
                        # Use DB id mapping if available
                        db_id = known_ids[best_match_index] if best_match_index < len(known_ids) else None
                        name = f"Verified ({known_names[best_match_index]})"
                        color = (0, 255, 0)

                        # Log detection with correct DB id
                        try:
                            db.log_detection(db_id, confidence, True)
                        except Exception:
                            # avoid crashing streaming loop on DB errors
                            pass
                        analytics_data['verified_count'] += 1
                    else:
                        try:
                            from flask import Flask, render_template, Response, jsonify, request
                            import cv2
                            import face_recognition
                            import os
                            import numpy as np
                            import time
                            import json
                            from database import FaceDatabase
                            from config import Config
                            from datetime import datetime
                            from functools import wraps
                            from collections import defaultdict, deque

                            # Load configuration
                            config = Config.from_env_file()
                            config.validate()

                            app = Flask(__name__)
                            app.secret_key = config.SECRET_KEY

                            # Initialize database
                            db = FaceDatabase(config.DATABASE_PATH)

                            # Rate limiting
                            request_counts = defaultdict(lambda: deque())

                            def rate_limit(max_requests=60, window=60):
                                def decorator(f):
                                    @wraps(f)
                                    def decorated_function(*args, **kwargs):
                                        client_ip = request.remote_addr
                                        now = time.time()
                                        while request_counts[client_ip] and request_counts[client_ip][0] <= now - window:
                                            request_counts[client_ip].popleft()
                                        if len(request_counts[client_ip]) >= max_requests:
                                            return jsonify({'error': 'Rate limit exceeded'}), 429
                                        request_counts[client_ip].append(now)
                                        return f(*args, **kwargs)
                                    return decorated_function
                                return decorator

                            # Globals
                            known_faces = []
                            known_names = []
                            known_ids = []
                            camera = None
                            detection_enabled = False
                            analytics_data = {
                                'total_detections': 0,
                                'verified_count': 0,
                                'session_start': datetime.now(),
                                'recent_detections': []
                            }

                            # Config values
                            TOLERANCE = config.TOLERANCE
                            TARGET_FPS = config.TARGET_FPS
                            FOLDER_PATH = config.FACES_FOLDER

                            # Load faces
                            def load_faces_from_database():
                                global known_faces, known_names, known_ids
                                faces_data = db.get_all_face_encodings()
                                known_faces = [face['encoding'] for face in faces_data]
                                known_names = [face['name'] for face in faces_data]
                                known_ids = [face['id'] for face in faces_data]
                                print(f"✅ Loaded {len(known_faces)} faces from database")

                            # If DB empty, load from folder
                            if len(db.get_all_face_encodings()) == 0:
                                print("📁 Loading faces from folder...")
                                db.load_faces_from_folder(FOLDER_PATH)

                            load_faces_from_database()


                            def generate_frames():
                                global detection_enabled, camera, analytics_data, TOLERANCE, TARGET_FPS

                                frame_count = 0
                                last_fps_time = time.time()
                                last_process_time = 0.0
                                last_detection = None  # (face_locations, face_names)

                                # Cap recognition frequency to avoid CPU spikes
                                recognition_interval = 1.0 / max(1, min(TARGET_FPS, 10))  # max 10Hz recognition

                                while True:
                                    if not detection_enabled or camera is None:
                                        time.sleep(0.1)
                                        continue

                                    success, frame = camera.read()
                                    if not success:
                                        break

                                    frame = cv2.flip(frame, 1)
                                    now = time.time()

                                    # Run heavy recognition only at limited rate
                                    if now - last_process_time >= recognition_interval:
                                        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                                        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                                        face_locations = face_recognition.face_locations(rgb_small_frame)
                                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                                        face_names = []
                                        for face_encoding in face_encodings:
                                            name = "Not Verified"
                                            color = (0, 0, 255)
                                            confidence = 0.0
                                            verified = False

                                            if len(known_faces) > 0:
                                                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                                                best_match_index = int(np.argmin(face_distances))
                                                best_distance = float(face_distances[best_match_index])

                                                if best_distance <= TOLERANCE:
                                                    confidence = max(0.0, 1.0 - best_distance)
                                                    verified = True
                                                    db_id = known_ids[best_match_index] if best_match_index < len(known_ids) else None
                                                    name = f"Verified ({known_names[best_match_index]})"
                                                    color = (0, 255, 0)
                                                    try:
                                                        db.log_detection(db_id, confidence, True)
                                                    except Exception:
                                                        pass
                                                    analytics_data['verified_count'] += 1
                                                else:
                                                    try:
                                                        db.log_detection(None, 0.0, False)
                                                    except Exception:
                                                        pass
                                            else:
                                                try:
                                                    db.log_detection(None, 0.0, False)
                                                except Exception:
                                                    pass

                                            analytics_data['total_detections'] += 1
                                            face_names.append((name, color, confidence, verified))

                                        last_detection = (face_locations, face_names)
                                        last_process_time = now

                                    # Draw last detection results for smoother experience
                                    if last_detection is not None:
                                        face_locations, face_names = last_detection
                                        for (top, right, bottom, left), (name, color, confidence, verified) in zip(face_locations, face_names):
                                            top *= 4
                                            right *= 4
                                            bottom *= 4
                                            left *= 4
                                            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                                            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                            if confidence > 0:
                                                cv2.putText(frame, f"Conf: {confidence:.2f}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                    # FPS overlay
                                    frame_count += 1
                                    if time.time() - last_fps_time >= 1.0:
                                        fps = frame_count / (time.time() - last_fps_time)
                                        frame_count = 0
                                        last_fps_time = time.time()
                                    else:
                                        fps = None

                                    if fps is not None:
                                        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                                    # Throttle overall frame rate to TARGET_FPS for smoother streaming
                                    if TARGET_FPS and TARGET_FPS > 0:
                                        frame_interval = 1.0 / TARGET_FPS
                                        elapsed = time.time() - now
                                        to_sleep = frame_interval - elapsed
                                        if to_sleep > 0:
                                            time.sleep(to_sleep)

                                    ret, buffer = cv2.imencode('.jpg', frame)
                                    frame_bytes = buffer.tobytes()

                                    yield (b'--frame\r\n'
                                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


                            @app.route('/')
                            def index():
                                return render_template('index.html')


                            @app.route('/video')
                            def video():
                                return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


                            @app.route('/start', methods=['POST'])
                            @rate_limit(max_requests=10, window=60)
                            def start_detection():
                                global detection_enabled, camera
                                if camera is None:
                                    camera = cv2.VideoCapture(config.CAMERA_INDEX)
                                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
                                detection_enabled = True
                                return ('', 204)


                            @app.route('/stop', methods=['POST'])
                            def stop_detection():
                                global detection_enabled, camera
                                detection_enabled = False
                                if camera:
                                    camera.release()
                                    camera = None
                                return ('', 204)


                            # API Endpoints
                            @app.route('/api/analytics')
                            @rate_limit(max_requests=30, window=60)
                            def get_analytics():
                                summary = db.get_analytics_summary()
                                return jsonify({
                                    'session_data': analytics_data,
                                    'database_summary': summary,
                                    'current_settings': {
                                        'tolerance': TOLERANCE,
                                        'target_fps': TARGET_FPS,
                                        'total_faces': len(known_faces)
                                    }
                                })


                            @app.route('/api/settings', methods=['POST'])
                            @rate_limit(max_requests=20, window=60)
                            def update_settings():
                                global TOLERANCE, TARGET_FPS
                                data = request.get_json()
                                if 'tolerance' in data:
                                    TOLERANCE = float(data['tolerance'])
                                    db.update_setting('tolerance', TOLERANCE)
                                if 'target_fps' in data:
                                    TARGET_FPS = int(data['target_fps'])
                                    db.update_setting('target_fps', TARGET_FPS)
                                return jsonify({'status': 'success', 'settings': {'tolerance': TOLERANCE, 'target_fps': TARGET_FPS}})


                            @app.route('/api/faces')
                            def get_faces():
                                faces = db.get_all_face_encodings()
                                return jsonify([{'id': f['id'], 'name': f['name'], 'image_path': f['image_path']} for f in faces])


                            @app.route('/api/faces', methods=['POST'])
                            @rate_limit(max_requests=10, window=60)
                            def add_face():
                                if 'image' not in request.files:
                                    return jsonify({'error': 'No image provided'}), 400
                                file = request.files['image']
                                name = request.form.get('name', 'Unknown')
                                if file.filename == '':
                                    return jsonify({'error': 'No file selected'}), 400
                                file.seek(0, 2)
                                file_size = file.tell()
                                file.seek(0)
                                if file_size > config.MAX_UPLOAD_SIZE:
                                    return jsonify({'error': f'File too large. Max size: {config.MAX_UPLOAD_SIZE} bytes'}), 400
                                try:
                                    temp_path = f"temp_{int(time.time())}.jpg"
                                    file.save(temp_path)
                                    image = face_recognition.load_image_file(temp_path)
                                    encodings = face_recognition.face_encodings(image)
                                    if len(encodings) > 0:
                                        face_id = db.add_face_encoding(name, encodings[0], temp_path)
                                        load_faces_from_database()
                                        os.remove(temp_path)
                                        return jsonify({'status': 'success', 'face_id': face_id, 'message': f'Face added for {name}'})
                                    else:
                                        os.remove(temp_path)
                                        return jsonify({'error': 'No face detected in image'}), 400
                                except Exception as e:
                                    return jsonify({'error': str(e)}), 500


                            @app.route('/api/faces/<int:face_id>', methods=['DELETE'])
                            def delete_face(face_id):
                                try:
                                    db.delete_face(face_id)
                                    load_faces_from_database()
                                    return jsonify({'status': 'success', 'message': 'Face deleted'})
                                except Exception as e:
                                    return jsonify({'error': str(e)}), 500


                            @app.route('/api/detections')
                            def get_detections():
                                days = request.args.get('days', 7, type=int)
                                stats = db.get_detection_stats(days)
                                return jsonify(stats)


                            @app.route('/api/export')
                            def export_data():
                                faces = db.get_all_face_encodings()
                                stats = db.get_detection_stats(30)
                                summary = db.get_analytics_summary()
                                export_data = {
                                    'export_time': datetime.now().isoformat(),
                                    'faces': [{'id': f['id'], 'name': f['name'], 'image_path': f['image_path']} for f in faces],
                                    'detection_stats': stats,
                                    'summary': summary,
                                    'session_data': analytics_data
                                }
                                return jsonify(export_data)


                            @app.route('/api/health')
                            def health_check():
                                return jsonify({
                                    'status': 'healthy',
                                    'timestamp': datetime.now().isoformat(),
                                    'faces_loaded': len(known_faces),
                                    'detection_active': detection_enabled,
                                    'camera_available': camera is not None
                                })


                            if __name__ == "__main__":
                                print("🚀 Starting Visionary AI Face Recognition System...")
                                print(f"📊 Database initialized with {len(known_faces)} faces")
                                print(f"⚙️ Configuration loaded: {config.to_dict()}")
                                print(f"🌐 Web interface available at: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
                                print("📡 API endpoints available at: http://localhost:5000/api/")
                                print("🔒 Security features: Rate limiting, input validation, file size limits")
                                app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)
    """Delete a face"""
    try:
        db.delete_face(face_id)
        load_faces_from_database()
        return jsonify({'status': 'success', 'message': 'Face deleted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detections')
def get_detections():
    """Get recent detection logs"""
    days = request.args.get('days', 7, type=int)
    stats = db.get_detection_stats(days)
    return jsonify(stats)


@app.route('/api/export')
def export_data():
    """Export all data as JSON"""
    faces = db.get_all_face_encodings()
    stats = db.get_detection_stats(30)
    summary = db.get_analytics_summary()
    
    export_data = {
        'export_time': datetime.now().isoformat(),
        'faces': [{'id': f['id'], 'name': f['name'], 'image_path': f['image_path']} for f in faces],
        'detection_stats': stats,
        'summary': summary,
        'session_data': analytics_data
    }
    
    return jsonify(export_data)


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'faces_loaded': len(known_faces),
        'detection_active': detection_enabled,
        'camera_available': camera is not None
    })


if __name__ == "__main__":
    print("🚀 Starting Visionary AI Face Recognition System...")
    print(f"📊 Database initialized with {len(known_faces)} faces")
    print(f"⚙️ Configuration loaded: {config.to_dict()}")
    print(f"🌐 Web interface available at: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print("📡 API endpoints available at: http://localhost:5000/api/")
    print("🔒 Security features: Rate limiting, input validation, file size limits")
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)
