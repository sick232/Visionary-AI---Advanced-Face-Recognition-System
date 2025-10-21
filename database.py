import sqlite3
import json
import numpy as np
import os
from datetime import datetime
import face_recognition

class FaceDatabase:
    def __init__(self, db_path="face_database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Face encodings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_encodings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                encoding TEXT NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Detection logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER,
                confidence REAL,
                verified BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_data BLOB,
                FOREIGN KEY (face_id) REFERENCES face_encodings (id)
            )
        ''')
        
        # Analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                total_detections INTEGER DEFAULT 0,
                verified_detections INTEGER DEFAULT 0,
                false_positives INTEGER DEFAULT 0,
                avg_confidence REAL,
                session_duration INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_face_encoding(self, name, encoding, image_path=None):
        """Add a new face encoding to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert numpy array to JSON string
        encoding_json = json.dumps(encoding.tolist())
        
        cursor.execute('''
            INSERT INTO face_encodings (name, encoding, image_path)
            VALUES (?, ?, ?)
        ''', (name, encoding_json, image_path))
        
        face_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return face_id
    
    def get_all_face_encodings(self):
        """Get all active face encodings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, encoding, image_path FROM face_encodings 
            WHERE is_active = 1
        ''')
        
        results = []
        for row in cursor.fetchall():
            face_id, name, encoding_json, image_path = row
            encoding = np.array(json.loads(encoding_json))
            results.append({
                'id': face_id,
                'name': name,
                'encoding': encoding,
                'image_path': image_path
            })
        
        conn.close()
        return results
    
    def log_detection(self, face_id, confidence, verified, image_data=None):
        """Log a detection event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detection_logs (face_id, confidence, verified, image_data)
            VALUES (?, ?, ?, ?)
        ''', (face_id, confidence, verified, image_data))
        
        conn.commit()
        conn.close()
    
    def get_detection_stats(self, days=7):
        """Get detection statistics for the last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_detections,
                SUM(CASE WHEN verified = 1 THEN 1 ELSE 0 END) as verified_detections,
                AVG(confidence) as avg_confidence
            FROM detection_logs 
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        '''.format(days))
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def update_setting(self, key, value):
        """Update a setting"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO settings (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (key, str(value)))
        
        conn.commit()
        conn.close()
    
    def get_setting(self, key, default=None):
        """Get a setting value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else default
    
    def load_faces_from_folder(self, folder_path):
        """Load all faces from a folder and store in database"""
        if not os.path.exists(folder_path):
            print(f"❌ Error: Folder '{folder_path}' does not exist!")
            return []
        
        loaded_faces = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(folder_path, filename)
                name = os.path.splitext(filename)[0]
                
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if len(encodings) > 0:
                        face_id = self.add_face_encoding(name, encodings[0], image_path)
                        loaded_faces.append({
                            'id': face_id,
                            'name': name,
                            'encoding': encodings[0],
                            'image_path': image_path
                        })
                        print(f"✅ Loaded face: {name}")
                    else:
                        print(f"⚠️ No face detected in {filename}, skipping...")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
        
        print(f"✅ Total loaded faces: {len(loaded_faces)}")
        return loaded_faces
    
    def delete_face(self, face_id):
        """Soft delete a face (mark as inactive)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE face_encodings SET is_active = 0, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (face_id,))
        
        conn.commit()
        conn.close()
    
    def get_analytics_summary(self):
        """Get overall analytics summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total faces
        cursor.execute('SELECT COUNT(*) FROM face_encodings WHERE is_active = 1')
        total_faces = cursor.fetchone()[0]
        
        # Total detections today
        cursor.execute('''
            SELECT COUNT(*) FROM detection_logs 
            WHERE DATE(timestamp) = DATE('now')
        ''')
        today_detections = cursor.fetchone()[0]
        
        # Verified detections today
        cursor.execute('''
            SELECT COUNT(*) FROM detection_logs 
            WHERE DATE(timestamp) = DATE('now') AND verified = 1
        ''')
        today_verified = cursor.fetchone()[0]
        
        # Average confidence
        cursor.execute('SELECT AVG(confidence) FROM detection_logs WHERE DATE(timestamp) = DATE("now")')
        avg_confidence = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_faces': total_faces,
            'today_detections': today_detections,
            'today_verified': today_verified,
            'today_accuracy': (today_verified / today_detections * 100) if today_detections > 0 else 0,
            'avg_confidence': round(avg_confidence, 3)
        }
