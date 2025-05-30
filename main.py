import cv2
import face_recognition
import sqlite3
import numpy as np
import pickle
import os
import json
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import argparse
import logging
from pathlib import Path
import pygame

# Configuration
CONFIG = {
    'db_path': 'faces.db',
    'snapshot_dir': 'snapshots',
    'logs_dir': 'logs',
    'confidence_threshold': 0.6,
    'frame_skip': 2,  # Process every nth frame
    'min_face_size': (50, 50),
    'max_faces_per_frame': 10,
    'alert_cooldown': 30,  # seconds between alerts for same face
    'face_detection_model': 'hog',  # 'hog' or 'cnn'
    'enable_audio_alerts': True,
    'enable_motion_detection': True,
    'auto_cleanup_days': 30
}

# Setup directories
for directory in [CONFIG['snapshot_dir'], CONFIG['logs_dir']]:
    os.makedirs(directory, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CONFIG['logs_dir'], 'face_recognition.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioManager:
    """Handles audio alerts and notifications"""
    def __init__(self):
        self.enabled = CONFIG['enable_audio_alerts']
        if self.enabled:
            try:
                pygame.mixer.init()
                self.alert_sound = None
                # You can add sound files here
                # self.alert_sound = pygame.mixer.Sound('alert.wav')
            except:
                logger.warning("Audio system not available")
                self.enabled = False
    
    def play_alert(self, alert_type="detection"):
        if not self.enabled or not self.alert_sound:
            return
        try:
            self.alert_sound.play()
        except:
            pass

class FaceDatabase:
    """Enhanced database manager with better tracking and analytics"""
    def __init__(self, db_path=None):
        self.db_path = db_path or CONFIG['db_path']
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._create_tables()
        self._cleanup_old_data()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY,
                encoding BLOB,
                first_seen TEXT,
                last_seen TEXT,
                seen_count INTEGER DEFAULT 1,
                name TEXT,
                notes TEXT,
                confidence_avg REAL DEFAULT 0.0,
                is_authorized BOOLEAN DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY,
                face_id INTEGER,
                timestamp TEXT,
                confidence REAL,
                location TEXT,
                snapshot_path TEXT,
                FOREIGN KEY (face_id) REFERENCES faces (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        self.conn.commit()

    def add_face(self, encoding, confidence=0.0, location=None, snapshot_path=None):
        with self.lock:
            now = datetime.utcnow().isoformat()
            data = pickle.dumps(encoding)
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO faces (encoding, first_seen, last_seen, confidence_avg) 
                VALUES (?, ?, ?, ?)
            ''', (data, now, now, confidence))
            
            face_id = cursor.lastrowid
            
            # Add detection record
            cursor.execute('''
                INSERT INTO detections (face_id, timestamp, confidence, location, snapshot_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (face_id, now, confidence, json.dumps(location) if location else None, snapshot_path))
            
            self.conn.commit()
            return face_id

    def update_face(self, face_id, confidence=0.0, location=None, snapshot_path=None):
        with self.lock:
            now = datetime.utcnow().isoformat()
            cursor = self.conn.cursor()
            
            # Update face record
            cursor.execute('''
                UPDATE faces 
                SET last_seen = ?, 
                    seen_count = seen_count + 1,
                    confidence_avg = (confidence_avg * (seen_count - 1) + ?) / seen_count
                WHERE id = ?
            ''', (now, confidence, face_id))
            
            # Add detection record
            cursor.execute('''
                INSERT INTO detections (face_id, timestamp, confidence, location, snapshot_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (face_id, now, confidence, json.dumps(location) if location else None, snapshot_path))
            
            self.conn.commit()

    def load_all(self):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT id, encoding, name, is_authorized FROM faces')
            rows = cursor.fetchall()
            known_ids, known_encodings, known_names, authorized_status = [], [], [], []
            
            for row in rows:
                fid, data, name, is_auth = row
                try:
                    encoding = pickle.loads(data)
                    known_ids.append(fid)
                    known_encodings.append(encoding)
                    known_names.append(name or f"Unknown_{fid}")
                    authorized_status.append(bool(is_auth))
                except:
                    logger.warning(f"Failed to load face encoding for ID {fid}")
            
            return known_ids, known_encodings, known_names, authorized_status

    def get_face_stats(self, face_id):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT f.name, f.seen_count, f.first_seen, f.last_seen, f.confidence_avg, f.is_authorized,
                       COUNT(d.id) as detection_count
                FROM faces f
                LEFT JOIN detections d ON f.id = d.face_id
                WHERE f.id = ?
                GROUP BY f.id
            ''', (face_id,))
            return cursor.fetchone()

    def set_face_name(self, face_id, name):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('UPDATE faces SET name = ? WHERE id = ?', (name, face_id))
            self.conn.commit()

    def set_authorization(self, face_id, authorized):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('UPDATE faces SET is_authorized = ? WHERE id = ?', (authorized, face_id))
            self.conn.commit()

    def _cleanup_old_data(self):
        """Remove old detection records based on cleanup policy"""
        if CONFIG['auto_cleanup_days'] <= 0:
            return
        
        cutoff_date = (datetime.utcnow() - timedelta(days=CONFIG['auto_cleanup_days'])).isoformat()
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM detections WHERE timestamp < ?', (cutoff_date,))
            deleted = cursor.rowcount
            self.conn.commit()
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old detection records")

    def close(self):
        self.conn.close()

class MotionDetector:
    """Motion detection to optimize face recognition processing"""
    def __init__(self, threshold=25, min_area=500):
        self.threshold = threshold
        self.min_area = min_area
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.motion_detected = False

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        fg_mask = self.background_subtractor.apply(gray)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > self.min_area]
        self.motion_detected = len(motion_areas) > 0
        
        return self.motion_detected

class FaceRecognitionSystem:
    """Main face recognition system with enhanced features"""
    def __init__(self):
        self.db = FaceDatabase()
        self.audio = AudioManager()
        self.motion_detector = MotionDetector() if CONFIG['enable_motion_detection'] else None
        
        # Load known faces
        self.known_ids, self.known_encodings, self.known_names, self.authorized_status = self.db.load_all()
        
        # Alert management
        self.last_alerts = {}
        self.frame_count = 0
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'new_faces': 0,
            'authorized_access': 0,
            'unauthorized_access': 0,
            'start_time': datetime.utcnow()
        }
        
        logger.info(f"System initialized with {len(self.known_ids)} known faces")

    def should_process_frame(self, frame):
        """Determine if frame should be processed based on motion detection and frame skipping"""
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % CONFIG['frame_skip'] != 0:
            return False
        
        # Check motion if enabled
        if self.motion_detector and not self.motion_detector.detect_motion(frame):
            return False
        
        return True

    def save_face_snapshot(self, frame, location, face_id, detection_type="unknown"):
        """Save enhanced face snapshot with metadata"""
        try:
            top, right, bottom, left = location
            
            # Add padding around face
            padding = 20
            h, w = frame.shape[:2]
            top = max(0, top - padding)
            bottom = min(h, bottom + padding)
            left = max(0, left - padding)
            right = min(w, right + padding)
            
            face_image = frame[top:bottom, left:right]
            
            if face_image.size == 0:
                return None
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f"face_{face_id}_{detection_type}_{timestamp}.jpg"
            filepath = os.path.join(CONFIG['snapshot_dir'], filename)
            
            # Add timestamp overlay
            cv2.putText(face_image, datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), 
                       (10, face_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imwrite(filepath, face_image)
            return filepath
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None

    def should_alert(self, face_id):
        """Check if enough time has passed since last alert for this face"""
        now = time.time()
        if face_id not in self.last_alerts:
            self.last_alerts[face_id] = now
            return True
        
        if now - self.last_alerts[face_id] > CONFIG['alert_cooldown']:
            self.last_alerts[face_id] = now
            return True
        
        return False

    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        if not self.should_process_frame(frame):
            return frame
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = small_frame[:, :, ::-1]
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small, model=CONFIG['face_detection_model'])
        
        if len(face_locations) > CONFIG['max_faces_per_frame']:
            face_locations = face_locations[:CONFIG['max_faces_per_frame']]
        
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        
        for location, encoding in zip(face_locations, face_encodings):
            # Scale back up face locations
            top, right, bottom, left = [coord * 4 for coord in location]
            
            # Check face size
            face_width = right - left
            face_height = bottom - top
            if face_width < CONFIG['min_face_size'][0] or face_height < CONFIG['min_face_size'][1]:
                continue
            
            # Compare with known faces
            if len(self.known_encodings) > 0:
                face_distances = face_recognition.face_distance(self.known_encodings, encoding)
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]
                
                if confidence >= CONFIG['confidence_threshold']:
                    # Known face detected
                    face_id = self.known_ids[best_match_index]
                    name = self.known_names[best_match_index]
                    is_authorized = self.authorized_status[best_match_index]
                    
                    self.stats['total_detections'] += 1
                    if is_authorized:
                        self.stats['authorized_access'] += 1
                    else:
                        self.stats['unauthorized_access'] += 1
                    
                    # Save snapshot and update database
                    snapshot_path = self.save_face_snapshot(frame, (top, right, bottom, left), 
                                                          face_id, "known")
                    self.db.update_face(face_id, confidence, (top, right, bottom, left), snapshot_path)
                    
                    # Alert logic
                    if self.should_alert(face_id):
                        alert_type = "AUTHORIZED" if is_authorized else "UNAUTHORIZED"
                        logger.info(f"[{alert_type}] {name} (ID: {face_id}) detected - Confidence: {confidence:.2f}")
                        self.audio.play_alert("authorized" if is_authorized else "unauthorized")
                    
                    # Draw rectangle (green for authorized, red for unauthorized)
                    color = (0, 255, 0) if is_authorized else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Add label
                    label = f"{name} ({confidence:.2f})"
                    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    continue
            
            # New face detected
            face_id = self.db.add_face(encoding, 0.0, (top, right, bottom, left))
            self.known_ids.append(face_id)
            self.known_encodings.append(encoding)
            self.known_names.append(f"Unknown_{face_id}")
            self.authorized_status.append(False)
            
            self.stats['new_faces'] += 1
            
            # Save snapshot
            snapshot_path = self.save_face_snapshot(frame, (top, right, bottom, left), face_id, "new")
            
            logger.info(f"[NEW FACE] ID {face_id} added to database")
            self.audio.play_alert("new_face")
            
            # Draw blue rectangle for new face
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, f"NEW: {face_id}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return frame

    def display_stats(self, frame):
        """Display system statistics on frame"""
        stats_text = [
            f"Known Faces: {len(self.known_ids)}",
            f"Total Detections: {self.stats['total_detections']}",
            f"New Faces: {self.stats['new_faces']}",
            f"Authorized: {self.stats['authorized_access']}",
            f"Unauthorized: {self.stats['unauthorized_access']}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def run(self, camera_index=0, display_stats=True):
        """Main system loop"""
        video_capture = cv2.VideoCapture(camera_index)
        
        # Set camera properties for better performance
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Starting face recognition system. Press 'q' to quit, 's' to show stats")
        show_stats = display_stats
        
        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                # Process frame
                frame = self.process_frame(frame)
                
                # Display statistics if enabled
                if show_stats:
                    frame = self.display_stats(frame)
                
                # Show frame
                cv2.imshow('Advanced Face Recognition System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    show_stats = not show_stats
                elif key == ord('r'):
                    # Reload known faces
                    self.known_ids, self.known_encodings, self.known_names, self.authorized_status = self.db.load_all()
                    logger.info("Reloaded known faces from database")
        
        except KeyboardInterrupt:
            logger.info("System interrupted by user")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
            self.db.close()
            logger.info("System shutdown complete")

def manage_faces():
    """Simple face management utility"""
    db = FaceDatabase()
    
    while True:
        print("\n=== Face Management ===")
        print("1. List all faces")
        print("2. Set face name")
        print("3. Set authorization status")
        print("4. View face statistics")
        print("5. Exit")
        
        choice = input("Enter choice: ").strip()
        
        if choice == '1':
            known_ids, _, known_names, authorized_status = db.load_all()
            print(f"\n{'ID':<5} {'Name':<20} {'Authorized':<10}")
            print("-" * 35)
            for i, (fid, name, auth) in enumerate(zip(known_ids, known_names, authorized_status)):
                print(f"{fid:<5} {name:<20} {'Yes' if auth else 'No':<10}")
        
        elif choice == '2':
            face_id = input("Enter face ID: ").strip()
            name = input("Enter name: ").strip()
            try:
                db.set_face_name(int(face_id), name)
                print("Name updated successfully")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '3':
            face_id = input("Enter face ID: ").strip()
            auth = input("Authorize? (y/n): ").strip().lower() == 'y'
            try:
                db.set_authorization(int(face_id), auth)
                print("Authorization updated successfully")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '4':
            face_id = input("Enter face ID: ").strip()
            try:
                stats = db.get_face_stats(int(face_id))
                if stats:
                    name, seen_count, first_seen, last_seen, confidence_avg, is_auth, detection_count = stats
                    print(f"\nFace Statistics:")
                    print(f"Name: {name or 'Unknown'}")
                    print(f"Seen Count: {seen_count}")
                    print(f"First Seen: {first_seen}")
                    print(f"Last Seen: {last_seen}")
                    print(f"Average Confidence: {confidence_avg:.2f}")
                    print(f"Authorized: {'Yes' if is_auth else 'No'}")
                    print(f"Total Detections: {detection_count}")
                else:
                    print("Face not found")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '5':
            break
    
    db.close()

def main():
    parser = argparse.ArgumentParser(description='Advanced Face Recognition System')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--manage', action='store_true', help='Launch face management utility')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load custom config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            CONFIG.update(custom_config)
    
    if args.manage:
        manage_faces()
    else:
        system = FaceRecognitionSystem()
        system.run(camera_index=args.camera)

if __name__ == '__main__':
    main()