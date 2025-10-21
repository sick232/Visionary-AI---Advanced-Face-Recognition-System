import cv2
import face_recognition
import os
import numpy as np
import time

# Folder path containing face images
FOLDER_PATH = r"E:\projects\face_detection\faces"

# Step 1: Load and Encode All Faces from Folder
known_faces = []
known_names = []

if not os.path.exists(FOLDER_PATH):
    print(f" ❌ Error: Folder '{FOLDER_PATH}' does not exist!")
    exit()

for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(FOLDER_PATH, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_faces.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])
        else:
            print(f" ⚠️ Warning: No face detected in '{filename}', skipping...")

print(f" ✅ Loaded {len(known_faces)} verified faces.")

# Step 2: Start Webcam
cap = cv2.VideoCapture(0)

process_this_frame = True  # skip alternate frames
flip_mode = 1  # default: horizontal flip
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔹 Apply current flip mode
    frame = cv2.flip(frame, flip_mode)

    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:  # only process every alternate frame
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            face_distances = face_recognition.face_distance(known_faces, face_encoding)

            name = "Not Verified"
            color = (0, 0, 255)  # Red

            if True in matches:
                best_match_index = np.argmin(face_distances)
                name = f"Verified {known_names[best_match_index]}"
                color = (0, 255, 0)  # Green

            face_names.append((name, color))

    process_this_frame = not process_this_frame  # skip alternate frame

    # Draw results back on original frame
    for (top, right, bottom, left), (name, color) in zip(face_locations, face_names):
        # Scale back up since we resized
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 🔹 Calculate and show FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show output
    cv2.imshow("Face Verification (Inverted)", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        # cycle between horizontal (1), vertical (0), 180° (-1)
        flip_mode = {1: 0, 0: -1, -1: 1}[flip_mode]

cap.release()
cv2.destroyAllWindows()
