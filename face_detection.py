import cv2
import face_recognition
import pandas as pd
import os
from datetime import datetime

# Files to store data
ENCODINGS_FILE = "encodings.npy"
DATA_FILE = "attendance.xlsx"

# Load existing attendance data or create new
if os.path.exists(DATA_FILE):
    attendance_df = pd.read_excel(DATA_FILE)
else:
    attendance_df = pd.DataFrame(columns=["Name", "Date", "Time"])

# Load known face encodings and names if exist
if os.path.exists("known_faces.npy") and os.path.exists("known_names.npy"):
    known_face_encodings = list(np.load("known_faces.npy", allow_pickle=True))
    known_face_names = list(np.load("known_names.npy", allow_pickle=True))
else:
    known_face_encodings = []
    known_face_names = []

video_capture = cv2.VideoCapture(0)

def save_data():
    # Save attendance data to Excel
    attendance_df.to_excel(DATA_FILE, index=False)
    # Save face encodings and names
    np.save("known_faces.npy", known_face_encodings)
    np.save("known_names.npy", known_face_names)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # BGR to RGB

    # Detect faces and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Check if face is recognized
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = None
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
        
        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]
            today_str = datetime.now().strftime("%Y-%m-%d")
            
            # Check today's attendance if already marked
            attendance_today = attendance_df[(attendance_df['Name'] == name) & (attendance_df['Date'] == today_str)]
            if attendance_today.empty:
                new_entry = {
                    "Name": name,
                    "Date": today_str,
                    "Time": datetime.now().strftime("%H:%M:%S")
                }
                attendance_df.loc[len(attendance_df)] = new_entry
                save_data()
        else:
            # Register new user
            cv2.putText(frame, "Registering new user. Enter name and press enter in console.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Automotive Face Detection Attendees System", frame)
            cv2.waitKey(1)
            new_name = input("New user detected. Enter name: ").strip()
            if new_name:
                known_face_encodings.append(face_encoding)
                known_face_names.append(new_name)
                new_entry = {
                    "Name": new_name,
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                    "Time": datetime.now().strftime("%H:%M:%S")
                }
                attendance_df.loc[len(attendance_df)] = new_entry
                save_data()
                name = new_name

        # Scale face locations back to original frame size
        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Automotive Face Detection Attendees System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()