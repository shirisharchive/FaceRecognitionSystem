import face_encoding   # <-- ADDED
import mysql.connector
import face_recognition
import json
import os
import cv2
import numpy as np
from PIL import Image

# -------------------- Database Encoding -------------------- #
def store_missing_encodings():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='voterdb'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT id, image FROM voterphoto WHERE face_encoding IS NULL")
        results = cursor.fetchall()

        for voter_id, image_blob in results:
            temp_path = f"temp_{voter_id}.jpg"
            with open(temp_path, "wb") as f:
                f.write(image_blob)

            image = face_recognition.load_image_file(temp_path)
            try:
                encoding = face_recognition.face_encodings(image)[0]
                encoding_json = json.dumps(encoding.tolist())
                cursor.execute("UPDATE voterphoto SET face_encoding=%s WHERE id=%s", (encoding_json, voter_id))
                conn.commit()
                print(f"[ENCODED] Voter ID: {voter_id}")
            except IndexError:
                print(f"[NO FACE] Voter ID: {voter_id}")
            finally:
                os.remove(temp_path)

    except mysql.connector.Error as e:
        print(f"[DB ERROR] {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()

# -------------------- Webcam Capture -------------------- #
def capture_and_encode(attempt_number):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return None, None

    print(f"Attempt {attempt_number}/3: Look at the camera...")
    captured_encoding = None
    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow(f"Webcam - Attempt {attempt_number}/3", frame)
        cv2.waitKey(1)

        if face_locations:
            try:
                captured_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                captured_frame = frame
                break
            except IndexError:
                print("[NO FACE] Detected but couldn't encode. Try again.")

    cap.release()
    cv2.destroyAllWindows()
    return captured_frame, captured_encoding

# -------------------- Face Recognition -------------------- #
def recognize_face():
    attempts = 3
    match_found = False

    for attempt in range(1, attempts + 1):
        frame, captured_encoding = capture_and_encode(attempt)
        if captured_encoding is None:
            print(f"[NO FACE] in attempt {attempt}")
            continue

        try:
            conn = mysql.connector.connect(
                host='localhost',
                user='root',
                password='',
                database='voterdb'
            )
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, dob, face_encoding, image FROM voterphoto WHERE face_encoding IS NOT NULL")
            results = cursor.fetchall()

            for voter_id, name, dob, face_encoding_json, photo_blob in results:
                db_encoding = np.array(json.loads(face_encoding_json))
                match = face_recognition.compare_faces([db_encoding], captured_encoding)[0]
                distance = face_recognition.face_distance([db_encoding], captured_encoding)[0]

                if match:
                    print(f"[MATCH] Voter ID: {voter_id}, Name: {name}, DOB: {dob}, Distance: {distance:.4f}")

                    temp_path = f"matched_{voter_id}.jpg"
                    with open(temp_path, "wb") as f:
                        f.write(photo_blob)
                    Image.open(temp_path).show()
                    os.remove(temp_path)

                    match_found = True
                    break

            if match_found:
                break
            else:
                print(f"[NO MATCH] Attempt {attempt}")

        except mysql.connector.Error as e:
            print(f"[DB ERROR] {e}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals() and conn.is_connected():
                conn.close()

    if not match_found:
        print("[EXIT] No match found after all attempts.")

# -------------------- Main -------------------- #
if __name__ == "__main__":

    # <-- ADDED THIS FUNCTION CALL
    face_encoding.store_encodings()

    print("[INFO] Storing missing face encodings...")
    store_missing_encodings()

    print("[INFO] Starting live face recognition...")
    recognize_face()
