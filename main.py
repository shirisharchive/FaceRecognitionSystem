import mysql.connector
import cv2
import numpy as np
import json
import os
from PIL import Image
import face_recognition




def capture_image(attempt_number):
    # Open the default camera (camera index 0)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return None

    print(f"Attempt {attempt_number}/3: Press 's' to capture an image or 'q' to quit.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture an image.")
            break

        # Find faces in the current frame
        face_locations = face_recognition.face_locations(frame)

        # Draw rectangles around the faces
        for face_location in face_locations:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (15, 255, 15), 1)

        # Display the frame
        cv2.imshow(f"Webcam - Attempt {attempt_number}/3", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # If 's' is pressed, save the image
            image_path = f"captured_image_{attempt_number}.jpg"  # Create unique image name
            cv2.imwrite(image_path, frame)  # Save the image
            print(f"Image {attempt_number} saved as {image_path}")
            cap.release()
            cv2.destroyAllWindows()
            return image_path

        elif key == ord('q'):  # If 'q' is pressed, exit
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return None


def attempt_face_recognition():
    attempts = 3
    match_found = False

    for attempt_number in range(1, attempts + 1):
        captured_image_path = capture_image(attempt_number)

        if captured_image_path is None:
            print("No image captured. Exiting...")
            exit()

        try:
            # Load and encode the captured image
            captured_image = face_recognition.load_image_file(captured_image_path)
            captured_image_encoding = face_recognition.face_encodings(captured_image)[0]
        except IndexError:
            print(f"No face detected in image from attempt {attempt_number}.")
            continue  # Move to the next attempt

        try:
            # Connect to the database
            conn = mysql.connector.connect(
                host='localhost',
                user='root',
                password='',  # Update with your database password
                database='voterdb'
            )
            cursor = conn.cursor()

            # Fetch all face encodings from the database
            query = 'SELECT id, name, dob, face_encoding FROM voterphoto'
            cursor.execute(query)
            results = cursor.fetchall()

            for row in results:
                voter_id, name, dob, face_encoding_json = row

                if face_encoding_json:
                    # Convert JSON encoding back to numpy array
                    db_face_encoding = np.array(json.loads(face_encoding_json))
                    

                    # Compare captured image encoding with database encoding
                    match = face_recognition.compare_faces([db_face_encoding], captured_image_encoding)
                    distance = face_recognition.face_distance([db_face_encoding], captured_image_encoding)[0]

                    if match[0]:
                        print(f"Match found! Voter ID: {voter_id}, Name: {name}, DOB: {dob} (Distance: {distance:.4f})")
                        matched_id = voter_id
                        match_found = True
                        break  # Stop searching after a match is found

            if match_found:
                # Fetch the photo of the matched voter ID
                photo_query = "SELECT image FROM voterphoto WHERE id = %s"
                cursor.execute(photo_query, (matched_id,))
                photo_result = cursor.fetchone()

                if photo_result and photo_result[0]:
                    # Save the fetched image as a temporary file
                    temp_image_path = f"matched_{matched_id}.jpg"
                    with open(temp_image_path, "wb") as f:
                        f.write(photo_result[0])

                    print(f"Photo of matched voter ID {matched_id} saved as {temp_image_path}.")

                    # Optionally, display the image
                    try:
                        img = Image.open(temp_image_path)
                        img.show()
                    except Exception as e:
                        print(f"Error displaying image: {e}")
                    finally:
                        # Remove the temporary file after displaying
                        os.remove(temp_image_path)

                break  # Exit the loop after a match is found

            if not match_found:
                print(f"No match found in the database for attempt {attempt_number}.")

        except mysql.connector.Error as e:
            print(f"Database error: {e}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals() and conn.is_connected():
                conn.close()

    if not match_found:
        print("No match found after 3 attempts. Exiting...")


# Start the face recognition attempt process
attempt_face_recognition()