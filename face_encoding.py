import mysql.connector
import face_recognition
import json
import os

def store_encodings():
    try:
        # Connect to the database
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',  # Update with your database password
            database='voterdb'
        )
        cursor = conn.cursor()

        # Fetch images for voters without face encodings
        query = 'SELECT id, image FROM voterphoto WHERE face_encoding IS NULL'
        cursor.execute(query)
        results = cursor.fetchall()

        for row in results:
            voter_id, image_blob = row

            # Save image temporarily to generate face encoding
            temp_image_path = f"temp_{voter_id}.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(image_blob)

            # Generate face encoding
            image = face_recognition.load_image_file(temp_image_path)
            try:
                face_encoding = face_recognition.face_encodings(image)[0]
                face_encoding_json = json.dumps(face_encoding.tolist())  # Convert encoding to JSON

                # Update the database
                update_query = 'UPDATE voterphoto SET face_encoding = %s WHERE id = %s'
                cursor.execute(update_query, (face_encoding_json, voter_id))
                conn.commit()
                print(f"Stored encoding for Voter ID: {voter_id}")
            except IndexError:
                print(f"No face detected for Voter ID: {voter_id}")
            finally:
                os.remove(temp_image_path)

    except mysql.connector.Error as e:
        print(f"Database error: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and conn.is_connected():
            conn.close()

# Run the script
store_encodings()