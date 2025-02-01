import os
import pickle
import cv2
import face_recognition
import numpy as np
import cvzone

# Constants
WIDTH, HEIGHT = 640, 480
FRAME_RECOGNITION_INTERVAL = 5
CENTER_OFFSET = 200

def load_face_encodings(file_path):
    try:
        with open(file_path, 'rb') as file:
            encodeListKnownWitIds = pickle.load(file)
            encodeListKnown, studentIds = encodeListKnownWitIds
            print(studentIds)
            return encodeListKnown
    except FileNotFoundError:
        print(f"Error: '{file_path}' file not found.")
        exit()
    except Exception as e:
        print(f"Error loading data from '{file_path}': {e}")
        exit()

def adjust_bounding_box_dimensions(faceLoc, frame_size):
    top, right, bottom, left = faceLoc
    center_x = (left + right + CENTER_OFFSET) // 2
    center_y = (top + bottom + CENTER_OFFSET) // 2

    # Calculate bounding box dimensions based on face size
    face_width = right - left
    face_height = bottom - top
    bbox_width = int(face_width * 1.5)  # You can adjust this multiplier as needed
    bbox_height = int(face_height * 1.5)

    # Calculate bounding box position to keep it centered on the face
    bbox_x = max(0, center_x - bbox_width // 2)
    bbox_y = max(0, center_y - bbox_height // 2)

    # Ensure the bounding box is within the frame boundaries
    bbox_x = min(bbox_x, frame_size[0] - bbox_width)
    bbox_y = min(bbox_y, frame_size[1] - bbox_height)

    bbox = (bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height)
    return bbox

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    encodeListKnown = load_face_encodings("encoding.p")

    frame_count = 0  # Counter to control face recognition frequency

    while True:
        success, img = cap.read()

        if not success:
            print("Error reading frame from webcam")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        if frame_count % FRAME_RECOGNITION_INTERVAL == 0:
            try:
                faceCurFrame = face_recognition.face_locations(imgS)
                encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

                imgBackground = img.copy()  # Initialize imgBackground as a copy of the original frame

                for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                    print("Face location tuple:", faceLoc)
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    matchIndex = np.argmin(faceDis)

                    bbox = adjust_bounding_box_dimensions(faceLoc, img.shape[:2])

                    # Change frame color based on whether the face is known or unknown
                    frame_color = (0, 255, 0)  # Green for known face
                    if not matches[matchIndex]:
                        frame_color = (0, 0, 255)  # Red for unknown face

                    cv2.rectangle(imgBackground, (bbox[0], bbox[1]), (bbox[2], bbox[3]), frame_color, 2)

            except Exception as e:
                print(f"Error in face recognition: {e}")

        cv2.imshow("Face Attendance", imgBackground)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
