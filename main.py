import os
import sys
import cv2
import face_recognition
import math
import numpy as np


def face_accuracy(face_distance, face_match_threshold=0.6):  # Used to determine accuracy of face match
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecog:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):  # Encodes Images and Stores Names
        faces = 'C:/Users/Adam McMullan/Desktop/Projects/AI Face Recognition/faces'  # Chose folder with faces for model to check for

        for image in os.listdir(faces):
            try:
                face_image = face_recognition.load_image_file(f'{faces}/{image}')
                face_encodings = face_recognition.face_encodings(face_image)

                if face_encodings:  # Only append if face encodings are found
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(image)
                else:
                    print(f"No faces found in {image}. Skipping...")

            except Exception as e:
                print(f"Error processing image {image}: {e}")

        print(self.known_face_names)

    def run_recognition(self):  # Main Function to run face recognition
        video_cap = cv2.VideoCapture(0)

        # Break if can't get access to camera
        if not video_cap.isOpened():
            sys.exit('Camera Source Not Found!')

        # Outer While loop to access each frame
        while True:
            ret, frame = video_cap.read()

            if ret and self.process_current_frame:  # Ensure frame is valid before processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown User'
                    confidence = '?'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match = np.argmin(face_distances)

                    if matches[best_match]:
                        name = self.known_face_names[best_match]
                        confidence = face_accuracy(face_distances[best_match])

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame  # Toggle frame processing

            # Annotations on video
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
                cv2.rectangle(frame, (left, top - 35), (right, top), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)
                                
            if cv2.waitKey(1) == ord('q'): # Q to close window
                cv2.destroyAllWindows('Face Recognition')
                video_cap.release()
                



if __name__ == '__main__':
    fr = FaceRecog()
    fr.run_recognition()

    
