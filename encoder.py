import cv2
import face_recognition
import pickle
import os


def load_images_and_labels(folder_path):
    path_list = os.listdir(folder_path)
    img_list = []
    student_ids = []

    for path in path_list:
        img_list.append(cv2.imread(os.path.join(folder_path, path)))
        student_ids.append(os.path.splitext(path)[0])

    return img_list, student_ids


def find_encodings(img_list):
    encode_list = []

    for img in img_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)

    return encode_list


def main():
    folder_path = 'images'

    try:
        img_list, student_ids = load_images_and_labels(folder_path)
    except Exception as e:
        print(f"Error loading images and labels: {e}")
        return

    print("Encoding Started")

    try:
        encode_list_known = find_encodings(img_list)
    except Exception as e:
        print(f"Error finding face encodings: {e}")
        return

    encode_list_known_with_ids = [encode_list_known, student_ids]

    print("Encoding Finished")

    try:
        with open("encoding.p", 'wb') as file:
            pickle.dump(encode_list_known_with_ids, file)
        print("Pickle file saved")
    except Exception as e:
        print(f"Error saving pickle file: {e}")


if __name__ == "__main__":
    main()
