import os
import tkinter as tk
from tkinter import messagebox
from matplotlib import pyplot as plt
from Fface_recognition import load_face_descriptors, compare_faces, get_face_landmarks
from utils import get_file_path
from video_processing import process_video, process_webcam

base_path = 'face_bd'

if not os.path.exists(base_path):
    raise FileNotFoundError(f"The directory for face database was not found: {base_path}")

def main():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    choice = tk.messagebox.askquestion("Select Input", "Do you want to process an image, a video file, or webcam feed?",
                                       type=tk.messagebox.YESNOCANCEL,
                                       icon='question',
                                       default=tk.messagebox.CANCEL,
                                       detail="YES for Image\nNO for Video\nCANCEL for Webcam")

    try:
        face_descriptors, faces = load_face_descriptors(base_path)
        print(f"Loaded {len(faces)} faces from the directory {base_path}.")
    except Exception as e:
        print(f"Error loading face descriptors: {e}")
        exit(1)

    if choice == 'yes':  # Image
        img_path = get_file_path(filetypes=[("Image files", "*.jpg *.jpeg *.png *.svg *.webp"), ("All files", "*.*")])
        if not img_path:
            print("No file selected. Exiting.")
            exit(1)

        print(f"Selected image: {img_path}")

        try:
            min_dist, closest_face, is_match = compare_faces(img_path, face_descriptors, faces)
            print(f'Эвклидово расстояние между дескрипторами: {min_dist}')
            if is_match:
                print(f'Полученное лицо совпадает с лицом на фото: {closest_face}')
            else:
                print('На фотографиях разные люди!')
        except Exception as e:
            print(f"Error comparing faces: {e}")
            exit(1)

        # Визуализация результата
        try:
            test_img, test_landmarks = get_face_landmarks(img_path)
            matched_img, matched_landmarks = get_face_landmarks(os.path.join(base_path, closest_face))

            plt.figure(figsize=(10, 6))

            plt.subplot(1, 2, 1)
            plt.title("Полученное лицо:")
            plt.imshow(test_img)
            for landmarks in test_landmarks:
                x, y = zip(*landmarks)
                plt.scatter(x, y, s=10, c='red', marker='.')

            plt.subplot(1, 2, 2)
            plt.title("Предполагаемое лицо:")
            plt.imshow(matched_img)
            for landmarks in matched_landmarks:
                x, y = zip(*landmarks)
                plt.scatter(x, y, s=10, c='red', marker='.')

            plt.show()
        except Exception as e:
            print(f"Error displaying images: {e}")

    elif choice == 'no':  # Video
        video_path = get_file_path(filetypes=[("Video files", "*.avi *.mp4 *.mkv *.mov"), ("All files", "*.*")])
        if not video_path:
            print("No file selected. Exiting.")
            exit(1)

        process_video(video_path, face_descriptors, faces)

    elif choice == 'cancel':  # Webcam
        process_webcam(face_descriptors, faces)

    else:
        print("Invalid choice. Exiting.")
        exit(1)

if __name__ == "__main__":
    main()
