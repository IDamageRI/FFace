import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from scipy.spatial import distance
import dlib
import cv2
from matplotlib import pyplot as plt

# Paths for pre-trained models
shape_predictor_path = 'face_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'face_model/dlib_face_recognition_resnet_model_v1.dat'
base_path = 'face_bd'

# Check if necessary files and directories exist
if not os.path.exists(base_path):
    raise FileNotFoundError(f"The directory for face database was not found: {base_path}")
if not os.path.exists(shape_predictor_path):
    raise FileNotFoundError(f"Shape predictor model not found at {shape_predictor_path}")
if not os.path.exists(face_rec_model_path):
    raise FileNotFoundError(f"Face recognition model not found at {face_rec_model_path}")

# Load models
sp = dlib.shape_predictor(shape_predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
detector = dlib.get_frontal_face_detector()

def load_image(img_path):
    try:
        # Используем OpenCV для загрузки изображения в формате BGR
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found at {img_path}")
        return img
    except Exception as e:
        raise ValueError(f"Error opening image {img_path}: {e}")

def load_face_descriptors(base_path):
    face_descriptors = []
    faces = os.listdir(base_path)

    if not faces:
        raise ValueError(f"No faces found in the directory {base_path}")

    for face in faces:
        img_path = os.path.join(base_path, face)
        try:
            img = load_image(img_path)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue

        dets = detector(img, 1)
        if len(dets) == 0:
            print(f"No faces detected in the image {img_path}")
            continue

        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptors.append(facerec.compute_face_descriptor(img, shape))
    return face_descriptors, faces

def compare_faces(img_or_frame, face_descriptors, faces, threshold=0.7):
    if isinstance(img_or_frame, str):
        img = load_image(img_or_frame)
    else:
        img = img_or_frame
    dets = detector(img, 1)
    if len(dets) == 0:
        raise ValueError(f"No faces detected in the image")

    for k, d in enumerate(dets):
        shape = sp(img, d)
        main_descriptor = facerec.compute_face_descriptor(img, shape)

    distances = [distance.euclidean(main_descriptor, fd) for fd in face_descriptors]
    min_dist = min(distances)
    closest_face_idx = distances.index(min_dist)

    return min_dist, faces[closest_face_idx], min_dist <= threshold

def get_face_landmarks(img_or_frame):
    if isinstance(img_or_frame, str):
        img = load_image(img_or_frame)
    else:
        img = img_or_frame
    dets = detector(img, 1)
    if len(dets) == 0:
        raise ValueError(f"No faces detected in the image")

    landmarks = []
    for k, d in enumerate(dets):
        shape = sp(img, d)
        landmarks.append([(point.x, point.y) for point in shape.parts()])
    return img, landmarks

def process_video(video_path, face_descriptors, faces):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            try:
                min_dist, closest_face, is_match = compare_faces(frame, face_descriptors, faces)
                if is_match:
                    print(f'Полученное лицо совпадает с лицом на фото: {closest_face}')
                else:
                    print('На фотографиях разные люди!')
            except Exception as e:
                print(f"Error processing frame: {e}")

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_webcam(face_descriptors, faces):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            min_dist, closest_face, is_match = compare_faces(frame, face_descriptors, faces)
            closest_face_img_path = os.path.join(base_path, closest_face)
            closest_face_img, _ = get_face_landmarks(closest_face_img_path)

            if is_match:
                print(f'Полученное лицо совпадает с лицом на фото: {closest_face}')
            else:
                print('На фотографиях разные люди!')

            img, landmarks = get_face_landmarks(frame)

            for landmarks_set in landmarks:
                for (x, y) in landmarks_set:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            closest_face_img = cv2.resize(closest_face_img, (frame.shape[1], frame.shape[0]))
            combined_frame = cv2.hconcat([frame, closest_face_img])
            cv2.imshow('Webcam and Most Similar Face', combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Error processing frame: {e}")

    cap.release()
    cv2.destroyAllWindows()

def select_input():
    window = tk.Tk()
    window.title("Select Input")
    window.geometry("300x200")

    style = ttk.Style()
    style.theme_use('clam')

    def choose_image():
        window.destroy()
        process_image()

    def choose_video():
        window.destroy()
        video_path = get_file_path(filetypes=[("Video files", "*.avi *.mp4 *.mkv *.mov"), ("All files", "*.*")])
        if video_path:
            process_video(video_path, face_descriptors, faces)

    def choose_webcam():
        window.destroy()
        process_webcam(face_descriptors, faces)

    image_button = ttk.Button(window, text="Process Image", command=choose_image)
    video_button = ttk.Button(window, text="Process Video", command=choose_video)
    webcam_button = ttk.Button(window, text="Process Webcam", command=choose_webcam)

    image_button.pack(pady=10)
    video_button.pack(pady=10)
    webcam_button.pack(pady=10)

    window.mainloop()

def get_file_path(filetypes=None):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=filetypes)
    return file_path

def process_image():
    file_path = get_file_path(filetypes=[("Image files", "*.jpg *.jpeg *.png *.svg *.webp"), ("All files", "*.*")])
    if file_path:
        try:
            min_dist, closest_face, is_match = compare_faces(file_path, face_descriptors, faces)
            print(f'Эвклидово расстояние между дескрипторами: {min_dist}')
            if is_match:
                print(f'Полученное лицо совпадает с лицом на фото: {closest_face}')
            else:
                print('На фотографиях разные люди!')
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return

        try:
            test_img, test_landmarks = get_face_landmarks(file_path)
            matched_img, matched_landmarks = get_face_landmarks(os.path.join(base_path, closest_face))

            plt.figure(figsize=(10, 6))

            plt.subplot(1, 2, 1)
            plt.title("Полученное лицо:")
            # Конвертируем изображение в RGB перед отображением
            plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
            for landmarks in test_landmarks:
                x, y = zip(*landmarks)
                plt.scatter(x, y, s=10, c='red', marker='.')

            plt.subplot(1, 2, 2)
            plt.title("Предполагаемое лицо:")
            # Конвертируем изображение в RGB перед отображением
            plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
            for landmarks in matched_landmarks:
                x, y = zip(*landmarks)
                plt.scatter(x, y, s=10, c='red', marker='.')

            plt.show()
        except Exception as e:
            print(f"Error displaying images: {e}")

if __name__ == "__main__":
    try:
        face_descriptors, faces = load_face_descriptors(base_path)
        print(f"Loaded {len(faces)} faces from the directory {base_path}.")
    except Exception as e:
        print(f"Error loading face descriptors: {e}")
        exit(1)

    select_input()
