import os
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from time import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


print("CUDA Enabled:", dlib.DLIB_USE_CUDA)
#print(cv2.getBuildInformation())

# Пути к файлам
shape_predictor_path = 'face_model/shape_predictor_68_face_landmarks_GTX.dat'
face_rec_model_path = 'face_model/dlib_face_recognition_resnet_model_v1.dat'
base_path = 'face_bd'
log_file = 'detection_log.txt'

# Проверка существования файлов
if not os.path.exists(shape_predictor_path) or not os.path.exists(face_rec_model_path):
    raise FileNotFoundError("Не найдены файлы моделей.")

# Загрузка ускоренных моделей dlib
sp = dlib.shape_predictor(shape_predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
detector = dlib.get_frontal_face_detector()

def log_detection(name):
    with open(log_file, 'a') as log:
        log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Detected: {name}\\n")

def load_face_descriptors(base_path):
    face_descriptors = []
    faces = os.listdir(base_path)
    for face in faces:
        img = cv2.imread(os.path.join(base_path, face))
        dets = detector(img, 1)
        if len(dets) > 0:
            shape = sp(img, dets[0])
            face_descriptors.append(facerec.compute_face_descriptor(img, shape))
    return face_descriptors, faces

def compare_faces(frame, face_descriptors, faces, threshold=0.6):
    """Асинхронная обработка для ускорения"""
    small_frame = cv2.resize(frame, (320, 240))  # Уменьшаем размер для обработки
    dets = detector(small_frame, 0)
    if len(dets) == 0:
        return None, None, False

    shape = sp(small_frame, dets[0])
    main_descriptor = facerec.compute_face_descriptor(small_frame, shape)
    distances = [distance.euclidean(main_descriptor, fd) for fd in face_descriptors]
    min_dist = min(distances)
    closest_face_idx = distances.index(min_dist)
    return min_dist, faces[closest_face_idx], min_dist <= threshold

def process_webcam(face_descriptors, faces, process_every_n_frames=10):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Уменьшаем разрешение
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Запрашиваем 60 FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Минимальный буфер кадров

    frame_count = 0
    fps_start = time()
    last_detected = None
    executor = ThreadPoolExecutor(max_workers=1)  # Асинхронная обработка

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % process_every_n_frames == 0:
            future = executor.submit(compare_faces, frame, face_descriptors, faces)
            min_dist, closest_face, is_match = future.result()

            if is_match and last_detected != closest_face:
                log_detection(closest_face)
                last_detected = closest_face

        # Вычисляем FPS
        frame_rate = frame_count / (time() - fps_start)
        cv2.putText(frame, f"FPS: {frame_rate:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_descriptors, faces = load_face_descriptors(base_path)
    process_webcam(face_descriptors, faces)
